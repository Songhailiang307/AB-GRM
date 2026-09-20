#!/usr/bin/env Rscript
# Summarize repeated cross-validation predictions and plot GS metrics.
# Usage: Rscript plot_gs_metrics.R [results_directory] [dataset_prefix]
# Dependencies: data.table, dplyr, ggplot2.
# Each input file must contain held-out predictions for one model and trait.
# Pool test folds within each repeat; then summarize across repeats.
# This differs from averaging fold-level correlations in run_gblup_cv.R.
# Outputs include GS_metrics_by_repeat.txt and GS_metrics_summary.txt.

library(data.table)
library(dplyr)
library(ggplot2)

############################################################
# 1. Input and output settings
#
# Usage:
# Rscript plot_gs_metrics.R [results_directory] [dataset_prefix]
# The results directory defaults to the current working directory.
############################################################

args <- commandArgs(trailingOnly = TRUE)
input_dir <- if (length(args) >= 1) args[1] else "."

if (!dir.exists(input_dir)) {
  stop("Input directory does not exist: ", input_dir)
}

output_dir <- file.path(input_dir, "GS_model_comparison_plots")
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# Read only model-specific prediction files matching the pattern below.
#
# Accepted filename examples:
#   dataset_GBLUP_predictions_20rep_5fold.txt
#   dataset_10_5_ABG_GBLUP_predictions_20rep_5fold.txt
#   dataset_500_250_ABG_GBLUP_predictions_20rep_5fold.txt
#
# The following files are excluded:
#   sommer_all_models_predictions_20rep_5fold.txt
#   sommer_all_models_accuracy_20rep_5fold.txt
#   dataset_GBLUP_accuracy_20rep_5fold.txt
#   dataset_500_250_ABG_GBLUP_accuracy_20rep_5fold.txt
#   GS_metrics_by_repeat.txt / GS_metrics_summary.txt
#
# The id, y, pred, rep and fold columns are required to recalculate metrics
# for each repeat. Supply prediction files, not fold-level accuracy files.
dataset_prefix <- if (length(args) >= 2) args[2] else "dataset"
# Restrict prefixes to characters that are safe in the filename regex.
if (!grepl("^[A-Za-z0-9_-]+$", dataset_prefix)) {
  stop("Dataset prefix must contain only letters, digits, underscores or hyphens.")
}

valid_prediction_pattern <- paste0(
  "^", dataset_prefix,
  "_(?:GBLUP|[0-9]+_[0-9]+_ABG_GBLUP)",
  "_predictions_20rep_5fold\\.txt$"
)

candidate_files <- list.files(
  path = input_dir,
  pattern = valid_prediction_pattern,
  full.names = TRUE,
  recursive = FALSE,
  ignore.case = TRUE
)

if (length(candidate_files) == 0) {
  stop(
    "No valid single-model prediction files were found in: ",
    normalizePath(input_dir, mustWork = FALSE),
    "\nExpected filenames such as:\n",
    "  dataset_GBLUP_predictions_20rep_5fold.txt\n",
    "  dataset_10_5_ABG_GBLUP_predictions_20rep_5fold.txt"
  )
}

message(
  "Prediction files selected: ",
  paste(basename(candidate_files), collapse = ", ")
)

############################################################
# 2. Helpers: normalize column names and identify and order models
############################################################

standardize_column_names <- function(dat) {
  old <- names(dat)
  key <- tolower(gsub("[^a-z0-9]+", "", old))
  aliases <- list(
    id    = c("id", "iid", "sample", "sampleid", "individual", "individualid"),
    y     = c("y", "obs", "observed", "phenotype", "true", "actual"),
    pred  = c("pred", "prediction", "predicted", "yhat", "gebv"),
    rep   = c("rep", "repeat", "repetition"),
    fold  = c("fold", "cvfold"),
    model = c("model", "method")
  )

  for (target in names(aliases)) {
    hit <- which(key %in% aliases[[target]])
    if (length(hit) >= 1 && !(target %in% names(dat))) {
      setnames(dat, old = old[hit[1]], new = target)
    }
  }
  dat
}

model_from_filename <- function(file) {
  x <- tools::file_path_sans_ext(basename(file))

  # Example: dataset_500_250_ABG_GBLUP_predictions_20rep_5fold.txt
  m <- regexec("(?i)(?:^|_)([0-9]+)_([0-9]+)_ABG(?:_GBLUP)?(?:_|$)", x, perl = TRUE)
  z <- regmatches(x, m)[[1]]
  if (length(z) == 3) return(paste0("ABG ", z[2], "/", z[3]))

  # Legacy label parser: a filename with only B produces a label without V.
  m <- regexec("(?i)(?:^|_)([0-9]+)_ABG(?:_GBLUP)?(?:_|$)", x, perl = TRUE)
  z <- regmatches(x, m)[[1]]
  if (length(z) == 2) return(paste0("ABG ", z[2]))

  if (grepl("(?i)(?:^|_)GBLUP(?:_|$)", x, perl = TRUE)) return("GBLUP")
  NA_character_
}

clean_model_name <- function(x) {
  x <- trimws(as.character(x))
  x <- gsub("(?i)^ABG[_ -]*", "ABG ", x, perl = TRUE)
  x <- gsub("_", "/", x, fixed = TRUE)
  x <- gsub("(?i)^GBLUP$", "GBLUP", x, perl = TRUE)
  x
}

model_sort_key <- function(x) {
  first_num <- suppressWarnings(as.numeric(sub(".*?([0-9]+).*", "\\1", x)))
  second_num <- suppressWarnings(as.numeric(sub(".*?[0-9]+[^0-9]+([0-9]+).*", "\\1", x)))
  first_num[is.na(first_num)] <- Inf
  second_num[is.na(second_num)] <- Inf
  data.frame(is_not_gblup = x != "GBLUP", first_num, second_num, label = x)
}

############################################################
# 3. Read compatible prediction files
############################################################

read_prediction_file <- function(file) {
  dat <- tryCatch(
    fread(file, header = TRUE, data.table = TRUE),
    error = function(e) {
      warning("Skipped unreadable file: ", basename(file), " (", e$message, ")")
      return(NULL)
    }
  )
  if (is.null(dat)) return(NULL)

  dat <- standardize_column_names(dat)
  required <- c("id", "y", "pred", "rep", "fold")
  missing <- setdiff(required, names(dat))
  if (length(missing) > 0) {
    warning("Skipped incompatible file: ", basename(file),
            "; missing columns: ", paste(missing, collapse = ", "))
    return(NULL)
  }

  # Use the model column when present; otherwise infer the model from the filename.
  if (!("model" %in% names(dat))) {
    inferred_model <- model_from_filename(file)
    if (is.na(inferred_model)) {
      warning("Skipped file because its model could not be inferred: ", basename(file))
      return(NULL)
    }
    dat[, model := inferred_model]
  }

  dat[, source_file := basename(file)]
  dat <- dat[, .(
    model = clean_model_name(model),
    id = as.character(id),
    y = suppressWarnings(as.numeric(y)),
    pred = suppressWarnings(as.numeric(pred)),
    rep = suppressWarnings(as.integer(rep)),
    fold = suppressWarnings(as.integer(fold)),
    source_file
  )]
  dat[is.finite(y) & is.finite(pred) & !is.na(rep) & !is.na(fold) & nzchar(model)]
}

prediction_list <- lapply(candidate_files, function(f) {
  message("Checking: ", basename(f))
  read_prediction_file(f)
})
prediction_list <- Filter(function(x) !is.null(x) && nrow(x) > 0, prediction_list)

if (length(prediction_list) == 0) {
  stop("Files were found, but none contained compatible prediction columns: ",
       "id, y, pred, rep and fold (a model/method column is also needed for all_models files).")
}

all_predictions <- rbindlist(prediction_list, use.names = TRUE)

# Remove exact duplicate prediction records.
duplicate_rows <- duplicated(all_predictions, by = c("model", "rep", "fold", "id", "y", "pred"))
if (any(duplicate_rows)) {
  message("Removed ", sum(duplicate_rows), " duplicate prediction rows.")
  all_predictions <- all_predictions[!duplicate_rows]
}

model_names <- unique(all_predictions$model)
keys <- model_sort_key(model_names)
model_levels <- keys$label[do.call(order, keys)]
message("Models detected: ", paste(model_levels, collapse = ", "))

############################################################
# 4. Check that models use identical cross-validation partitions
############################################################

n_models <- length(model_levels)
fold_check <- all_predictions %>%
  distinct(model, rep, fold, id) %>%
  group_by(rep, id) %>%
  summarise(
    models_present = n_distinct(model),
    folds_present = n_distinct(fold),
    .groups = "drop"
  )

bad_fold <- fold_check %>%
  filter(models_present != n_models | folds_present != 1)

if (nrow(bad_fold) > 0) {
  stop("The models did not use identical CV partitions. Found ", nrow(bad_fold),
       " inconsistent rep-ID combinations. Please check whether all result files belong to the same dataset.")
}
message("CV partition check passed: all models used identical folds.")

############################################################
# 5. Pool held-out predictions across folds within each repeat and compute metrics
############################################################

repeat_metrics <- all_predictions %>%
  group_by(model, rep) %>%
  summarise(
    Accuracy = if (n() >= 2 && sd(y) > 0 && sd(pred) > 0) cor(y, pred) else NA_real_,
    Bias = mean(pred - y),
    MSE = mean((pred - y)^2),
    RMSE = sqrt(MSE),
    MAE = mean(abs(pred - y)),
    n_test = n(),
    .groups = "drop"
  )

repeat_metrics$model <- factor(repeat_metrics$model, levels = model_levels)

summary_metrics <- repeat_metrics %>%
  group_by(model) %>%
  summarise(
    n_rep = n(),
    across(c(Accuracy, Bias, MSE, RMSE, MAE),
           list(mean = ~mean(.x, na.rm = TRUE), sd = ~sd(.x, na.rm = TRUE)),
           .names = "{.fn}_{.col}"),
    .groups = "drop"
  )

fwrite(repeat_metrics, file.path(output_dir, "GS_metrics_by_repeat.txt"), sep = "\t")
fwrite(summary_metrics, file.path(output_dir, "GS_metrics_summary.txt"), sep = "\t")
print(summary_metrics)

############################################################
# 6. Plot and save Accuracy, Bias, MSE and MAE separately
############################################################

plot_one_metric <- function(metric_name) {
  dat <- repeat_metrics %>%
    select(model, rep, value = all_of(metric_name)) %>%
    filter(is.finite(value))

  if (nrow(dat) == 0) {
    warning("No finite values available for ", metric_name, "; plot skipped.")
    return(invisible(NULL))
  }

  # The red dashed line is the GBLUP mean across repeats, when available.
  gblup_mean <- mean(dat$value[dat$model == "GBLUP"], na.rm = TRUE)

  p <- ggplot(dat, aes(x = model, y = value, fill = model))
  if (is.finite(gblup_mean)) {
    p <- p + geom_hline(
      yintercept = gblup_mean, color = "red", linetype = "dashed",
      linewidth = 0.8, alpha = 0.85
    )
  }

  p <- p +
    geom_boxplot(width = 0.65, alpha = 0.80, outlier.shape = NA) +
    geom_jitter(width = 0.12, height = 0, size = 1.5, alpha = 0.55) +
    stat_summary(fun = mean, geom = "point", shape = 23, size = 3, fill = "white") +
    theme_bw(base_size = 12) +
    theme(
      legend.position = "none",
      axis.text.x = element_text(angle = 40, hjust = 1, vjust = 1),
      axis.title = element_text(face = "bold"),
      panel.grid.minor = element_blank()
    ) +
    labs(x = "Model", y = metric_name)

  base <- file.path(output_dir, paste0("GS_model_comparison_", tolower(metric_name)))
  ggsave(paste0(base, ".pdf"), p, width = 8, height = 6)
  ggsave(paste0(base, ".png"), p, width = 8, height = 6, dpi = 600)
  print(p)
  invisible(p)
}

invisible(lapply(c("Accuracy", "Bias", "MSE", "MAE"), plot_one_metric))

message("Finished. Results were saved in: ", normalizePath(output_dir))
