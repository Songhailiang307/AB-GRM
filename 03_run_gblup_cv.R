#!/usr/bin/env Rscript
# Run GBLUP and AB-GBLUP with shared repeated cross-validation partitions.
# Usage: Rscript run_gblup_cv.R
# Configure the settings and GRM table below before execution.
# Dependencies: sommer, data.table, dplyr; gzip input may require R.utils.
# Phenotypes: FID, IID, trait in the first three columns. Matching uses IID
# only, so IIDs must be unique across the entire dataset. Missing phenotypes
# must be NA (not numeric sentinel values such as -9).
# Square GRMs must contain a header and a first column of sample IDs.
# This script retains the original sommer mmer/vsr API and extraction layout.
# Use the sommer version from the analysis environment; no version is assumed.
# Defaults: 20 repeats, 5 folds, seed 123. Output suffixes retain 20rep_5fold;
# changing these settings also requires coordinated filename-pattern changes.

library(sommer)
library(data.table)
library(dplyr)

############################################################
# 1. GENERAL SETTINGS
############################################################

# Edit this configuration for each dataset and trait before running.
# Paths are resolved relative to the working directory.
dataset_prefix <- "dataset"
pheno_file <- "data/phenotypes.txt"
pheno_header <- FALSE  # TRUE for a phenotype file with a header row.

trait_name <- "trait"

nrep <- 20
nfold <- 5
random_seed <- 123

############################################################
# Output directory
############################################################

result_dir <- file.path("results", trait_name)

dir.create(
  result_dir,
  recursive = TRUE,
  showWarnings = FALSE
)

############################################################
# Reuse an existing CV fold file when available.
#
# TRUE:
# Reuse the saved fold assignments to ensure that future
# reruns use exactly the same training and testing sets.
#
# FALSE:
# Generate a new fold file using random_seed.
############################################################

reuse_existing_folds <- TRUE

fold_file <- file.path(
  result_dir,
  "CV_folds_20rep_5fold.txt"
)

############################################################
# 2. GRM CONFIGURATION
#
# type:
#   gcta   = GCTA .grm.gz + .grm.id
#   square = Python AB-GRM square matrix
#
# output_stem determines the output filenames used by
# plot_gs_metrics.R.
#
# Modify the GRM filenames below to match your actual files.
############################################################

gcta_grm_file <- "data/G_text.grm.gz"
gcta_id_file  <- "data/G_text.grm.id"

abg_dir <- "results/ab_grm"

grm_config <- data.frame(
  model = c(
    "GBLUP",
    "ABG 10/5",
    "ABG 20/10",
    "ABG 50/25",
    "ABG 100/50",
    "ABG 200/100",
    "ABG 500/250",
    "ABG 1000/500",
    "ABG 2000/1000",
    "ABG 3000/1500",
    "ABG 5000/2500",
    "ABG 7000/3500",
    "ABG 10000/5000"
  ),

  type = c(
    "gcta",
    rep("square", 12)
  ),

  grm_file = c(
    gcta_grm_file,

    file.path(abg_dir, paste0(dataset_prefix, "_B10_V5_GRM.txt")),
    file.path(abg_dir, paste0(dataset_prefix, "_B20_V10_GRM.txt")),
    file.path(abg_dir, paste0(dataset_prefix, "_B50_V25_GRM.txt")),
    file.path(abg_dir, paste0(dataset_prefix, "_B100_V50_GRM.txt")),
    file.path(abg_dir, paste0(dataset_prefix, "_B200_V100_GRM.txt")),
    file.path(abg_dir, paste0(dataset_prefix, "_B500_V250_GRM.txt")),
    file.path(abg_dir, paste0(dataset_prefix, "_B1000_V500_GRM.txt")),
    file.path(abg_dir, paste0(dataset_prefix, "_B2000_V1000_GRM.txt")),
    file.path(abg_dir, paste0(dataset_prefix, "_B3000_V1500_GRM.txt")),
    file.path(abg_dir, paste0(dataset_prefix, "_B5000_V2500_GRM.txt")),
    file.path(abg_dir, paste0(dataset_prefix, "_B7000_V3500_GRM.txt")),
    file.path(abg_dir, paste0(dataset_prefix, "_B10000_V5000_GRM.txt"))
  ),

  id_file = c(
    gcta_id_file,
    rep(NA_character_, 12)
  ),

  output_stem = c(
    paste0(dataset_prefix, "_GBLUP"),
    paste0(dataset_prefix, "_10_5_ABG_GBLUP"),
    paste0(dataset_prefix, "_20_10_ABG_GBLUP"),
    paste0(dataset_prefix, "_50_25_ABG_GBLUP"),
    paste0(dataset_prefix, "_100_50_ABG_GBLUP"),
    paste0(dataset_prefix, "_200_100_ABG_GBLUP"),
    paste0(dataset_prefix, "_500_250_ABG_GBLUP"),
    paste0(dataset_prefix, "_1000_500_ABG_GBLUP"),
    paste0(dataset_prefix, "_2000_1000_ABG_GBLUP"),
    paste0(dataset_prefix, "_3000_1500_ABG_GBLUP"),
    paste0(dataset_prefix, "_5000_2500_ABG_GBLUP"),
    paste0(dataset_prefix, "_7000_3500_ABG_GBLUP"),
    paste0(dataset_prefix, "_10000_5000_ABG_GBLUP")
  ),

  stringsAsFactors = FALSE
)

############################################################
# 3. CHECK INPUT FILES
############################################################

for (i in seq_len(nrow(grm_config))) {

  if (!file.exists(grm_config$grm_file[i])) {

    stop(
      paste0(
        "GRM file does not exist for ",
        grm_config$model[i],
        ":\n",
        grm_config$grm_file[i]
      )
    )
  }

  if (
    grm_config$type[i] == "gcta" &&
    !file.exists(grm_config$id_file[i])
  ) {

    stop(
      paste0(
        "GCTA ID file does not exist:\n",
        grm_config$id_file[i]
      )
    )
  }
}

fwrite(
  grm_config,
  file.path(result_dir, "GRM_model_config.txt"),
  sep = "\t"
)

############################################################
# 4. READ PHENOTYPE
############################################################

pheno <- fread(
  pheno_file,
  header = pheno_header
)

if (ncol(pheno) < 3) {
  stop("Phenotype file must contain at least FID, IID and trait.")
}

colnames(pheno)[1:3] <- c(
  "FID",
  "IID",
  trait_name
)

pheno <- pheno %>%
  transmute(
    id = as.character(IID),
    y = as.numeric(.data[[trait_name]])
  ) %>%
  filter(
    !is.na(id),
    is.finite(y)
  )

if (anyDuplicated(pheno$id)) {
  stop("Duplicated IID values were found in the phenotype file.")
}

cat(
  "Phenotyped individuals:",
  nrow(pheno),
  "\n"
)

############################################################
# 5. FUNCTION: READ GCTA GRM
############################################################

read_gcta_grm <- function(grm_file, id_file) {

  ids <- fread(
    id_file,
    header = FALSE
  )

  if (ncol(ids) < 2) {
    stop("GCTA GRM ID file must contain FID and IID.")
  }

  colnames(ids)[1:2] <- c(
    "FID",
    "IID"
  )

  ids$id <- as.character(ids$IID)

  n <- nrow(ids)

  grm_long <- fread(
    grm_file,
    header = FALSE
  )

  if (ncol(grm_long) < 4) {
    stop("GCTA GRM file must contain i, j, N and G.")
  }

  colnames(grm_long)[1:4] <- c(
    "i",
    "j",
    "N",
    "G"
  )

  G <- matrix(
    NA_real_,
    nrow = n,
    ncol = n
  )

  G[
    cbind(grm_long$i, grm_long$j)
  ] <- grm_long$G

  G[
    cbind(grm_long$j, grm_long$i)
  ] <- grm_long$G

  rownames(G) <- ids$id
  colnames(G) <- ids$id

  storage.mode(G) <- "double"

  if (anyNA(G)) {
    stop("Missing values remained after reconstructing the GCTA GRM.")
  }

  return(G)
}

############################################################
# 6. FUNCTION: READ SQUARE AB-GRM
############################################################

read_square_grm <- function(grm_file) {

  G_df <- fread(
    grm_file,
    header = TRUE,
    data.table = FALSE,
    check.names = FALSE
  )

  ids <- as.character(
    G_df[[1]]
  )

  G <- as.matrix(
    G_df[, -1, drop = FALSE]
  )

  storage.mode(G) <- "double"

  rownames(G) <- ids
  colnames(G) <- colnames(G_df)[-1]

  if (nrow(G) != ncol(G)) {

    stop(
      paste0(
        "The AB-GRM is not square:\n",
        grm_file
      )
    )
  }

  if (!setequal(rownames(G), colnames(G))) {

    stop(
      paste0(
        "Row and column IDs do not match in:\n",
        grm_file
      )
    )
  }

  # Reorder columns to match the row order
  G <- G[
    ids,
    ids,
    drop = FALSE
  ]

  return(G)
}

############################################################
# 7. READ ALL GRMS
############################################################

G_list <- vector(
  mode = "list",
  length = nrow(grm_config)
)

names(G_list) <- grm_config$model

for (i in seq_len(nrow(grm_config))) {

  model_name <- grm_config$model[i]

  cat(
    "\nReading GRM:",
    model_name,
    "\n"
  )

  if (grm_config$type[i] == "gcta") {

    G <- read_gcta_grm(
      grm_file = grm_config$grm_file[i],
      id_file = grm_config$id_file[i]
    )

  } else if (grm_config$type[i] == "square") {

    G <- read_square_grm(
      grm_file = grm_config$grm_file[i]
    )

  } else {

    stop(
      paste0(
        "Unknown GRM type: ",
        grm_config$type[i]
      )
    )
  }

  symmetry_diff <- max(
    abs(G - t(G)),
    na.rm = TRUE
  )

  cat(
    "Individuals:",
    nrow(G),
    "\n"
  )

  cat(
    "Symmetry max difference:",
    symmetry_diff,
    "\n"
  )

  if (symmetry_diff > 1e-5) {

    stop(
      paste0(
        "GRM is not sufficiently symmetric: ",
        model_name
      )
    )
  }

  # Remove tiny rounding asymmetry
  G <- (G + t(G)) / 2

  G_list[[model_name]] <- G
}

############################################################
# 8. IDENTIFY COMMON INDIVIDUALS
#
# Every model must use exactly the same individuals.
############################################################

common_id <- Reduce(
  intersect,
  c(
    list(pheno$id),
    lapply(G_list, rownames)
  )
)

# Preserve phenotype order
analysis_ids <- pheno$id[
  pheno$id %in% common_id
]

if (length(analysis_ids) < 3) {
  stop("Too few common individuals among phenotype and GRMs.")
}

pheno <- pheno %>%
  filter(id %in% analysis_ids) %>%
  arrange(
    match(id, analysis_ids)
  )

G_list <- lapply(
  G_list,
  function(G) {

    G[
      analysis_ids,
      analysis_ids,
      drop = FALSE
    ]
  }
)

cat(
  "\nCommon individuals used by all models:",
  length(analysis_ids),
  "\n"
)

fwrite(
  data.frame(id = analysis_ids),
  file.path(result_dir, "analysis_sample_ids.txt"),
  sep = "\t"
)

############################################################
# 9. FINAL GRM CHECK
############################################################

grm_check <- bind_rows(
  lapply(
    names(G_list),
    function(model_name) {

      G <- G_list[[model_name]]

      data.frame(
        model = model_name,
        n_individual = nrow(G),
        symmetry_max_diff = max(
          abs(G - t(G)),
          na.rm = TRUE
        ),
        mean_diagonal = mean(
          diag(G),
          na.rm = TRUE
        ),
        sd_diagonal = sd(
          diag(G),
          na.rm = TRUE
        ),
        mean_offdiagonal = mean(
          G[lower.tri(G)],
          na.rm = TRUE
        ),
        sd_offdiagonal = sd(
          G[lower.tri(G)],
          na.rm = TRUE
        )
      )
    }
  )
)

fwrite(
  grm_check,
  file.path(result_dir, "GRM_basic_check.txt"),
  sep = "\t"
)

print(grm_check)

############################################################
# 10. CREATE OR READ SHARED CV FOLDS
#
# All GRMs use this exact same fold table.
############################################################

if (
  reuse_existing_folds &&
  file.exists(fold_file)
) {

  cat(
    "\nReading existing CV folds:\n",
    fold_file,
    "\n"
  )

  cv_folds <- fread(
    fold_file,
    header = TRUE,
    data.table = FALSE
  )

  required_fold_columns <- c(
    "rep",
    "id",
    "fold"
  )

  missing_fold_columns <- setdiff(
    required_fold_columns,
    colnames(cv_folds)
  )

  if (length(missing_fold_columns) > 0) {

    stop(
      paste0(
        "Missing columns in CV fold file: ",
        paste(missing_fold_columns, collapse = ", ")
      )
    )
  }

  cv_folds <- cv_folds %>%
    transmute(
      rep = as.integer(rep),
      id = as.character(id),
      fold = as.integer(fold)
    )

  if (!setequal(cv_folds$id, pheno$id)) {

    stop(
      paste0(
        "The IDs in the existing CV fold file do not match ",
        "the current analysis IDs. Delete the old fold file ",
        "or set reuse_existing_folds <- FALSE."
      )
    )
  }

  if (
    !setequal(unique(cv_folds$rep), seq_len(nrep)) ||
    !all(cv_folds$fold %in% seq_len(nfold))
  ) {

    stop(
      "The existing CV fold file does not match nrep or nfold."
    )
  }

} else {

  cat(
    "\nGenerating a new shared CV fold table...\n"
  )

  set.seed(random_seed)

  cv_list <- vector(
    mode = "list",
    length = nrep
  )

  for (r in seq_len(nrep)) {

    fold_id <- sample(
      rep(
        seq_len(nfold),
        length.out = nrow(pheno)
      )
    )

    cv_list[[r]] <- data.frame(
      rep = r,
      id = pheno$id,
      fold = fold_id
    )
  }

  cv_folds <- bind_rows(
    cv_list
  )

  fwrite(
    cv_folds,
    fold_file,
    sep = "\t"
  )
}

############################################################
# Validate fold assignments
############################################################

cv_validation <- cv_folds %>%
  count(
    rep,
    id,
    name = "n"
  ) %>%
  filter(n != 1)

if (nrow(cv_validation) > 0) {
  stop("Some rep-ID combinations occur more than once in CV folds.")
}

fold_size <- cv_folds %>%
  count(
    rep,
    fold,
    name = "n_test"
  )

fwrite(
  fold_size,
  file.path(result_dir, "CV_fold_sizes.txt"),
  sep = "\t"
)

cat(
  "Shared CV folds are ready.\n"
)

############################################################
# 11. FUNCTION: EXTRACT GENOMIC EFFECT
############################################################

extract_genomic_effect <- function(fit) {

  u <- fit$U$`u:id`$y_cv

  if (is.null(u)) {
    stop("Could not extract genomic effect from sommer output.")
  }

  if (is.null(dim(u))) {

    u_df <- data.frame(
      id = names(u),
      g_hat = as.numeric(u),
      stringsAsFactors = FALSE
    )

  } else {

    u_df <- data.frame(
      id = rownames(u),
      g_hat = as.numeric(u[, 1]),
      stringsAsFactors = FALSE
    )
  }

  if (is.null(u_df$id)) {
    stop("Genomic effect does not contain individual IDs.")
  }

  return(u_df)
}

############################################################
# 12. FUNCTION: RUN ONE GRM MODEL
############################################################

run_one_model <- function(
    model_name,
    G,
    output_stem
) {

  cat(
    "\n========================================\n"
  )

  cat(
    "Running model:",
    model_name,
    "\n"
  )

  cat(
    "========================================\n"
  )

  all_pred <- vector(
    mode = "list",
    length = nrep * nfold
  )

  all_acc <- vector(
    mode = "list",
    length = nrep * nfold
  )

  result_index <- 1

  for (r in seq_len(nrep)) {

    fold_r <- cv_folds %>%
      filter(rep == r)

    fold_id <- fold_r$fold[
      match(pheno$id, fold_r$id)
    ]

    if (anyNA(fold_id)) {
      stop(
        paste0(
          "Missing fold assignment in repeat ",
          r
        )
      )
    }

    for (f in seq_len(nfold)) {

      cat(
        "Model:",
        model_name,
        "| repeat:",
        r,
        "| fold:",
        f,
        "\n"
      )

      test_idx <- which(
        fold_id == f
      )

      test_ids <- pheno$id[
        test_idx
      ]

      dat <- pheno

      ######################################################
      # Keep identical factor levels for all GRMs
      ######################################################

      dat$id <- factor(
        dat$id,
        levels = pheno$id
      )

      ######################################################
      # Mask test phenotypes
      ######################################################

      dat$y_cv <- dat$y
      dat$y_cv[test_idx] <- NA_real_

      ######################################################
      # Fit GBLUP
      ######################################################

      fit <- tryCatch(

        mmer(
          fixed = y_cv ~ 1,
          random = ~ vsr(
            id,
            Gu = G
          ),
          rcov = ~ units,
          data = dat,
          verbose = FALSE
        ),

        error = function(e) {

          stop(
            paste0(
              "sommer failed for model ",
              model_name,
              ", repeat ",
              r,
              ", fold ",
              f,
              ":\n",
              conditionMessage(e)
            ),
            call. = FALSE
          )
        }
      )

      ######################################################
      # Extract prediction
      ######################################################

      mu <- as.numeric(
        fit$Beta$Estimate[1]
      )

      u_df <- extract_genomic_effect(
        fit
      )

      pred_df <- data.frame(
        id = as.character(dat$id),
        y = dat$y,
        stringsAsFactors = FALSE
      ) %>%
        left_join(
          u_df,
          by = "id"
        ) %>%
        mutate(
          pred = mu + g_hat,
          rep = r,
          fold = f,
          set = ifelse(
            id %in% test_ids,
            "test",
            "train"
          ),
          model = model_name
        )

      test_pred <- pred_df %>%
        filter(set == "test")

      complete_idx <- is.finite(test_pred$y) &
        is.finite(test_pred$pred)

      if (sum(complete_idx) < 3) {

        stop(
          paste0(
            "Too few complete predictions for model ",
            model_name,
            ", repeat ",
            r,
            ", fold ",
            f
          )
        )
      }

      ######################################################
      # Fold-level evaluation
      ######################################################

      observed <- test_pred$y[
        complete_idx
      ]

      predicted <- test_pred$pred[
        complete_idx
      ]

      accuracy <- cor(
        observed,
        predicted,
        method = "pearson"
      )

      bias <- mean(
        predicted - observed
      )

      mse <- mean(
        (predicted - observed)^2
      )

      rmse <- sqrt(
        mse
      )

      mae <- mean(
        abs(predicted - observed)
      )

      acc_df <- data.frame(
        model = model_name,
        rep = r,
        fold = f,
        accuracy = accuracy,
        Bias = bias,
        MSE = mse,
        RMSE = rmse,
        MAE = mae,
        n_test = length(observed)
      )

      all_pred[[result_index]] <- test_pred %>%
        select(
          model,
          id,
          y,
          pred,
          rep,
          fold,
          set
        )

      all_acc[[result_index]] <- acc_df

      result_index <- result_index + 1
    }
  }

  pred_res <- bind_rows(
    all_pred
  )

  acc_res <- bind_rows(
    all_acc
  )

  ##########################################################
  # Save model-specific outputs
  ##########################################################

  prediction_file <- file.path(
    result_dir,
    paste0(
      output_stem,
      "_predictions_20rep_5fold.txt"
    )
  )

  accuracy_file <- file.path(
    result_dir,
    paste0(
      output_stem,
      "_accuracy_20rep_5fold.txt"
    )
  )

  summary_file <- file.path(
    result_dir,
    paste0(
      output_stem,
      "_accuracy_summary.txt"
    )
  )

  fwrite(
    pred_res,
    prediction_file,
    sep = "\t"
  )

  fwrite(
    acc_res,
    accuracy_file,
    sep = "\t"
  )

  summary_res <- acc_res %>%
    summarise(
      model = first(model),

      mean_accuracy = mean(
        accuracy,
        na.rm = TRUE
      ),

      sd_accuracy = sd(
        accuracy,
        na.rm = TRUE
      ),

      mean_Bias = mean(
        Bias,
        na.rm = TRUE
      ),

      sd_Bias = sd(
        Bias,
        na.rm = TRUE
      ),

      mean_MSE = mean(
        MSE,
        na.rm = TRUE
      ),

      sd_MSE = sd(
        MSE,
        na.rm = TRUE
      ),

      mean_RMSE = mean(
        RMSE,
        na.rm = TRUE
      ),

      sd_RMSE = sd(
        RMSE,
        na.rm = TRUE
      ),

      mean_MAE = mean(
        MAE,
        na.rm = TRUE
      ),

      sd_MAE = sd(
        MAE,
        na.rm = TRUE
      )
    )

  fwrite(
    summary_res,
    summary_file,
    sep = "\t"
  )

  print(summary_res)

  return(
    list(
      predictions = pred_res,
      accuracy = acc_res,
      summary = summary_res
    )
  )
}

############################################################
# 13. RUN ALL GRM MODELS
############################################################

model_results <- vector(
  mode = "list",
  length = nrow(grm_config)
)

names(model_results) <- grm_config$model

for (i in seq_len(nrow(grm_config))) {

  model_name <- grm_config$model[i]

  model_results[[model_name]] <- run_one_model(
    model_name = model_name,
    G = G_list[[model_name]],
    output_stem = grm_config$output_stem[i]
  )
}

############################################################
# 14. COMBINE ALL MODEL RESULTS
############################################################

all_model_predictions <- bind_rows(
  lapply(
    model_results,
    function(x) x$predictions
  )
)

all_model_accuracy <- bind_rows(
  lapply(
    model_results,
    function(x) x$accuracy
  )
)

all_model_summary <- bind_rows(
  lapply(
    model_results,
    function(x) x$summary
  )
)

fwrite(
  all_model_predictions,
  file.path(
    result_dir,
    "sommer_all_models_predictions_20rep_5fold.txt"
  ),
  sep = "\t"
)

fwrite(
  all_model_accuracy,
  file.path(
    result_dir,
    "sommer_all_models_accuracy_20rep_5fold.txt"
  ),
  sep = "\t"
)

fwrite(
  all_model_summary,
  file.path(
    result_dir,
    "sommer_all_models_accuracy_summary.txt"
  ),
  sep = "\t"
)

############################################################
# 15. FINAL COMPARABILITY CHECK
#
# Every model must contain the same rep, fold and test IDs.
############################################################

comparison_check <- all_model_predictions %>%
  distinct(
    model,
    rep,
    fold,
    id
  ) %>%
  group_by(
    rep,
    fold,
    id
  ) %>%
  summarise(
    n_model = n_distinct(model),
    .groups = "drop"
  ) %>%
  filter(
    n_model != nrow(grm_config)
  )

if (nrow(comparison_check) > 0) {

  stop(
    paste0(
      "Final comparability check failed. ",
      nrow(comparison_check),
      " rep-fold-ID combinations were not present in all models."
    )
  )
}

cat(
  "\n========================================\n"
)

cat(
  "All GRM models finished successfully.\n"
)

cat(
  "All models used identical individuals and CV folds.\n"
)

cat(
  "Results directory:\n",
  result_dir,
  "\n"
)

cat(
  "========================================\n"
)
