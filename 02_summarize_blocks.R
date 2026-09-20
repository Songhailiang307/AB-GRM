#!/usr/bin/env Rscript
# Summarize original LD blocks, retained AB-GRM blocks and adaptive weights.
# Usage: Rscript summarize_blocks.R [input_directory] [dataset_prefix] [det_filename]
# Dependencies: data.table, dplyr, tidyr, ggplot2, scales.
# Place the .blocks.det and matching block-statistics files in input_directory.
# det_filename is a filename relative to input_directory.
# Outputs: four summary/data tables and four plots in PDF and PNG formats.

############################################################
# Summarize and plot LD block statistics from AB-GRM
#
# User-defined inputs:
# 1. <det_file_name>, for example data.blocks.det
# 2. <stat_prefix>_B*_V*_block_statistics.txt
#
# Example:
# stat_prefix <- "dataset"
# dataset_B10_V8_block_statistics.txt
# dataset_B50_V30_block_statistics.txt
#
# Outputs:
# 1. SNP-number summary for raw and final blocks
# 2. Adaptive-weight summary
# 3. Block SNP-number violin plot
# 4. Block SNP-number ECDF plot
# 5. Adaptive-weight violin plot
# 6. Adaptive-weight ECDF plot
############################################################

library(data.table)
library(dplyr)
library(tidyr)
library(ggplot2)
library(scales)

############################################################
# 1. User settings
############################################################

############################################################
# Directory containing input files
############################################################

args <- commandArgs(trailingOnly = TRUE)
result_dir <- if (length(args) >= 1) args[1] else "."
if (!dir.exists(result_dir)) stop("Input directory does not exist: ", result_dir)

############################################################
# Filename prefix for AB-GRM block statistics
#
# Example filenames:
# dataset_B10_V5_block_statistics.txt
# dataset_B500_V250_block_statistics.txt
#
# Set the prefix accordingly:
# stat_prefix <- "dataset"
############################################################

stat_prefix <- if (length(args) >= 2) args[2] else "dataset"

############################################################
# Original PLINK blocks.det filename
#
# This filename can be configured independently of stat_prefix.
############################################################

det_file_name <- if (length(args) >= 3) args[3] else paste0(stat_prefix, ".blocks.det")

############################################################
# Output filename prefix
#
# Use the block-statistics prefix by default.
############################################################

output_prefix <- stat_prefix

############################################################
# Output subdirectory name
############################################################

out_dir_name <- "block_statistics_summary"

############################################################
# Construct full paths
############################################################

det_file <- file.path(
  result_dir,
  det_file_name
)

out_dir <- file.path(
  result_dir,
  out_dir_name
)

dir.create(
  out_dir,
  recursive = TRUE,
  showWarnings = FALSE
)

############################################################
# Construct output filenames with a consistent prefix
############################################################

output_file <- function(suffix) {

  file.path(
    out_dir,
    paste0(
      output_prefix,
      "_",
      suffix
    )
  )
}

############################################################
# 2. Find AB-GRM block statistics files
#
# Recognized filename formats:
# <stat_prefix>_B10_V8_block_statistics.txt
# <stat_prefix>_B3000_V100_block_statistics.txt
############################################################

all_stat_files <- list.files(
  path = result_dir,
  pattern = "_B[0-9]+_V[0-9]+_block_statistics\\.txt$",
  full.names = TRUE
)

############################################################
# Search by suffix, then match the literal prefix with startsWith().
# This avoids interpreting regex metacharacters in the prefix.
############################################################

stat_files <- all_stat_files[
  startsWith(
    basename(all_stat_files),
    paste0(
      stat_prefix,
      "_B"
    )
  )
]

if (length(stat_files) == 0) {

  stop(
    paste0(
      "No ",
      stat_prefix,
      "_B*_V*_block_statistics.txt files found in:\n",
      result_dir
    )
  )
}

if (!file.exists(det_file)) {

  stop(
    paste0(
      "Cannot find PLINK block file:\n",
      det_file
    )
  )
}

cat(
  "Statistics prefix:",
  stat_prefix,
  "\n"
)

cat(
  "Number of statistics files found:",
  length(stat_files),
  "\n"
)

cat(
  "Original PLINK block file:",
  det_file,
  "\n"
)

cat(
  "Statistics files:\n",
  paste(
    basename(stat_files),
    collapse = "\n"
  ),
  "\n"
)

############################################################
# 3. Extract parameters from filenames
############################################################

parse_parameter <- function(
    file,
    prefix) {

  file_name <- basename(
    file
  )

  expected_start <- paste0(
    prefix,
    "_"
  )

  if (!startsWith(
    file_name,
    expected_start
  )) {

    stop(
      paste0(
        "File does not start with expected prefix '",
        expected_start,
        "': ",
        file_name
      )
    )
  }

  ##########################################################
  # Remove the configured prefix by character position.
  #
  # Example:
  # dataset_B500_V250_block_statistics.txt
  # Removing dataset_ gives:
  # B500_V250_block_statistics.txt
  ##########################################################

  parameter_part <- substring(
    file_name,
    nchar(expected_start) + 1
  )

  match_result <- regexec(
    "^B([0-9]+)_V([0-9]+)_block_statistics\\.txt$",
    parameter_part
  )

  matched_text <- regmatches(
    parameter_part,
    match_result
  )[[1]]

  if (length(matched_text) != 3) {

    stop(
      paste(
        "Cannot parse B/V parameters from:",
        file_name
      )
    )
  }

  min_block_snp <- as.integer(
    matched_text[2]
  )

  min_valid_snp <- as.integer(
    matched_text[3]
  )

  data.frame(
    file = file,

    file_prefix = prefix,

    min_block_snp = min_block_snp,

    min_valid_snp = min_valid_snp,

    parameter = paste0(
      "B",
      min_block_snp,
      "/V",
      min_valid_snp
    ),

    stringsAsFactors = FALSE
  )
}

parameter_table <- bind_rows(
  lapply(
    stat_files,
    function(file) {

      parse_parameter(
        file = file,
        prefix = stat_prefix
      )
    }
  )
) %>%
  arrange(
    min_block_snp,
    min_valid_snp
  )

print(
  parameter_table
)

############################################################
# 4. Read one AB-GRM block statistics file
############################################################

read_abg_statistics <- function(
    file,
    min_block_snp,
    min_valid_snp,
    parameter) {

  dat <- fread(
    file,
    header = TRUE,
    data.table = FALSE
  )

  ##########################################################
  # Prefer the n_valid_snp column.
  #
  # In v2.4, n_valid_snp records SNP counts after polymorphism filtering.
  # Use this column when available.
  #
  # Otherwise fall back to n_snp.
  ##########################################################

  if ("n_valid_snp" %in% colnames(dat)) {

    snp_column <- "n_valid_snp"

  } else if ("n_snp" %in% colnames(dat)) {

    snp_column <- "n_snp"

  } else {

    stop(
      paste0(
        "Neither n_valid_snp nor n_snp was found in:\n",
        file
      )
    )
  }

  ##########################################################
  # Check the raw_weight column
  ##########################################################

  if (!"raw_weight" %in% colnames(dat)) {

    stop(
      paste0(
        "raw_weight column was not found in:\n",
        file
      )
    )
  }

  raw_weight <- as.numeric(
    dat$raw_weight
  )

  ##########################################################
  # Obtain normalized weights
  #
  # Order of preference:
  # 1. normalized_weight
  # 2. Renormalize weight_after_cap
  # 3. Renormalize raw_weight
  ##########################################################

  if ("normalized_weight" %in% colnames(dat)) {

    normalized_weight <- as.numeric(
      dat$normalized_weight
    )

    normalized_weight_source <-
      "normalized_weight"

  } else if (
    "weight_after_cap" %in% colnames(dat)
  ) {

    weight_after_cap <- as.numeric(
      dat$weight_after_cap
    )

    weight_sum <- sum(
      weight_after_cap[
        is.finite(weight_after_cap) &
        weight_after_cap > 0
      ],
      na.rm = TRUE
    )

    if (
      !is.finite(weight_sum) ||
      weight_sum <= 0
    ) {

      normalized_weight <- rep(
        NA_real_,
        nrow(dat)
      )

    } else {

      normalized_weight <-
        weight_after_cap / weight_sum
    }

    normalized_weight_source <-
      "weight_after_cap"

  } else {

    weight_sum <- sum(
      raw_weight[
        is.finite(raw_weight) &
        raw_weight > 0
      ],
      na.rm = TRUE
    )

    if (
      !is.finite(weight_sum) ||
      weight_sum <= 0
    ) {

      normalized_weight <- rep(
        NA_real_,
        nrow(dat)
      )

    } else {

      normalized_weight <-
        raw_weight / weight_sum
    }

    normalized_weight_source <-
      "raw_weight_recalculated"
  }

  ##########################################################
  # Assemble the standardized data table
  ##########################################################

  output <- data.frame(
    parameter = parameter,

    source = "Final valid block",

    min_block_snp = min_block_snp,

    min_valid_snp = min_valid_snp,

    block = if (
      "block" %in% colnames(dat)
    ) {
      as.integer(dat$block)
    } else {
      seq_len(nrow(dat))
    },

    chr = if (
      "chr" %in% colnames(dat)
    ) {
      as.character(dat$chr)
    } else {
      NA_character_
    },

    start_bp = if (
      "start_bp" %in% colnames(dat)
    ) {
      as.numeric(dat$start_bp)
    } else {
      NA_real_
    },

    end_bp = if (
      "end_bp" %in% colnames(dat)
    ) {
      as.numeric(dat$end_bp)
    } else {
      NA_real_
    },

    n_snp = as.numeric(
      dat[[snp_column]]
    ),

    snp_count_column = snp_column,

    stability = if (
      "stability" %in% colnames(dat)
    ) {
      as.numeric(dat$stability)
    } else {
      NA_real_
    },

    raw_weight = raw_weight,

    normalized_weight =
      normalized_weight,

    normalized_weight_source =
      normalized_weight_source,

    stringsAsFactors = FALSE
  )

  return(
    output
  )
}

############################################################
# 5. Read results for all AB-GRM parameter settings
############################################################

abg_list <- vector(
  mode = "list",
  length = nrow(parameter_table)
)

for (i in seq_len(nrow(parameter_table))) {

  cat(
    "Reading:",
    basename(parameter_table$file[i]),
    "\n"
  )

  abg_list[[i]] <- read_abg_statistics(
    file =
      parameter_table$file[i],

    min_block_snp =
      parameter_table$min_block_snp[i],

    min_valid_snp =
      parameter_table$min_valid_snp[i],

    parameter =
      parameter_table$parameter[i]
  )
}

abg_data <- bind_rows(
  abg_list
)

############################################################
# 6. Read the original PLINK blocks.det file
############################################################

det_data <- fread(
  det_file,
  header = TRUE,
  data.table = FALSE
)

############################################################
# Obtain SNP counts for the original blocks
############################################################

if ("NSNPS" %in% colnames(det_data)) {

  raw_n_snp <- as.numeric(
    det_data$NSNPS
  )

  raw_snp_column <- "NSNPS"

} else if ("SNPS" %in% colnames(det_data)) {

  ##########################################################
  # If NSNPS is absent, count pipe-separated entries in SNPS.
  ##########################################################

  raw_n_snp <- lengths(
    strsplit(
      as.character(det_data$SNPS),
      split = "\\|"
    )
  )

  raw_snp_column <- "SNPS_counted"

} else {

  stop(
    paste0(
      "Neither NSNPS nor SNPS was found in:\n",
      det_file
    )
  )
}

raw_block_data <- data.frame(
  parameter = "Raw DET",

  source = "Raw PLINK block",

  min_block_snp = NA_integer_,

  min_valid_snp = NA_integer_,

  block = seq_len(
    nrow(det_data)
  ),

  chr = if (
    "CHR" %in% colnames(det_data)
  ) {
    as.character(det_data$CHR)
  } else {
    NA_character_
  },

  start_bp = if (
    "BP1" %in% colnames(det_data)
  ) {
    as.numeric(det_data$BP1)
  } else {
    NA_real_
  },

  end_bp = if (
    "BP2" %in% colnames(det_data)
  ) {
    as.numeric(det_data$BP2)
  } else {
    NA_real_
  },

  n_snp = raw_n_snp,

  snp_count_column = raw_snp_column,

  stability = NA_real_,

  raw_weight = NA_real_,

  normalized_weight = NA_real_,

  normalized_weight_source =
    NA_character_,

  stringsAsFactors = FALSE
)

############################################################
# 7. Combine original blocks and final valid blocks
############################################################

all_block_data <- bind_rows(
  raw_block_data,
  abg_data
)

parameter_levels <- c(
  "Raw DET",
  parameter_table$parameter
)

all_block_data$parameter <- factor(
  all_block_data$parameter,
  levels = parameter_levels
)

abg_data$parameter <- factor(
  abg_data$parameter,
  levels = parameter_table$parameter
)

############################################################
# Remove records with invalid SNP counts
############################################################

all_block_data <- all_block_data %>%
  filter(
    is.finite(n_snp),
    n_snp > 0
  )

############################################################
# 8. Summarize block SNP counts for each parameter setting
############################################################

quantile_value <- function(
    x,
    probability) {

  as.numeric(
    quantile(
      x,
      probs = probability,
      na.rm = TRUE,
      names = FALSE,
      type = 7
    )
  )
}

block_snp_summary <- all_block_data %>%
  group_by(
    parameter,
    source,
    min_block_snp,
    min_valid_snp,
    snp_count_column
  ) %>%

  summarise(
    n_block = n(),

    min_snp = min(
      n_snp,
      na.rm = TRUE
    ),

    max_snp = max(
      n_snp,
      na.rm = TRUE
    ),

    mean_snp = mean(
      n_snp,
      na.rm = TRUE
    ),

    median_snp = median(
      n_snp,
      na.rm = TRUE
    ),

    q90_snp = quantile_value(
      n_snp,
      0.90
    ),

    q95_snp = quantile_value(
      n_snp,
      0.95
    ),

    q99_snp = quantile_value(
      n_snp,
      0.99
    ),

    sd_snp = sd(
      n_snp,
      na.rm = TRUE
    ),

    .groups = "drop"
  ) %>%

  arrange(
    match(
      as.character(parameter),
      parameter_levels
    )
  )

############################################################
# Save the SNP-count summary
############################################################

fwrite(
  block_snp_summary,
  output_file(
    "block_snp_summary.txt"
  ),
  sep = "\t"
)

print(
  block_snp_summary
)

############################################################
# 9. Prepare weight data
############################################################

weight_data <- abg_data %>%
  filter(
    is.finite(raw_weight),
    raw_weight > 0,
    is.finite(normalized_weight),
    normalized_weight > 0
  ) %>%

  group_by(
    parameter
  ) %>%

  mutate(
    n_block_weight = n(),

    ########################################################
    # The number of blocks varies between parameter settings, affecting
    # direct comparisons of normalized weights.
    #
    # relative_weight = normalized_weight * number of retained weight records
    #
    # A relative weight of 1 represents equal weighting.
    ########################################################

    relative_weight =
      normalized_weight *
      n_block_weight
  ) %>%

  ungroup()

############################################################
# 10. Summarize weight concentration
############################################################

top_weight_share <- function(
    weights,
    proportion) {

  weights <- weights[
    is.finite(weights) &
    weights >= 0
  ]

  if (length(weights) == 0) {

    return(
      NA_real_
    )
  }

  number_top <- max(
    1,
    ceiling(
      length(weights) *
      proportion
    )
  )

  sum(
    sort(
      weights,
      decreasing = TRUE
    )[seq_len(number_top)]
  )
}

weight_summary <- weight_data %>%
  group_by(
    parameter,
    min_block_snp,
    min_valid_snp
  ) %>%

  summarise(
    n_block = n(),

    sum_normalized_weight = sum(
      normalized_weight,
      na.rm = TRUE
    ),

    min_raw_weight = min(
      raw_weight,
      na.rm = TRUE
    ),

    max_raw_weight = max(
      raw_weight,
      na.rm = TRUE
    ),

    mean_raw_weight = mean(
      raw_weight,
      na.rm = TRUE
    ),

    median_raw_weight = median(
      raw_weight,
      na.rm = TRUE
    ),

    q90_raw_weight = quantile_value(
      raw_weight,
      0.90
    ),

    q95_raw_weight = quantile_value(
      raw_weight,
      0.95
    ),

    q99_raw_weight = quantile_value(
      raw_weight,
      0.99
    ),

    max_normalized_weight = max(
      normalized_weight,
      na.rm = TRUE
    ),

    median_normalized_weight = median(
      normalized_weight,
      na.rm = TRUE
    ),

    max_relative_weight = max(
      relative_weight,
      na.rm = TRUE
    ),

    median_relative_weight = median(
      relative_weight,
      na.rm = TRUE
    ),

    effective_block_number =
      1 / sum(
        normalized_weight^2,
        na.rm = TRUE
      ),

    effective_block_ratio =
      effective_block_number / n(),

    top_1_percent_weight =
      top_weight_share(
        normalized_weight,
        0.01
      ),

    top_5_percent_weight =
      top_weight_share(
        normalized_weight,
        0.05
      ),

    top_10_percent_weight =
      top_weight_share(
        normalized_weight,
        0.10
      ),

    .groups = "drop"
  ) %>%

  arrange(
    min_block_snp,
    min_valid_snp
  )

fwrite(
  weight_summary,
  output_file(
    "adaptive_weight_summary.txt"
  ),
  sep = "\t"
)

print(
  weight_summary
)

############################################################
# 11. Save the combined data tables
############################################################

fwrite(
  all_block_data,
  output_file(
    "all_block_snp_data.txt"
  ),
  sep = "\t"
)

fwrite(
  weight_data,
  output_file(
    "all_block_weight_data.txt"
  ),
  sep = "\t"
)

############################################################
# 12. Block SNP-count distributions: violin plots
############################################################

p_snp_violin <- ggplot(
  all_block_data,
  aes(
    x = parameter,
    y = n_snp,
    fill = parameter
  )
) +

  geom_violin(
    scale = "width",
    trim = TRUE,
    alpha = 0.75,
    linewidth = 0.35
  ) +

  geom_boxplot(
    width = 0.16,
    fill = "white",
    outlier.shape = NA,
    linewidth = 0.45
  ) +

  stat_summary(
    fun = mean,
    geom = "point",
    shape = 23,
    size = 2.2,
    fill = "white"
  ) +

  scale_y_log10(
    labels = label_number(
      accuracy = 1
    )
  ) +

  coord_flip() +

  theme_bw(
    base_size = 12
  ) +

  theme(
    legend.position = "none",

    panel.grid.minor =
      element_blank(),

    plot.title = element_text(
      face = "bold"
    )
  ) +

  labs(
    title = "Distribution of SNP number per LD block",
    subtitle = "White diamonds indicate arithmetic means",
    x = "Block parameter",
    y = "Number of SNPs per block (log10 scale)"
  )

ggsave(
  filename = output_file(
    "block_snp_distribution_violin.pdf"
  ),
  plot = p_snp_violin,
  width = 9,
  height = 7
)

ggsave(
  filename = output_file(
    "block_snp_distribution_violin.png"
  ),
  plot = p_snp_violin,
  width = 9,
  height = 7,
  dpi = 600
)

############################################################
# 13. Block SNP-count distributions: empirical cumulative distributions
#
# ECDFs facilitate comparisons of skewed and long-tailed distributions.
############################################################

p_snp_ecdf <- ggplot(
  all_block_data,
  aes(
    x = n_snp,
    color = parameter,
    group = parameter
  )
) +

  stat_ecdf(
    geom = "step",
    linewidth = 0.85,
    pad = FALSE
  ) +

  scale_x_log10(
    labels = label_number(
      accuracy = 1
    )
  ) +

  scale_y_continuous(
    labels = label_percent(
      accuracy = 1
    )
  ) +

  theme_bw(
    base_size = 12
  ) +

  theme(
    legend.position = "right",

    panel.grid.minor =
      element_blank(),

    plot.title = element_text(
      face = "bold"
    )
  ) +

  labs(
    title = "Cumulative distribution of SNP number per block",
    x = "Number of SNPs per block (log10 scale)",
    y = "Cumulative proportion",
    color = "Parameter"
  )

ggsave(
  filename = output_file(
    "block_snp_distribution_ECDF.pdf"
  ),
  plot = p_snp_ecdf,
  width = 9,
  height = 6
)

ggsave(
  filename = output_file(
    "block_snp_distribution_ECDF.png"
  ),
  plot = p_snp_ecdf,
  width = 9,
  height = 6,
  dpi = 600
)

############################################################
# 14. Reshape both weight measures for plotting
#
# Raw adaptive weight:
#   1 / (stability + EPS)
#
# Relative normalized weight:
#   normalized_weight × n_block
#
# A relative weight of 1 represents equal weighting.
############################################################

weight_plot_data <- weight_data %>%
  select(
    parameter,
    raw_weight,
    relative_weight
  ) %>%

  pivot_longer(
    cols = c(
      raw_weight,
      relative_weight
    ),

    names_to = "weight_type",

    values_to = "weight"
  ) %>%

  mutate(
    weight_type = recode(
      weight_type,

      raw_weight =
        "Raw adaptive weight",

      relative_weight =
        "Relative normalized weight"
    ),

    weight_type = factor(
      weight_type,
      levels = c(
        "Raw adaptive weight",
        "Relative normalized weight"
      )
    )
  ) %>%

  filter(
    is.finite(weight),
    weight > 0
  )

############################################################
# 15. Weight distributions: violin plots
############################################################

relative_reference <- data.frame(
  weight_type =
    factor(
      "Relative normalized weight",
      levels = levels(
        weight_plot_data$weight_type
      )
    ),

  reference = 1
)

p_weight_violin <- ggplot(
  weight_plot_data,
  aes(
    x = parameter,
    y = weight,
    fill = parameter
  )
) +

  geom_violin(
    scale = "width",
    trim = TRUE,
    alpha = 0.75,
    linewidth = 0.35
  ) +

  geom_boxplot(
    width = 0.16,
    fill = "white",
    outlier.shape = NA,
    linewidth = 0.45
  ) +

  stat_summary(
    fun = median,
    geom = "point",
    shape = 23,
    size = 2.0,
    fill = "white"
  ) +

  geom_hline(
    data = relative_reference,
    aes(
      yintercept = reference
    ),
    inherit.aes = FALSE,
    color = "red",
    linetype = "dashed",
    linewidth = 0.7
  ) +

  facet_wrap(
    ~weight_type,
    scales = "free_y",
    ncol = 1
  ) +

  scale_y_log10(
    labels = label_scientific()
  ) +

  theme_bw(
    base_size = 12
  ) +

  theme(
    legend.position = "none",

    axis.text.x = element_text(
      angle = 45,
      hjust = 1
    ),

    panel.grid.minor =
      element_blank(),

    strip.text = element_text(
      face = "bold"
    ),

    plot.title = element_text(
      face = "bold"
    )
  ) +

  labs(
    title = "Distribution of adaptive block weights",
    subtitle = paste0(
      "Relative normalized weight = normalized weight × block number; ",
      "red line represents equal weighting"
    ),
    x = "Block parameter",
    y = "Weight (log10 scale)"
  )

ggsave(
  filename = output_file(
    "adaptive_weight_distribution_violin.pdf"
  ),
  plot = p_weight_violin,
  width = 10,
  height = 9
)

ggsave(
  filename = output_file(
    "adaptive_weight_distribution_violin.png"
  ),
  plot = p_weight_violin,
  width = 10,
  height = 9,
  dpi = 600
)

############################################################
# 16. Relative weight distributions: empirical cumulative distributions
############################################################

p_weight_ecdf <- ggplot(
  weight_data,
  aes(
    x = relative_weight,
    color = parameter,
    group = parameter
  )
) +

  stat_ecdf(
    geom = "step",
    linewidth = 0.85,
    pad = FALSE
  ) +

  geom_vline(
    xintercept = 1,
    color = "red",
    linetype = "dashed",
    linewidth = 0.7
  ) +

  scale_x_log10(
    labels = label_number(
      accuracy = 0.01
    )
  ) +

  scale_y_continuous(
    labels = label_percent(
      accuracy = 1
    )
  ) +

  theme_bw(
    base_size = 12
  ) +

  theme(
    legend.position = "right",

    panel.grid.minor =
      element_blank(),

    plot.title = element_text(
      face = "bold"
    )
  ) +

  labs(
    title = "Cumulative distribution of relative adaptive weights",
    subtitle = "Relative weight = 1 represents equal weighting",
    x = "Relative normalized weight (log10 scale)",
    y = "Cumulative proportion",
    color = "Parameter"
  )

ggsave(
  filename = output_file(
    "adaptive_weight_distribution_ECDF.pdf"
  ),
  plot = p_weight_ecdf,
  width = 9,
  height = 6
)

ggsave(
  filename = output_file(
    "adaptive_weight_distribution_ECDF.png"
  ),
  plot = p_weight_ecdf,
  width = 9,
  height = 6,
  dpi = 600
)

############################################################
# 17. Completion report
############################################################

cat(
  "\nAll results were saved to:\n",
  out_dir,
  "\n"
)

cat(
  "\nGenerated summary tables:\n",

  "1. ",
  basename(
    output_file(
      "block_snp_summary.txt"
    )
  ),
  "\n",

  "2. ",
  basename(
    output_file(
      "adaptive_weight_summary.txt"
    )
  ),
  "\n",

  "3. ",
  basename(
    output_file(
      "all_block_snp_data.txt"
    )
  ),
  "\n",

  "4. ",
  basename(
    output_file(
      "all_block_weight_data.txt"
    )
  ),
  "\n",

  sep = ""
)

cat(
  "\nGenerated plots:\n",

  "1. ",
  basename(
    output_file(
      "block_snp_distribution_violin.pdf"
    )
  ),
  "/png\n",

  "2. ",
  basename(
    output_file(
      "block_snp_distribution_ECDF.pdf"
    )
  ),
  "/png\n",

  "3. ",
  basename(
    output_file(
      "adaptive_weight_distribution_violin.pdf"
    )
  ),
  "/png\n",

  "4. ",
  basename(
    output_file(
      "adaptive_weight_distribution_ECDF.pdf"
    )
  ),
  "/png\n",

  sep = ""
)
