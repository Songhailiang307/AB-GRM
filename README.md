# AB-GRM: adaptive block genomic relationship matrices

This repository contains four scripts for constructing AB-GRM matrices, evaluating GBLUP and AB-GBLUP with repeated cross-validation, and summarizing prediction performance and block characteristics.

| Script | Purpose |
| --- | --- |
| `01_build_ab_grm.py` | Construct an AB-GRM from PLINK genotypes and LD blocks |
| `02_summarize_blocks.R` | Summarize LD blocks, retained blocks and adaptive weights |
| `03_run_gblup_cv.R` | Evaluate conventional GBLUP and AB-GBLUP using shared cross-validation partitions |
| `04_plot_gs_metrics.R` | Calculate repeat-level prediction metrics and generate comparison plots |


## Dependencies

### Python

The AB-GRM construction script requires Python 3, NumPy, pandas,
pandas-plink, joblib, and Dask.

Recorded versions: Python 3.13.0, NumPy 2.2.2, pandas 2.2.3,
and joblib 1.5.3. Versions of pandas-plink and Dask are not specified.

### R

The analysis scripts require R and the following packages.
Recorded versions are:

- R 4.3.3
- sommer 4.3.2
- data.table 1.17.8
- dplyr 1.1.4
- ggplot2 3.5.2
- tidyr 1.3.1
- scales 1.4.0
- R.utils 2.13.0

The cross-validation script uses the mmer/vsr interface in sommer.

## Input data and naming

Use a consistent dataset prefix, illustrated below as `dataset`. Run all examples from the repository root. Example paths refer to user-supplied files, not bundled datasets.

| Input | Format |
| --- | --- |
| `data/dataset.bed`, `.bim`, `.fam` | PLINK binary genotypes with a common prefix |
| `data/dataset.blocks.det` | PLINK LD blocks from the matching marker dataset |
| `data/phenotypes.txt` | FID, IID and one selected trait in the first three columns |
| `data/G_text.grm.gz` | Conventional GBLUP GRM: four columns containing i, j, marker count and relationship value |
| `data/G_text.grm.id` | FID and IID, in the order used by the GCTA matrix |

SNP IDs must be unique. IIDs must be globally unique because the square GRM and CV script use IID, not the FID/IID pair, for matching. The LD block file must be ordered by chromosome and genomic position, with a header and PLINK's standard columns; the last column lists pipe-separated SNP IDs. Use matching genotype and LD-block files. SNPs not listed in blocks do not contribute to the AB-GRM; block SNP IDs absent from the genotype dataset are skipped by the original matching logic.

The four scripts do not perform read processing, variant calling, genotype QC, PLINK LD-block detection, or conventional GRM construction. Supply the resulting inputs and document the commands used for those upstream steps separately.

## 1. Build AB-GRM matrices

```bash
python 01_build_ab_grm.py \
  --bfile data/dataset \
  --blocks data/dataset.blocks.det \
  --out-prefix results/ab_grm/dataset_B500_V250 \
  --min-block-snp 500 \
  --min-valid-snp 250 \
  --n-jobs 4
```

The output directory is created automatically. Existing files with the same output prefix are overwritten, so use a distinct prefix for each parameter setting.

| Argument | Default | Meaning |
| --- | --- | --- |
| `--bfile` | Required | PLINK prefix without an extension |
| `--blocks` | Required | PLINK `.blocks.det` path |
| `--out-prefix` | `AB` | Prefix for both output files |
| `--min-block-snp` | `500` | B: SNP-count threshold for sequential block merging |
| `--min-valid-snp` | `250` | V: minimum valid SNP count for a local GRM |
| `--n-jobs` | min(allocated CPUs, 16) | Number of blocks processed concurrently |
| `--use-weight-cap` | Disabled | Apply an upper cap to raw weights |
| `--weight-cap-quantile` | `0.95` | Cap quantile when capping is enabled |

For the main-analysis grid, a Bash example is:

```bash
for B in 10 20 50 100 200 500 1000 2000 3000 5000 7000 10000; do
  V=$((B / 2))
  python 01_build_ab_grm.py \
    --bfile data/dataset \
    --blocks data/dataset.blocks.det \
    --out-prefix results/ab_grm/dataset_B${B}_V${V} \
    --min-block-snp "$B" --min-valid-snp "$V" --n-jobs 4 || exit 1
done
```

Adapt this grid to each dataset. A setting with no retained blocks raises an error; do not include its absent matrix in the CV configuration.

### Calculation details retained from v2.4

- Adjacent LD blocks are merged in input order within chromosomes. A remaining block at a chromosome boundary or file end is retained as a candidate even when it contains fewer than B SNPs; it must still pass subsequent V checks.
- Local VanRaden GRMs use SNPs with finite allele frequencies strictly between 0 and 1. missing genotypes are replaced by twice the relevant allele frequency before centering.
- Stability is evaluated using 50 random subsamples, each containing `floor(0.8 * n)` individuals sampled **without replacement**. Allele frequencies are estimated in each subsample, but every resulting matrix includes all individuals. SNP validity and the V threshold are checked again using subsample allele frequencies.
- A block is excluded when fewer than two subsampling matrices are valid. Stability is `numpy.var` (default `ddof=0`) of the Frobenius distances between valid subsampling matrices and their mean matrix. It is not the mean squared Frobenius deviation.
- The raw weight is `1 / (stability + 1e-8)`. Weights are optionally capped, normalized and used to combine local GRMs.
- The global seed is 999; each merged block uses seed `999 + its zero-based index`. Changing input block order can therefore change results.

These settings are constants in the script (`N_BOOT`, `SAMPLE_FRAC`, `RANDOM_SEED`, `EPS`), not command-line arguments. Historical variable names containing `boot` have been retained for compatibility; they refer to subsampling without replacement.

### Outputs and resources

Each run writes:

- `<prefix>_GRM.txt`: a square tab-delimited matrix with row and column IIDs, written to six decimal places.
- `<prefix>_block_statistics.txt`: retained-block coordinates, SNP counts, B/V, stability and raw/final normalized weights.

No separate `block_weights.txt` is produced; weights are included in the statistics file.

The n_snp column reports the number of matched SNPs before polymorphism filtering and may differ from the number used to calculate the local GRM.

The full genotype matrix is loaded into memory as uint8 (approximately individuals × SNPs bytes, excluding metadata and temporaries). Each concurrent block also retains up to 50 full subsampling GRMs and temporary arrays. Batching bounds memory across blocks, but does not eliminate within-block matrix storage. Reduce `--n-jobs` if necessary.

CPU allocation is read from `SLURM_CPUS_PER_TASK`, falling back to 4 outside SLURM. Explicit `--n-jobs` values are capped by that allocation, so the unchanged code uses at most four block workers outside SLURM unless the environment variable is set. Numerical-library threads are separate from block workers.

## 2. Run shared cross-validation

Edit the configuration near the top of `03_run_gblup_cv.R`:

```r
dataset_prefix <- "dataset"
pheno_file <- "data/phenotypes.txt"
pheno_header <- FALSE
trait_name <- "trait"
result_dir <- file.path("results", trait_name)
gcta_grm_file <- "data/G_text.grm.gz"
gcta_id_file <- "data/G_text.grm.id"
abg_dir <- "results/ab_grm"
```

The configured matrix directory matches the construction commands above. Keep `grm_config` columns aligned and include only existing matrices. The provided configuration lists a conventional GBLUP matrix and 12 B/V=2 AB-GRM settings.

```bash
Rscript 03_run_gblup_cv.R
```

Use a separate output directory for every dataset/trait combination. Phenotypes must occupy the third column. Use `pheno_header <- TRUE` for a file with column names; Missing phenotypes must use NA, not numeric sentinels such as -9. Only individuals present in the phenotype and every configured matrix are analyzed.

Defaults remain 20 repetitions, five folds and seed 123. All models in a run use the same individuals and fold assignments. Saved assignments are reused when compatible. Output suffixes remain `20rep_5fold`; if changing nrep or nfold, update filenames and the plotting pattern together.

Outputs include shared folds, sample IDs, GRM checks, per-model held-out predictions and fold-level metrics, and combined result tables. The CV script's accuracy summaries average fold-level correlations.

## 3. Summarize prediction metrics and plot

```bash
Rscript 04_plot_gs_metrics.R results/trait dataset
```

Arguments are the results directory and dataset prefix (defaults: `.` and `dataset`). Prefixes may contain letters, digits, underscores and hyphens. The directory must contain model-specific files such as:

```text
dataset_GBLUP_predictions_20rep_5fold.txt
dataset_500_250_ABG_GBLUP_predictions_20rep_5fold.txt
```

The script excludes combined all-model files and fold-level accuracy files. Each input must contain held-out predictions for one dataset and trait, with `id`, `y`, `pred`, `rep` and `fold` columns. It verifies consistent cross-model partitions.

Predictions from all test folds are pooled within each repeat. Repeat-level metrics are Pearson correlation (Accuracy), mean prediction minus observation (Bias), MSE, RMSE and MAE. Means and standard deviations across repeats are saved to:

- `GS_model_comparison_plots/GS_metrics_by_repeat.txt`
- `GS_model_comparison_plots/GS_metrics_summary.txt`

**`GS_metrics_summary.txt` is the summary used for the manuscript; pooling predictions within repeats differs from averaging fold-level correlations.** Accuracy, Bias, MSE and MAE plots are saved as PDF and PNG. The red dashed line indicates mean GBLUP performance. Plot jitter retains the original unseeded behavior.

## 4. Summarize blocks and weights

The block-statistics directory must also contain the original PLINK `.blocks.det` file. For the paths used above:

```bash
cp data/dataset.blocks.det results/ab_grm/dataset.blocks.det
Rscript 02_summarize_blocks.R results/ab_grm dataset dataset.blocks.det
```

Arguments are the input directory, dataset prefix and block filename (defaults: `.`, `dataset`, and `dataset.blocks.det`). The block filename is relative to the input directory. Statistics files must follow `<prefix>_B<number>_V<number>_block_statistics.txt`.

Four tabular outputs and four plots (each in PDF and PNG) are written to `block_statistics_summary/`. The script prefers `n_valid_snp` when available, otherwise uses `n_snp`. Normalized weights are read directly or derived from capped/raw weights using the original fallback logic. Effective block number is `1 / sum(normalized_weight^2)`. Top-weight shares sum the largest `ceiling(proportion * number_of_blocks)` weights.

## References
Chang CC, Chow CC, Tellier LC, Vattikuti S, Purcell SM, Lee JJ. Second-generation PLINK: rising to the challenge of larger and richer datasets. GigaScience. 2015;4:7. https://doi.org/10.1186/s13742-015-0047-8

Covarrubias-Pazaran G. Genome-Assisted Prediction of Quantitative Traits Using the R Package sommer. PLOS ONE. 2016;11(6):e0156744. https://doi.org/10.1371/journal.pone.0156744
