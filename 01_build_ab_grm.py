#!/usr/bin/env python3
"""Construct an adaptive block genomic relationship matrix from PLINK data.

Inputs: a PLINK .bed/.bim/.fam prefix and a matching PLINK .blocks.det file.
Outputs: <prefix>_GRM.txt and <prefix>_block_statistics.txt.
Run python build_ab_grm.py --help for command-line options."""

import os
import gc
import argparse

import numpy as np
import pandas as pd

from pandas_plink import read_plink
from joblib import Parallel, delayed

############################################################
# PARAMETERS
############################################################

N_BOOT = 50
SAMPLE_FRAC = 0.8

############################################################
# Parallel settings
#
# Efficiency 2:
# Use thread-based parallelism to avoid copying the full
# genotype matrix across processes.
#
# Efficiency 7:
# Record CPUs allocated by SLURM, while allowing block-level
# parallelism to be controlled separately.
############################################################

ALLOCATED_CPUS = int(
    os.environ.get("SLURM_CPUS_PER_TASK", 4)
)

RANDOM_SEED = 999

EPS = 1e-8

############################################################
# COMMAND-LINE ARGUMENTS
#
# Efficiency 3:
# Input files are provided by command-line arguments.
############################################################

parser = argparse.ArgumentParser(
    description="Construct Adaptive Block GRM from PLINK bfile and LD blocks."
)

parser.add_argument(
    "--bfile",
    required=True,
    help="PLINK binary file prefix, without .bed/.bim/.fam"
)

parser.add_argument(
    "--blocks",
    required=True,
    help="PLINK LD block file, usually *.blocks.det"
)

parser.add_argument(
    "--out-prefix",
    default="AB",
    help="Output prefix for AB-GRM and block statistics"
)

parser.add_argument(
    "--use-weight-cap",
    action="store_true",
    help="Enable raw weight capping before normalization"
)

parser.add_argument(
    "--weight-cap-quantile",
    type=float,
    default=0.95,
    help="Upper quantile for raw weight capping when --use-weight-cap is enabled"
)

parser.add_argument(
    "--min-block-snp",
    type=int,
    default=500,
    help="Minimum SNP number for merging fragmented LD blocks"
)

parser.add_argument(
    "--min-valid-snp",
    type=int,
    default=250,
    help="Minimum matched valid SNP number required for local GRM"
)

parser.add_argument(
    "--n-jobs",
    type=int,
    default=None,
    help=(
        "Number of LD blocks processed in parallel. "
        "Default: min(SLURM_CPUS_PER_TASK, 16)"
    )
)

args = parser.parse_args()

# Validate settings before loading large genotype files.
if args.min_block_snp < 1:
    parser.error("--min-block-snp must be at least 1.")
if args.min_valid_snp < 0:
    parser.error("--min-valid-snp must be nonnegative.")
if not 0 < args.weight_cap_quantile <= 1:
    parser.error("--weight-cap-quantile must be in (0, 1].")
if ALLOCATED_CPUS < 1:
    parser.error("SLURM_CPUS_PER_TASK must be a positive integer.")
for input_file in [args.bfile + ext for ext in (".bed", ".bim", ".fam")] + [args.blocks]:
    if not os.path.isfile(input_file):
        parser.error(f"Input file does not exist: {input_file}")
if not args.out_prefix or args.out_prefix.endswith(os.sep):
    parser.error("--out-prefix must include a filename prefix.")
os.makedirs(os.path.dirname(os.path.abspath(args.out_prefix)), exist_ok=True)

############################################################
# Efficiency 7:
# Limit simultaneous block calculations. Sixteen jobs are
# used by default because full 56-way block parallelism can
# saturate memory bandwidth without improving throughput.
############################################################

if args.n_jobs is None:

    N_JOBS = min(
        ALLOCATED_CPUS,
        16
    )

else:

    if args.n_jobs < 1:

        raise ValueError(
            "--n-jobs must be at least 1."
        )

    N_JOBS = min(
        args.n_jobs,
        ALLOCATED_CPUS
    )

BATCH_SIZE = N_JOBS

print(
    "SLURM allocated CPUs:",
    ALLOCATED_CPUS
)

print(
    "Block-level parallel jobs:",
    N_JOBS
)

print(
    "Blocks retained per result batch:",
    BATCH_SIZE
)

############################################################
# COMPACT GENOTYPE ENCODING
#
# Efficiency 5:
# Genotypes 0, 1 and 2 are stored as uint8. Missing values
# are represented by 255 and restored to NaN when an LD block
# is extracted.
############################################################

MISSING_GENOTYPE_CODE = np.uint8(
    255
)


def encode_genotype_chunk(chunk):

    missing = np.isnan(
        chunk
    )

    encoded = np.rint(
        np.nan_to_num(
            chunk,
            nan=0.0
        )
    ).astype(
        np.uint8
    )

    encoded[
        missing
    ] = MISSING_GENOTYPE_CODE

    return encoded

############################################################
# 1. LOAD GENOTYPE
############################################################

print("\nLoading genotype...")

############################################################
# Efficiency 3:
# Read PLINK bfile prefix from command-line argument.
############################################################

bim, fam, bed = read_plink(
    args.bfile
)

# Unique SNP IDs prevent silent dictionary collisions. Unique IIDs are also
# required by the square-matrix format and downstream CV script.
if bim.snp.astype(str).duplicated().any():
    raise ValueError("Duplicate SNP IDs in .bim; unique SNP IDs are required.")
if fam.iid.astype(str).duplicated().any():
    raise ValueError("Duplicate IIDs in .fam; IIDs must be globally unique.")
if int(len(fam) * SAMPLE_FRAC) < 1:
    raise ValueError("Too few individuals for the configured subsampling fraction.")

############################################################
# Efficiency 5:
# Encode each Dask chunk directly as uint8 before the full
# genotype matrix is materialized. This avoids retaining a
# complete float32 genotype matrix in memory.
#
# The final X matrix has dimensions:
# individual x SNP
############################################################

X = bed.T.map_blocks(
    encode_genotype_chunk,
    dtype=np.uint8
).compute()

snp_ids = bim.snp.astype(
    str
).to_numpy(
    copy=True
)

ids = fam.iid.astype(
    str
).to_numpy(
    copy=True
)

############################################################
# Release PLINK metadata and the lazy BED object after the
# compact genotype matrix has been loaded.
############################################################

del bim
del fam
del bed

gc.collect()

print(
    "Compact genotype matrix:",
    X.shape
)

print(
    "Genotype dtype:",
    X.dtype
)

print(
    "Genotype memory:",
    round(
        X.nbytes / 1024**3,
        2
    ),
    "GB"
)

############################################################
# Efficiency 1:
# Build SNP-to-index dictionary for fast block matching.
############################################################

snp_to_idx = {
    snp: i for i, snp in enumerate(snp_ids)
}

print("Genotype matrix:", X.shape)

############################################################
# 2. READ PLINK BLOCKS
############################################################

print("\nReading PLINK LD blocks...")

raw_blocks = []

block_chr = []
block_bp1 = []
block_bp2 = []

############################################################
# Efficiency 3:
# Read PLINK LD block file from command-line argument.
############################################################

with open(args.blocks) as f:

    ########################################################
    # skip header
    ########################################################

    next(f)

    for line in f:

        line = line.strip()

        if line == "":
            continue

        parts = line.split()

        ####################################################
        # PLINK block format
        ####################################################

        chr_id = parts[0]
        bp1 = int(parts[1])
        bp2 = int(parts[2])

        snp_string = parts[-1]

        snps = snp_string.split("|")

        raw_blocks.append(snps)

        block_chr.append(chr_id)
        block_bp1.append(bp1)
        block_bp2.append(bp2)

print("Raw blocks:", len(raw_blocks))

############################################################
# 3. MERGE FRAGMENTED BLOCKS
#
# Modification 3:
# Merge small LD blocks only within the same chromosome.
############################################################

print("\nMerging fragmented blocks...")

merged_blocks = []

merged_chr = []
merged_bp1 = []
merged_bp2 = []

current_block = []

current_chr = None
current_bp1 = None
current_bp2 = None

for i, block in enumerate(raw_blocks):

    chr_i = block_chr[i]
    bp1_i = block_bp1[i]
    bp2_i = block_bp2[i]

    ########################################################
    # start a new merged block
    ########################################################

    if current_chr is None:

        current_chr = chr_i
        current_bp1 = bp1_i
        current_bp2 = bp2_i
        current_block = []

    ########################################################
    # chromosome changed: save current block first
    ########################################################

    if chr_i != current_chr:

        if len(current_block) > 0:

            merged_blocks.append(current_block)

            merged_chr.append(current_chr)
            merged_bp1.append(current_bp1)
            merged_bp2.append(current_bp2)

        ####################################################
        # start a new chromosome-specific block
        ####################################################

        current_chr = chr_i
        current_bp1 = bp1_i
        current_bp2 = bp2_i
        current_block = []

    ########################################################
    # merge block within the same chromosome
    ########################################################

    current_block.extend(block)

    current_bp2 = bp2_i

    ########################################################
    # save when enough SNPs are accumulated
    ########################################################

    if len(current_block) >= args.min_block_snp:

        merged_blocks.append(current_block)

        merged_chr.append(current_chr)
        merged_bp1.append(current_bp1)
        merged_bp2.append(current_bp2)

        ####################################################
        # reset
        ####################################################

        current_block = []

        current_chr = None
        current_bp1 = None
        current_bp2 = None

############################################################
# save remaining block
############################################################

if len(current_block) > 0:

    merged_blocks.append(current_block)

    merged_chr.append(current_chr)
    merged_bp1.append(current_bp1)
    merged_bp2.append(current_bp2)

print("Merged blocks:", len(merged_blocks))

############################################################
# EXTRACT BLOCK GENOTYPES
#
# Efficiency 5:
# Convert only the currently used LD block from compact uint8
# storage to float32. Missing code 255 is restored to NaN.
############################################################

def extract_block_genotypes(idx):

    X_block = X[
        :,
        idx
    ].astype(
        np.float32
    )

    X_block[
        X_block == float(
            MISSING_GENOTYPE_CODE
        )
    ] = np.nan

    return X_block

############################################################
# 4. VANRADEN LOCAL GRM
#
# Modification 2:
# Ignore missing genotypes when estimating p and impute
# missing values as 2p before centering.
############################################################

def compute_grm(X_block):

    ########################################################
    # allele frequency
    ########################################################

    p = np.nanmean(
        X_block,
        axis=0
    ) / 2.0

    ########################################################
    # polymorphic SNPs
    ########################################################

    valid = np.isfinite(p) & (p > 0) & (p < 1)

    X_block = X_block[:, valid]

    p = p[valid]

    ########################################################
    # too few SNPs
    ########################################################

    if X_block.shape[1] < args.min_valid_snp:

        return None

    ########################################################
    # mean imputation for missing genotypes
    ########################################################

    X_block = np.where(
        np.isnan(X_block),
        2 * p,
        X_block
    )

    ########################################################
    # center genotype
    ########################################################

    Z = X_block - 2 * p

    ########################################################
    # denominator
    ########################################################

    denom = 2 * np.sum(
        p * (1 - p)
    )

    if denom <= 0:

        return None

    ########################################################
    # VanRaden GRM
    ########################################################

    G = (Z @ Z.T) / denom

    ########################################################
    # numerical check
    ########################################################

    if np.isnan(G).any():

        return None

    return G

############################################################
# 4.1 VANRADEN GRM WITH GIVEN ALLELE FREQUENCY
#
# Modification 1:
# Compute a full-size GRM using externally estimated p.
#
# Modification 2:
# Impute missing genotypes as 2p before centering.
############################################################

def compute_grm_with_given_p(X_all, p):

    ########################################################
    # keep polymorphic SNPs
    ########################################################

    valid = np.isfinite(p) & (p > 0) & (p < 1)

    X_use = X_all[:, valid]

    p_use = p[valid]

    ########################################################
    # too few SNPs
    ########################################################

    if X_use.shape[1] < args.min_valid_snp:

        return None

    ########################################################
    # mean imputation for missing genotypes
    ########################################################

    X_use = np.where(
        np.isnan(X_use),
        2 * p_use,
        X_use
    )

    ########################################################
    # center genotype using supplied p
    ########################################################

    Z = X_use - 2 * p_use

    ########################################################
    # denominator
    ########################################################

    denom = 2 * np.sum(
        p_use * (1 - p_use)
    )

    if denom <= 0:

        return None

    ########################################################
    # VanRaden GRM
    ########################################################

    G = (Z @ Z.T) / denom

    ########################################################
    # numerical check
    ########################################################

    if np.isnan(G).any():

        return None

    return G

############################################################
# 5. RELATIONSHIP STABILITY
#
# Modification 1:
# Subsampled individuals only estimate p_boot.
# GRMs are computed for all individuals to keep matrices aligned.
#
# Efficiency 2:
# Use a block-specific random seed for reproducible parallel
# subsampling sampling.
############################################################

def relationship_stability(
        X_block,
        n_boot=N_BOOT,
        sample_frac=SAMPLE_FRAC,
        seed=None):

    n = X_block.shape[0]

    ########################################################
    # block-specific random generator
    ########################################################

    rng = np.random.default_rng(seed)

    G_list = []

    ########################################################
    # pre-filter SNPs using full-sample allele frequency
    #
    # Modification 2:
    # estimate p while ignoring missing genotypes
    ########################################################

    p_full = np.nanmean(
        X_block,
        axis=0
    ) / 2.0

    valid_full = np.isfinite(p_full) & (p_full > 0) & (p_full < 1)

    X_use = X_block[:, valid_full]

    ########################################################
    # too few SNPs
    ########################################################

    if X_use.shape[1] < args.min_valid_snp:

        return None

    ########################################################
    # subsampling individuals
    ########################################################

    for i in range(n_boot):

        ####################################################
        # Efficiency 2:
        # reproducible individual subsampling
        ####################################################
        
        idx = rng.choice(
            n,
            size=int(n * sample_frac),
            replace=False
        )

        ####################################################
        # estimate p_boot from sampled individuals
        ####################################################

        X_sub = X_use[idx, :]
        ####################################################
        # Modification 2:
        # estimate p_boot while ignoring missing genotypes
        ####################################################
        p_boot = np.nanmean(
            X_sub,
            axis=0
        ) / 2.0

        ####################################################
        # compute full-size GRM using p_boot
        ####################################################

        G_boot = compute_grm_with_given_p(
            X_use,
            p_boot
        )

        if G_boot is None:
            continue

        G_list.append(G_boot)

    ########################################################
    # failed
    ########################################################

    if len(G_list) < 2:

        return None

    ########################################################
    # mean GRM
    ########################################################

    G_mean = np.mean(
        G_list,
        axis=0
    )

    ########################################################
    # Frobenius distance
    ########################################################

    distances = []

    for G in G_list:

        d = np.linalg.norm(
            G - G_mean,
            ord="fro"
        )

        distances.append(d)

    ########################################################
    # stability variance
    ########################################################

    return np.var(distances)

############################################################
# 5.1 PROCESS ONE LD BLOCK
#
# Efficiency 2:
# Wrap per-block GRM, stability and weight calculation for
# parallel execution.
############################################################

def process_one_block(i, block):

    ########################################################
    # fast SNP matching
    ########################################################

    idx = np.array(
        [snp_to_idx[snp] for snp in block if snp in snp_to_idx],
        dtype=np.int64
    )

    n_match = len(idx)

    ########################################################
    # skip small blocks
    ########################################################

    if n_match < args.min_valid_snp:

        return None

    ########################################################
    # genotype subset
    #
    # Efficiency 5:
    # Decode only the current LD block from compact uint8
    # storage to a temporary float32 matrix.
    ########################################################

    X_block = extract_block_genotypes(
        idx
    )

    ########################################################
    # local GRM
    ########################################################

    G_block = compute_grm(X_block)

    if G_block is None:

        return None

    ########################################################
    # stability
    ########################################################

    stability = relationship_stability(
        X_block,
        seed=RANDOM_SEED + i
    )

    if stability is None:

        return None

    ########################################################
    # adaptive weight
    ########################################################

    weight = 1.0 / (
        stability + EPS
    )

    ########################################################
    # statistics
    ########################################################

    stat = {

        "block": i + 1,

        "chr": merged_chr[i],

        "start_bp": merged_bp1[i],

        "end_bp": merged_bp2[i],

        # Matched SNP count before polymorphism filtering (legacy column).
        "n_snp": n_match,

        "min_block_snp": args.min_block_snp,

        "min_valid_snp": args.min_valid_snp,

        "stability": stability,

        "raw_weight": weight

    }

    ########################################################
    # Efficiency 6:
    # Return the original merged-block index so that the same
    # block can be identified if a second GRM pass is required
    # when weight capping is enabled.
    ########################################################

    return i, G_block, weight, stat

############################################################
# COMPUTE LOCAL GRM WITHOUT SUBSAMPLING
#
# Efficiency 6:
# This function is only needed when weight capping is enabled.
# Stability and raw weights are calculated in the first pass;
# local GRMs are then recomputed without repeating subsampling.
############################################################

def compute_local_grm_only(block_index):

    block = merged_blocks[
        block_index
    ]

    idx = np.array(
        [
            snp_to_idx[snp]
            for snp in block
            if snp in snp_to_idx
        ],
        dtype=np.int64
    )

    if idx.size < args.min_valid_snp:

        return None

    X_block = extract_block_genotypes(
        idx
    )

    G_block = compute_grm(
        X_block
    )

    del X_block

    return G_block

############################################################
# 6. CONSTRUCT LOCAL GRMS IN BOUNDED BATCHES
#
# Efficiency 6:
# Only BATCH_SIZE block results are retained at one time.
# Without weight capping, each local GRM is immediately
# multiplied by its raw weight and added to a running sum.
############################################################

print(
    "\nConstructing local GRMs in bounded parallel batches..."
)

valid_block_indices = []

weights = []

block_statistics = []

############################################################
# Use float64 for the running weighted sum. This matrix is
# small relative to the genotype matrix and reduces numerical
# accumulation error across thousands of blocks.
############################################################

G_weighted_sum = np.zeros(
    (
        X.shape[0],
        X.shape[0]
    ),
    dtype=np.float64
)

total_merged_blocks = len(
    merged_blocks
)

for batch_start in range(
        0,
        total_merged_blocks,
        BATCH_SIZE):

    batch_end = min(
        batch_start + BATCH_SIZE,
        total_merged_blocks
    )

    print(
        "Processing merged blocks:",
        batch_start + 1,
        "-",
        batch_end,
        "/",
        total_merged_blocks
    )

    batch_results = Parallel(
        n_jobs=N_JOBS,
        prefer="threads",
        verbose=10
    )(
        delayed(
            process_one_block
        )(
            i,
            merged_blocks[i]
        )
        for i in range(
            batch_start,
            batch_end
        )
    )

    for result in batch_results:

        if result is None:

            continue

        block_index, G_block, weight, stat = result

        valid_block_indices.append(
            block_index
        )

        weights.append(
            weight
        )

        block_statistics.append(
            stat
        )

        ####################################################
        # Fast path:
        # When capping is disabled, accumulate the raw
        # weighted local GRM immediately.
        ####################################################

        if not args.use_weight_cap:

            np.multiply(
                G_block,
                np.float32(weight),
                out=G_block
            )

            G_weighted_sum += G_block

        ####################################################
        # Release the local GRM after accumulation.
        ####################################################

        del G_block

    del batch_results

    gc.collect()

############################################################
# 7. CHECK
############################################################

print(
    "\nValid blocks:",
    len(valid_block_indices)
)

if len(valid_block_indices) == 0:

    raise ValueError(
        "No valid blocks."
    )

############################################################
# 8. OPTIONAL CAP AND NORMALIZE WEIGHTS
############################################################

raw_weights = np.asarray(
    weights,
    dtype=np.float64
)

if args.use_weight_cap:

    weight_cap = np.quantile(
        raw_weights,
        args.weight_cap_quantile
    )

    final_raw_weights = np.minimum(
        raw_weights,
        weight_cap
    )

    weight_capping_used = True

else:

    weight_cap = np.nan

    final_raw_weights = raw_weights.copy()

    weight_capping_used = False

weight_sum = np.sum(
    final_raw_weights
)

if weight_sum <= 0 or not np.isfinite(weight_sum):

    raise ValueError(
        "Invalid weight sum before normalization."
    )

weights = final_raw_weights / weight_sum

############################################################
# Add final weight information to block statistics.
############################################################

for i in range(len(block_statistics)):

    block_statistics[i][
        "weight_capping_used"
    ] = weight_capping_used

    block_statistics[i][
        "weight_after_cap"
    ] = final_raw_weights[i]

    block_statistics[i][
        "normalized_weight"
    ] = weights[i]

############################################################
# 9. FINAL AB-GRM
############################################################

if args.use_weight_cap:

    ########################################################
    # Weight-capping path:
    # GRMs were not accumulated during the first pass because
    # capped weights were not known. Recompute only local GRMs;
    # subsampling stability is not repeated.
    ########################################################

    print(
        "\nRecomputing local GRMs using capped weights..."
    )

    G_weighted_sum.fill(
        0.0
    )

    n_valid_blocks = len(
        valid_block_indices
    )

    for batch_start in range(
            0,
            n_valid_blocks,
            BATCH_SIZE):

        batch_end = min(
            batch_start + BATCH_SIZE,
            n_valid_blocks
        )

        positions = list(
            range(
                batch_start,
                batch_end
            )
        )

        local_grms = Parallel(
            n_jobs=N_JOBS,
            prefer="threads",
            verbose=10
        )(
            delayed(
                compute_local_grm_only
            )(
                valid_block_indices[position]
            )
            for position in positions
        )

        for position, G_block in zip(
                positions,
                local_grms):

            if G_block is None:

                raise ValueError(
                    "Local GRM failed during capped-weight "
                    f"pass for block "
                    f"{valid_block_indices[position] + 1}."
                )

            np.multiply(
                G_block,
                np.float32(
                    final_raw_weights[position]
                ),
                out=G_block
            )

            G_weighted_sum += G_block

            del G_block

        del local_grms

        gc.collect()

############################################################
# Normalize the accumulated raw-weighted GRM.
############################################################

G_final = (
    G_weighted_sum / weight_sum
).astype(
    np.float32
)

del G_weighted_sum

gc.collect()

############################################################
# 10. SAVE FINAL GRM
############################################################

print(f"\nSaving {args.out_prefix}_GRM.txt")

G_df = pd.DataFrame(
    G_final,
    index=ids,
    columns=ids
)

G_df.to_csv(
    f"{args.out_prefix}_GRM.txt",
    sep="\t",
    float_format="%.6f"
)

############################################################
# 11. SAVE BLOCK STATISTICS
############################################################

print(f"\nSaving {args.out_prefix}_block_statistics.txt")

stat_df = pd.DataFrame(
    block_statistics
)

stat_df.to_csv(
    f"{args.out_prefix}_block_statistics.txt",
    sep="\t",
    index=False
)

############################################################
# 12. SUMMARY
############################################################

print("\n========== SUMMARY ==========")

print("Individuals:", X.shape[0])

print("Total SNP:", X.shape[1])

print(
    "Valid blocks:",
    len(valid_block_indices)
)

print("Minimum SNPs for block merging:",
      args.min_block_snp)

print("Minimum valid SNPs for local GRM:",
      args.min_valid_snp)

print("Mean block weight:",
      np.mean(weights))

print("Mean stability:",
      stat_df["stability"].mean())

############################################################
# Efficiency summary:
# Report memory-related execution settings.
############################################################

print(
    "Genotype storage dtype:",
    X.dtype
)

print(
    "Genotype memory:",
    round(
        X.nbytes / 1024**3,
        2
    ),
    "GB"
)

print(
    "Parallel block jobs:",
    N_JOBS
)

print(
    "Block result batch size:",
    BATCH_SIZE
)

print(
    "Local GRMs retained across all blocks:",
    False
)

############################################################
# Modification 4 summary:
# Report weight-capping status and final weight distribution.
############################################################

print("Weight capping used:",
      weight_capping_used)

print("Weight cap quantile:",
      args.weight_cap_quantile)

print("Raw weight cap:",
      weight_cap)

print("Max normalized weight:",
      np.max(weights))

print("Effective number of blocks:",
      1.0 / np.sum(weights ** 2))

print("=============================")

############################################################
# DONE
############################################################

print("\nAB-GRM finished successfully.")
