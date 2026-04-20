# GWAS_XL

`GWAS_XL` contains scripts for building text-style phenotype representations from UK Biobank fields and generating frozen biomedical language-model embeddings for downstream analysis. The current workflow focuses on smoking-related phenotype extraction, conversion to participant-level text summaries, and PubMedBERT embedding generation, with example downstream alignment to diabetes labels.

## Overview

The repository currently supports a simple pipeline:

1. Extract selected UK Biobank phenotype columns from raw tab files.
2. Convert smoking-related fields into person-level natural-language summaries.
3. Align those text summaries with downstream outcomes such as diabetes.
4. Generate fixed text embeddings using a Hugging Face biomedical encoder.

This project is designed for cluster execution on Longleaf and currently uses several hard-coded absolute paths. If you run it elsewhere, update the paths in the scripts before launching jobs.

## Repository Layout

```text
.
├── data/
│   ├── ukb_phenotype/      # extracted phenotype tables and text tables
│   └── results/            # embedding outputs and derived results
├── scripts/
│   ├── 1_extract_pheno.sh  # extract smoking/diabetes/COPD/FEV-FVC phenotype columns
│   ├── 2_clean_2text_smoke.r
│   ├── 3_align_diease.r
│   ├── 4_embedding.py
│   ├── 4.2_lunch.sh        # example local launch command
│   └── 4.3_slurm.sh        # example SLURM submission script
├── pubmed_embd/            # alternative / exploratory embedding scripts and analyses
├── model/
├── GWAS/
└── README.md
```

## Requirements

### Python

- Python 3.10+
- `torch`
- `transformers`
- `tqdm`

Install with:

```bash
pip install -r requirements.txt
```

### R

The R scripts use packages such as:

- `data.table`
- `tidyverse`

Install them in your R environment if needed.

## Main Workflow

### 1. Extract phenotype columns

Use `scripts/1_extract_pheno.sh` to pull selected fields from UK Biobank tab files.

Outputs include:

- `data/ukb_phenotype/extracted_pheno_smoke.txt`
- `data/ukb_phenotype/extracted_pheno_diabetes.txt`
- `data/ukb_phenotype/COPD_new_IID.txt`
- `data/ukb_phenotype/FEV_FVC_new_IID.txt`

Before running, update:

- `InDir` to point to your UK Biobank source files
- `OutDir` to point to your project output directory

Run:

```bash
bash scripts/1_extract_pheno.sh
```

### 2. Convert smoking phenotypes to text

Use `scripts/2_clean_2text_smoke.r` to convert the extracted smoking phenotype table into participant-level text descriptions.

Default input:

- `data/ukb_phenotype/extracted_pheno_smoke.txt`

Default output:

- `data/ukb_phenotype/smoke_text_df.tsv`

Run:

```bash
Rscript scripts/2_clean_2text_smoke.r
```

Or provide custom input/output paths:

```bash
Rscript scripts/2_clean_2text_smoke.r input.tsv output.tsv
```

### 3. Align text with downstream labels

Use `scripts/3_align_diease.r` to join diabetes labels with the smoking text table.

Outputs include:

- `data/ukb_phenotype/diabetes_smoke_text_df.txt.gz`
- `data/ukb_phenotype/smoke_text_df_lite.tsv`

Run:

```bash
Rscript scripts/3_align_diease.r
```

### 4. Generate text embeddings

Use `scripts/4_embedding.py` to create fixed embeddings from any IID + text table.

Example:

```bash
python scripts/4_embedding.py \
  --input_file data/ukb_phenotype/diabetes_smoke_text_df.txt.gz \
  --output_pt data/ukb_phenotype/diabetes_smoke_pubmedbert_emb.pt \
  --iid_col IID \
  --text_col Text \
  --keep_cols diabetes \
  --model_name microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext \
  --batch_size 32 \
  --max_length 256 \
  --pooling mean
```

The saved `.pt` file contains:

- participant IDs
- embedding matrix
- model metadata
- optional preserved columns from `--keep_cols`

## Cluster Usage

Example launch helpers are included:

- `scripts/4.2_lunch.sh`: example command-line launch
- `scripts/4.3_slurm.sh`: SLURM submission script for GPU execution

These scripts assume:

- Longleaf module system
- a Conda environment named `text_embedding`
- project-specific absolute paths under `/proj/yunligrp/users/qinwen/...`

Adjust them to your environment before use.

## Notes

- Many scripts currently use hard-coded absolute paths; portability will improve if these are replaced with relative paths or command-line arguments.
- The repository contains both pipeline scripts in `scripts/` and exploratory analysis code in `pubmed_embd/`.
- Some filenames currently preserve the original naming in the repo, such as `3_align_diease.r` and `4.2_lunch.sh`.
