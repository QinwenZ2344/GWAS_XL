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
├── scripts/
│   ├── 2_clean_2text_smoke.r    #convert smoking variable to text
│   └── 3_align_diease.r
├── pubmed_embd/            # alternative / exploratory embedding scripts and analyses
├── 1_smoke_pubmedbert_embeddings.py   #generate embeddings
└── README.md
```
