#!/bin/bash
#SBATCH -p l40-gpu
#SBATCH --job-name=4_embedding
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --qos=gpu_access
#SBATCH --output=/proj/yunligrp/users/qinwen/R_works/LLM_embedding/scripts/logs/4_embedding_out.txt
#SBATCH --error=/proj/yunligrp/users/qinwen/R_works/LLM_embedding/scripts/logs/4_embedding_error.txt

module add anaconda/2024.02
conda activate text_embedding
cd /proj/yunligrp/users/qinwen/R_works/LLM_embedding/scripts
python /proj/yunligrp/users/qinwen/R_works/LLM_embedding/scripts/4_embedding.py \
  --input_file /proj/yunligrp/users/qinwen/R_works/LLM_embedding/data/ukb_phenotype/diabetes_smoke_text_df.txt.gz \
  --output_pt /proj/yunligrp/users/qinwen/R_works/LLM_embedding/data/ukb_phenotype/diabetes_smoke_pubmedbert_emb.pt \
  --iid_col IID \
  --text_col Text \
  --keep_cols diabetes \
  --model_name microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext \
  --batch_size 32 \
  --max_length 256 \
  --pooling mean

