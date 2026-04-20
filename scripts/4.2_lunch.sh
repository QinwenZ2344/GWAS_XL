# module add anaconda/2024.02
# conda create -n text_embedding python=3.10 -y
# conda activate text_embedding
# pip install -r /proj/yunligrp/users/qinwen/R_works/LLM_embedding/requirements.txt



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

