library(data.table)
library(tidyverse)

diabetes <- fread("/proj/yunligrp/users/qinwen/R_works/LLM_embedding/data/ukb_phenotype/extracted_pheno_diabetes.txt")
smoke <- fread("/proj/yunligrp/users/qinwen/R_works/LLM_embedding/data/ukb_phenotype/smoke_text_df.tsv")

##save test version
smoke_lite <- head(smoke, 100)
fwrite(smoke_lite, "/proj/yunligrp/users/qinwen/R_works/LLM_embedding/data/ukb_phenotype/smoke_text_df_lite.tsv", sep = "\t")

names(diabetes) <- c("IID", "diabetes")
table(diabetes$diabetes)

df <- filter(diabetes, diabetes == 1 | diabetes == 0)
df <- left_join(df, smoke, by = "IID")
head(df)
fwrite(df, "/proj/yunligrp/users/qinwen/R_works/LLM_embedding/data/ukb_phenotype/diabetes_smoke_text_df.txt.gz", sep = "\t")


# smoke
# ref <- fread("/proj/yunligrp/users/qinwen/R_works/LLM_embedding/data/ukb_phenotype/extracted_pheno_smoke.txt")
# names(ref)
# head(filter(ref, !is.na(f.1249.0.0)),20)

# filter(smoke, IID == 2248630)
