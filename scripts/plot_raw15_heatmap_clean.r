#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
})

in_file <- "/proj/yunligrp/users/qinwen/R_works/LLM_embedding/data/results/smoke_feature_embedding_correlations_raw15.tsv"
out_png <- "/proj/yunligrp/users/qinwen/R_works/LLM_embedding/data/results/smoke_feature_embedding_correlation_heatmap_raw15_clean.png"
top_n_embeddings <- NA_integer_

args <- commandArgs(trailingOnly = TRUE)
if (length(args) >= 1 && nzchar(args[1])) in_file <- args[1]
if (length(args) >= 2 && nzchar(args[2])) out_png <- args[2]
if (length(args) >= 3 && nzchar(args[3])) top_n_embeddings <- as.integer(args[3])

dt <- fread(in_file)
if (!all(c("feature", "embedding", "correlation") %in% names(dt))) {
  stop("Input table must have columns: feature, embedding, correlation")
}

feature_order <- c(
  "current_tobacco", "past_tobacco", "smokers_household",
  "exposure_home", "exposure_outside", "light_smoker_100",
  "age_started_former", "type_tobacco_former", "age_stopped",
  "stopped_6m", "age_started_current", "type_tobacco_current",
  "cigs_daily_current", "difficulty_1day", "tried_stop"
)

label_map <- data.table(
  feature = feature_order,
  label = c(
    "Current tobacco",
    "Past tobacco",
    "Smokers household",
    "Exposure home",
    "Exposure outside",
    "Light smoker 100",
    "Age started former",
    "Type tobacco former",
    "Age stopped",
    "Stopped 6m",
    "Age started current",
    "Type tobacco current",
    "Cigs daily current",
    "Difficulty 1day",
    "Tried stop"
  )
)

dt <- dt[feature %in% feature_order]
dt[, feature := factor(feature, levels = feature_order)]

if (!is.na(top_n_embeddings)) {
  emb_keep <- dt[, .(max_abs_corr = max(abs(correlation), na.rm = TRUE)), by = embedding][
    order(-max_abs_corr)
  ]
  if (nrow(emb_keep) == 0) stop("No embeddings available after filtering.")
  top_n_embeddings <- max(10L, min(top_n_embeddings, nrow(emb_keep)))
  keep_embeddings <- emb_keep[1:top_n_embeddings, embedding]
  dt <- dt[embedding %in% keep_embeddings]
}

# Cluster x-axis embeddings by correlation profile across features.
mat <- dcast(dt, feature ~ embedding, value.var = "correlation")
mat_m <- as.matrix(mat[, -1])
rownames(mat_m) <- as.character(mat$feature)
mat_m[is.na(mat_m)] <- 0
emb_hc <- hclust(dist(t(mat_m)), method = "ward.D2")
emb_order <- emb_hc$labels[emb_hc$order]
dt[, embedding := factor(embedding, levels = emb_order)]

dt <- merge(dt, label_map, by = "feature", all.x = TRUE)
dt[, label := factor(label, levels = label_map$label)]

p <- ggplot(dt, aes(x = embedding, y = label, fill = correlation)) +
  geom_tile(color = NA) +
  scale_fill_gradient2(
    low = "#2B6CB0",
    mid = "#F7FAFC",
    high = "#C53030",
    midpoint = 0,
    limits = c(-1, 1),
    oob = scales::squish
  ) +
  labs(
    x = if (is.na(top_n_embeddings)) "Embedding dimensions" else paste0("Embedding dimensions (top ", top_n_embeddings, " by |correlation|)"),
    y = NULL
  ) +
  theme_bw(base_size = 12) +
  theme(
    panel.grid = element_blank(),
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank(),
    axis.text.y = element_text(size = 11),
    legend.position = "none"
  )

ggsave(out_png, p, width = 8, height = 5, dpi = 320)
cat("Saved:", out_png, "\n")
