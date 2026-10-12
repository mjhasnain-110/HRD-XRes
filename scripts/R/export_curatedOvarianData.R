#!/usr/bin/env Rscript
# Export serous ovarian tumour cohorts from curatedOvarianData (Bioconductor) for external validation.
# TCGA cohorts are excluded (TCGA is the training set).
# Usage: Rscript scripts/R/export_curatedOvarianData.R <output_dir> [min_samples_with_OS]
suppressPackageStartupMessages({
  if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager", repos = "https://cloud.r-project.org")
  if (!requireNamespace("curatedOvarianData", quietly = TRUE))
    BiocManager::install("curatedOvarianData", ask = FALSE, update = FALSE)
  library(curatedOvarianData)
  library(Biobase)
})

args <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args) >= 1) args[1] else "Inputs/hgsoc_hrd_data/bulk_validation/curatedOvarianData"
min_os <- if (length(args) >= 2) as.integer(args[2]) else 40
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

items <- data(package = "curatedOvarianData")$results[, "Item"]
items <- items[grepl("_eset$", items) & !grepl("TCGA", items)]
summary <- data.frame()
for (nm in items) {
  env <- new.env()
  data(list = nm, package = "curatedOvarianData", envir = env)
  e <- get(nm, envir = env)
  ph <- pData(e)
  keep <- rep(TRUE, nrow(ph))
  if ("histological_type" %in% colnames(ph)) keep <- keep & !is.na(ph$histological_type) & ph$histological_type == "ser"
  if ("sample_type" %in% colnames(ph)) keep <- keep & !is.na(ph$sample_type) & ph$sample_type == "tumor"
  e <- e[, keep]
  ph <- pData(e)
  n_os <- if (all(c("days_to_death", "vital_status") %in% colnames(ph))) sum(!is.na(ph$days_to_death) & !is.na(ph$vital_status)) else 0
  summary <- rbind(summary, data.frame(cohort = nm, n_serous_tumour = ncol(e), n_os = n_os, n_genes = nrow(e)))
  if (n_os < min_os) next
  ex <- exprs(e)
  ex <- ex[rowSums(is.na(ex)) == 0, , drop = FALSE]
  con <- gzfile(file.path(out_dir, paste0(nm, "_expr.tsv.gz")), "w")
  write.table(data.frame(gene = rownames(ex), ex, check.names = FALSE), con, sep = "\t", quote = FALSE, row.names = FALSE)
  close(con)
  write.table(data.frame(sample = rownames(ph), ph, check.names = FALSE),
              file.path(out_dir, paste0(nm, "_pheno.tsv")), sep = "\t", quote = FALSE, row.names = FALSE)
  message(sprintf("exported %s: %d samples, %d genes, %d with OS", nm, ncol(ex), nrow(ex), n_os))
}
write.table(summary, file.path(out_dir, "curated_summary.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
