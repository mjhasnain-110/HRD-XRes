#!/usr/bin/env Rscript
# Runs the published IdentifiHR model (Weir et al., Commun Med 2026) on every
# <name>_counts.csv.gz in the given folder (genes x samples, raw counts, HGNC symbols)
# and writes <name>_identifihr.csv (Sample, hrPrediction, predictionProbability = P(HRD)).
#
# One-time install (inside the R used by Rscript):
#   install.packages(c("devtools", "BiocManager"))
#   BiocManager::install(c("edgeR", "SummarizedExperiment"))
#   devtools::install_github("DavidsonGroup/IdentifiHR")

args <- commandArgs(trailingOnly = TRUE)
indir <- if (length(args) >= 1) args[1] else "Results/identifihr"
suppressPackageStartupMessages(library(IdentifiHR))

files <- list.files(indir, pattern = "_counts\\.csv\\.gz$", full.names = TRUE)
if (length(files) == 0) stop("No *_counts.csv.gz files in ", indir)

for (f in files) {
  message("IdentifiHR: ", basename(f))
  y <- read.csv(gzfile(f), row.names = 1, check.names = FALSE)
  y <- as.matrix(y)
  storage.mode(y) <- "numeric"
  pc <- suppressWarnings(processCounts(y = y, geneIds = "HGNC"))
  present <- tryCatch(sum(rownames(y) %in% IdentifiHR:::modelGeneId$hgnc_symbol), error = function(e) NA)
  pr <- predictHr(pc)
  pr$genes_present <- present
  write.csv(pr, sub("_counts\\.csv\\.gz$", "_identifihr.csv", f), row.names = FALSE)
}
message("IdentifiHR done.")
