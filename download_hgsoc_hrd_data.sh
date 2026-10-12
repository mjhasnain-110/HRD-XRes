#!/usr/bin/env bash
# =====================================================================
# Cross-resolution HRD & platinum-response predictor (HGSOC)
# Data download script
# Usage: bash download_hgsoc_hrd_data.sh [target_dir]
# Requires: wget, curl, jq, unzip
# Approx. total size: ~15-25 GB (SPECTRUM h5ad files are the bulk)
# =====================================================================
set -uo pipefail
ROOT="${1:-./hgsoc_hrd_data}"
mkdir -p "$ROOT"/{tcga,scrna,spatial,bulk_validation,priors}
cd "$ROOT" || exit 1

# Helper: mirror a GEO series' supplementary folder
geo_suppl () {
  local gse="$1"; local stub="${gse:0:${#gse}-3}nnn"
  mkdir -p "$gse"
  wget -c -r -np -nd -e robots=off -R "index.html*" -P "$gse" \
    "https://ftp.ncbi.nlm.nih.gov/geo/series/${stub}/${gse}/suppl/"
}
# Helper: GEO series matrix (processed expression + sample annotations)
geo_matrix () {
  local gse="$1"; local stub="${gse:0:${#gse}-3}nnn"
  mkdir -p "$gse"
  wget -c -r -np -nd -e robots=off -R "index.html*" -P "$gse" \
    "https://ftp.ncbi.nlm.nih.gov/geo/series/${stub}/${gse}/matrix/"
}

# ---------------------------------------------------------------------
# 1. TCGA (training data: expression + HRD labels + outcome)
#    Source: GDC PanCanAtlas & PanCan-DDR-2018 publication pages
# ---------------------------------------------------------------------
cd tcga
# Batch-corrected RNA-seq (all cancers; subset to OV). ~1.9 GB
wget -c -O EBPlusPlusAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.tsv \
  https://api.gdc.cancer.gov/data/3586c0da-64d0-4b74-a449-5ff4d9136611
# Curated clinical outcomes (OS, PFI) - use this first for survival
wget -c -O TCGA-CDR-SupplementalTableS1.xlsx \
  https://api.gdc.cancer.gov/data/1b5f413e-a8d1-4d10-92eb-7c4ae739ed81
# Clinical with follow-up
wget -c -O clinical_PANCAN_patient_with_followup.tsv \
  https://api.gdc.cancer.gov/data/0fc78496-818b-4896-bd83-52db1f533c5c
# Somatic mutations (BRCA1/2 somatic status)
wget -c -O mc3.v0.2.8.PUBLIC.maf.gz \
  https://api.gdc.cancer.gov/data/1c8cfe5f-e52d-41ba-94da-f15ea1337efc
# ABSOLUTE purity/ploidy (purity is a key confounder - adjust for it)
wget -c -O TCGA_mastercalls.abs_tables_JSedit.fixed.txt \
  https://api.gdc.cancer.gov/data/4f277128-f793-4354-a13d-30cc7fe9f6b5
# ABSOLUTE segments (to recompute HRD with scarHRD if desired)
wget -c -O TCGA_mastercalls.abs_segtabs.fixed.txt \
  https://api.gdc.cancer.gov/data/0f4f5701-7b61-41ae-bda9-2805d1ca9781
# Sample QC whitelist
wget -c -O merged_sample_quality_annotations.tsv \
  https://api.gdc.cancer.gov/data/1a7d7be8-675d-4e60-a105-19d4121bdebf
# DDR resource (Knijnenburg 2018): HRD scores, DDR gene alteration calls
wget -c -O TCGA_DDR_Data_Resources.zip \
  https://api.gdc.cancer.gov/data/d9fdd899-bffe-4325-b0ce-58cbf9924c0a
wget -c -O TCGA_DDR_Data_Resources.xlsx \
  https://api.gdc.cancer.gov/data/5dd5a767-8f9f-4579-abee-b1306a4d0ad2
wget -c -O TCGA_DDR_Data_Resources_README.pdf \
  https://api.gdc.cancer.gov/data/1350f59c-2b4c-4bde-b03b-7f81e6cdbd3c
unzip -o TCGA_DDR_Data_Resources.zip
# Optional: GDC STAR counts for TCGA-OV via UCSC Xena GDC hub (raw counts,
# more OV samples than EB++). If filenames changed, browse:
# https://xenabrowser.net/datapages/?cohort=GDC%20TCGA%20Ovarian%20Cancer%20(OV)
wget -c https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-OV.star_counts.tsv.gz \
  || echo "[warn] Xena STAR counts filename changed - see Xena browser"
wget -c https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-OV.clinical.tsv.gz \
  || echo "[warn] Xena clinical filename changed - see Xena browser"
cd ..

# ---------------------------------------------------------------------
# 2. Single-cell
# ---------------------------------------------------------------------
cd scrna
# 2a. MSK SPECTRUM (Vazquez-Garcia et al., Nature 2022) - CELLxGENE h5ad
#     Resolve current file URLs via the CELLxGENE Curation API
COLL=4796c91c-9d8f-4692-be43-347b1727f9d8
mkdir -p spectrum && cd spectrum
curl -s "https://api.cellxgene.cziscience.com/curation/v1/collections/${COLL}" \
  > spectrum_collection.json
jq -r '.datasets[] | .title as $t | .assets[] | select(.filetype=="H5AD") | "\($t)\t\(.url)"' \
  spectrum_collection.json | tee spectrum_h5ad_urls.tsv
cut -f2 spectrum_h5ad_urls.tsv | xargs -n1 wget -c
cd ..
# 2b. Paired pre/post-NACT scRNA (Zhang et al., Sci Adv 2022)
geo_suppl GSE165897
cd ..

# ---------------------------------------------------------------------
# 3. Spatial (Visium)
# ---------------------------------------------------------------------
cd spatial
# 3a. Stur et al., iScience 2022: 12 pre-NACT HGSOC, 6 excellent / 6 poor responders
geo_suppl GSE189843
# 3b. Denisenko et al., Nat Commun 2024: 8 post-NACT HGSOC with CRS 1/2/3
geo_suppl GSE211956
cd ..

# ---------------------------------------------------------------------
# 4. External bulk validation (series matrices)
# ---------------------------------------------------------------------
cd bulk_validation
geo_matrix GSE9891     # AOCS / Tothill (n~285), survival, residual disease
geo_matrix GSE140082   # ICON7 trial cohort (n~380), PFS/OS
geo_matrix GSE63885    # chemotherapy response cohort (check labels)
geo_matrix GSE51373    # platinum sensitive vs resistant (check labels)
cd ..

# ---------------------------------------------------------------------
# 5. Priors for the pathway-constrained encoder
# ---------------------------------------------------------------------
cd priors
wget -c https://reactome.org/download/current/ReactomePathways.gmt.zip
unzip -o ReactomePathways.gmt.zip
cd ..

echo "Done. Data in: $(pwd)"
