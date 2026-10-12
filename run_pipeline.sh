#!/usr/bin/env bash
# =====================================================================
# HRD-XRes full pipeline. Run from the project root:
#   cd ~/Mirza_Analysis/Ovarian_Q1 && bash run_pipeline.sh            # all steps
#   bash run_pipeline.sh 7 13                                          # steps 7 to 13 only
# Every step logs to Logs/<step>.log; training steps resume from saved checkpoints.
# =====================================================================
set -euo pipefail
cd "$(dirname "$0")"
FROM=${1:-0}
TO=${2:-13}
CFG=config/config.yaml
run () { local n=$1; shift; if (( n >= FROM && n <= TO )); then echo "=== step $n: $* ==="; "$@"; fi; }

run 0  Rscript scripts/R/export_curatedOvarianData.R Inputs/hgsoc_hrd_data/bulk_validation/curatedOvarianData 40
run 1  python scripts/01a_compute_hrd_scar.py $CFG
run 1  python scripts/01_prepare_tcga.py $CFG
run 2  python scripts/02_prepare_spectrum.py $CFG
run 3  python scripts/03_prepare_spatial.py $CFG
run 4  python scripts/04_prepare_bulk.py $CFG
run 5  python scripts/05_prepare_gse165897.py $CFG
run 6  python scripts/06_build_features.py $CFG
run 7  python scripts/07_cross_validation.py --config $CFG
run 8  python scripts/08_train_final.py --config $CFG
run 9  python scripts/09_validate_bulk.py $CFG
run 10 python scripts/10_validate_singlecell.py $CFG
run 11 python scripts/11_validate_spatial.py $CFG
run 12 python scripts/12_interpret.py $CFG
run 13 python scripts/13_make_figures.py $CFG
echo "Pipeline finished. Results in Results/, models in Models/, logs in Logs/."
