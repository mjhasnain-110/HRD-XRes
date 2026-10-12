#!/usr/bin/env python
"""Apply the trained HRD-XRes ensemble to new data.

Input: an .h5ad (obs = samples/cells/spots, var = gene symbols) or a genes x samples TSV.
  --value-type counts | log2 | linear    (default counts)
Output: CSV with pred_hrd, p_hr, p_brca, p_plat, risk, pred_purity per sample.
Example:
  python scripts/predict.py --input my_cohort.tsv --value-type log2 --output my_cohort_predictions.csv
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import anndata as ad
import pandas as pd

from hrdx.data import load_features, model_input
from hrdx.trainer import ensemble_predict, get_device, load_ensemble
from hrdx.utils import load_config

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True)
ap.add_argument("--output", required=True)
ap.add_argument("--value-type", default="counts", choices=["counts", "log2", "linear"])
ap.add_argument("--variant", default="full")
ap.add_argument("--config", default=None)
args = ap.parse_args()

cfg = load_config(args.config)
feat = load_features(cfg["paths"]["processed"])
if args.input.endswith(".h5ad"):
    a = ad.read_h5ad(args.input)
else:
    df = pd.read_csv(args.input, sep="\t", index_col=0)
    df = df.groupby(df.index.astype(str)).sum() if args.value_type == "counts" else df.groupby(df.index.astype(str)).max()
    a = ad.AnnData(X=df.T.values.astype("float32"), obs=pd.DataFrame(index=df.columns.astype(str)),
                   var=pd.DataFrame(index=df.index))
X, cov = model_input(a, feat["genes"], value_type=args.value_type)
print(f"{a.n_obs} profiles; model gene coverage {cov:.1%}")
if cov < cfg["features"]["min_external_gene_coverage"]:
    sys.exit(f"Gene coverage {cov:.1%} is below the {cfg['features']['min_external_gene_coverage']:.0%} minimum; "
             "check that rows are gene symbols and the file is complete.")
device = get_device(cfg["training"]["device"])
p = ensemble_predict(load_ensemble(cfg["paths"]["models"], args.variant, device), X, device)
p.index = a.obs_names
p.to_csv(args.output)
print(f"Predictions written to {args.output}")
