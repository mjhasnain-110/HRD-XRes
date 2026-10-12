#!/usr/bin/env python
"""00b - Update the Peng et al. (2014) 230-gene HRD signature to current gene symbols.

Input : Inputs/hgsoc_hrd_data/priors/peng2014_HRD_signature_230genes.tsv (extracted from Suppl. Table 1)
Uses  : NCBI Homo_sapiens.gene_info.gz (downloaded automatically) - official symbols + synonyms
Output: Inputs/hgsoc_hrd_data/priors/peng2014_HRD_signature_230genes_current.tsv
        columns: gene (current symbol), gene_original, mapping, fold changes, mean_log2fc, direction_in_HRD
"""
import gzip
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from hrdx.utils import download, get_logger, inp, load_config

cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else None)
log = get_logger("00b_update_peng_symbols", cfg)
pri = inp(cfg, "priors")
sig = pd.read_csv(pri / "peng2014_HRD_signature_230genes.tsv", sep="\t")
gi = download("https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz",
              pri / "Homo_sapiens.gene_info.gz", log)

with gzip.open(gi, "rt") as fh:
    info = pd.read_csv(fh, sep="\t", usecols=["Symbol", "Synonyms", "type_of_gene"], dtype=str)
current = set(info["Symbol"])
syn = {}
for s, synonyms in zip(info["Symbol"], info["Synonyms"]):
    if synonyms and synonyms != "-":
        for a in synonyms.split("|"):
            syn.setdefault(a.upper(), set()).add(s)

out, status = [], []
for g in sig["gene"].astype(str):
    if g in current:
        out.append(g); status.append("current")
    elif g.upper() in syn and len(syn[g.upper()]) == 1:
        out.append(next(iter(syn[g.upper()]))); status.append("updated_from_synonym")
    elif g.upper() in syn:
        out.append(""); status.append("ambiguous:" + "|".join(sorted(syn[g.upper()])))
    else:
        out.append(""); status.append("not_found")
sig.insert(0, "gene_original", sig.pop("gene"))
sig.insert(0, "gene", out)
sig.insert(2, "mapping", status)
dup = sig["gene"].ne("") & sig["gene"].duplicated(keep=False)
if dup.any():
    log.warning(f"Several original symbols map to the same current symbol:\n{sig.loc[dup, ['gene_original', 'gene']]}")
outp = pri / "peng2014_HRD_signature_230genes_current.tsv"
sig.to_csv(outp, sep="\t", index=False)
log.info(f"Mapping summary: {pd.Series(status).str.split(':').str[0].value_counts().to_dict()}")
upd = sig[sig["mapping"] != "current"]
if len(upd):
    log.info("Renamed / unresolved genes:\n" + upd[["gene_original", "gene", "mapping"]].to_string(index=False))
log.info(f"Saved {outp}")
