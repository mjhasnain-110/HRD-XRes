"""GEO series-matrix and platform (GPL) parsing."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import download, ensure_dir, open_text


def _split_line(line: str) -> list[str]:
    return [p.strip().strip('"') for p in line.rstrip("\n").split("\t")]


def parse_series_matrix(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Return (sample_meta [samples x fields], expression [probes x samples], platform_id).

    Characteristics fields of the form "key: value" are expanded into one column per key.
    """
    meta_rows: dict[str, list[list[str]]] = {}
    expr_lines: list[list[str]] = []
    platform = ""
    in_table = False
    with open_text(path) as fh:
        for line in fh:
            if line.startswith("!series_matrix_table_begin"):
                in_table = True
                continue
            if line.startswith("!series_matrix_table_end"):
                in_table = False
                continue
            if in_table:
                expr_lines.append(_split_line(line))
                continue
            if line.startswith("!Series_platform_id"):
                platform = _split_line(line)[1]
            if line.startswith("!Sample_"):
                parts = _split_line(line)
                meta_rows.setdefault(parts[0][len("!Sample_"):], []).append(parts[1:])

    gsm = meta_rows["geo_accession"][0]
    meta = pd.DataFrame(index=gsm)
    for key, rows in meta_rows.items():
        if key == "characteristics_ch1":
            for row in rows:
                for gsm_id, val in zip(gsm, row):
                    if ":" in val:
                        k, v = val.split(":", 1)
                        meta.loc[gsm_id, f"char::{k.strip().lower()}"] = v.strip()
            continue
        for i, row in enumerate(rows):
            col = key if i == 0 else f"{key}_{i}"
            if len(row) == len(gsm):
                meta[col] = row

    expr = pd.DataFrame()
    if len(expr_lines) > 1:
        header = expr_lines[0]
        body = [r for r in expr_lines[1:] if len(r) == len(header)]
        expr = pd.DataFrame([r[1:] for r in body], index=[r[0] for r in body], columns=header[1:])
        expr = expr.apply(pd.to_numeric, errors="coerce")
    return meta, expr, platform


def platform_url_stub(gpl: str) -> str:
    num = gpl[3:]
    return "GPLnnn" if len(num) <= 3 else f"GPL{num[:-3]}nnn"


SYMBOL_COLUMNS = ["gene symbol", "symbol", "gene_symbol", "ilmn_gene", "genesymbol", "gene"]


def load_platform_symbols(gpl: str, platform_dir: Path, logger=None) -> pd.Series:
    """Probe ID -> gene symbol for a GEO platform, downloading the annotation if needed."""
    ensure_dir(platform_dir)
    stub = platform_url_stub(gpl)
    base = f"https://ftp.ncbi.nlm.nih.gov/geo/platforms/{stub}/{gpl}"
    candidates = [
        (f"{base}/annot/{gpl}.annot.gz", platform_dir / f"{gpl}.annot.gz", "!platform_table_begin"),
        (f"{base}/soft/{gpl}_family.soft.gz", platform_dir / f"{gpl}_family.soft.gz", "!platform_table_begin"),
    ]
    for url, dest, marker in candidates:
        try:
            download(url, dest, logger)
        except Exception as exc:  # noqa: BLE001
            if logger:
                logger.info(f"{gpl}: {url} unavailable ({exc}); trying next source")
            continue
        rows, in_tab = [], False
        with open_text(dest) as fh:
            for line in fh:
                if line.lower().startswith(marker):
                    in_tab = True
                    continue
                if line.lower().startswith("!platform_table_end"):
                    break
                if in_tab:
                    rows.append(line.rstrip("\n").split("\t"))
        if len(rows) < 2:
            continue
        tab = pd.DataFrame(rows[1:], columns=rows[0])
        cols = {c.lower(): c for c in tab.columns}
        id_col = cols.get("id", tab.columns[0])
        sym_col = next((cols[c] for c in SYMBOL_COLUMNS if c in cols), None)
        if sym_col is None:
            sym_col = next((c for c in tab.columns if "symbol" in c.lower()), None)
        if sym_col is None and "gene_assignment" in cols:
            tab["__sym"] = tab[cols["gene_assignment"]].str.split("//").str[1].str.strip()
            sym_col = "__sym"
        if sym_col is None:
            continue
        sym = tab[sym_col].astype(str).str.split("///").str[0].str.strip()
        sym = sym.where(~sym.isin(["", "---", "nan", "NA"]))
        s = pd.Series(sym.values, index=tab[id_col].astype(str).values).dropna()
        if logger:
            logger.info(f"{gpl}: {len(s)} probes mapped to symbols using column '{sym_col}'")
        return s
    raise RuntimeError(f"Could not obtain probe annotation for {gpl}")


def to_log2(expr: pd.DataFrame) -> pd.DataFrame:
    """Put microarray values on log2 scale if they are linear intensities."""
    v = expr.values[np.isfinite(expr.values)]
    if v.size and np.nanpercentile(v, 99) > 50:
        return np.log2(expr.clip(lower=1.0))
    return expr


def find_field(meta: pd.DataFrame, patterns: list[str], exclude: list[str] | None = None) -> str | None:
    exclude = exclude or []
    for pat in patterns:
        for c in meta.columns:
            lc = c.lower()
            if re.search(pat, lc) and not any(re.search(e, lc) for e in exclude):
                return c
    return None
