"""Reusable LAFA scoring harness. Builds a single prediction file over the 7401
LAFA query set (reranked scores for proteins present in our eval frame, KNN
fallback for the rest) and scores it with the EXACT LAFA cafaeval CLI invocation
against the official NK/LK/PK ground truths, extracting per-namespace f_micro_w.

LAFA invocation (per category):
  cafaeval <OBO> <predDir> <GT> -ia <IA> -out_dir <out> -toi <TOI>
           -prop fill -norm cafa -no_orphans   (PK also: -known <PK_known>)
"""
import subprocess
from pathlib import Path

import numpy as np

CAFA = "/home/frapercan/Thesis2/repositories/PROTEA/.venv/bin/cafaeval"
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
QFASTA = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_queries_7401.fasta"
KNN_FALLBACK = "/home/frapercan/Thesis2/protea-lafa-knn/predictions_7401.tsv"
GTD = "/tmp/claude-1000/-home-frapercan-Thesis2/afd2c43a-ede7-46dc-94dd-9745808d2194/scratchpad/lafa_gt"
TOI = f"{GTD}/groundtruth_terms_of_interest.txt"
NS = {"bpo": "biological_process", "mfo": "molecular_function", "cco": "cellular_component"}
NS_INV = {v: k for k, v in NS.items()}
GT = {"NK": f"{GTD}/groundtruth_NK.tsv", "LK": f"{GTD}/groundtruth_LK.tsv",
      "PK": f"{GTD}/groundtruth_PK.tsv"}
PK_KNOWN = f"{GTD}/groundtruth_PK_known.tsv"


def query_ids():
    ids = []
    with open(QFASTA) as fh:
        for line in fh:
            if line.startswith(">"):
                ids.append(line[1:].strip().split()[0])
    return ids


def load_fallback():
    """KNN fallback predictions keyed by protein -> list[(go, score)]."""
    fb = {}
    with open(KNN_FALLBACK) as fh:
        for line in fh:
            p, g, s = line.rstrip("\n").split("\t")
            fb.setdefault(p, []).append((g, float(s)))
    return fb


def build_pred_file(rows, out_path, fallback_prots=None):
    """rows: iterable of (protein, go, score) for eval-covered queries.
    For query proteins not covered, append KNN fallback predictions.
    fallback_prots: optional set; if None, computed as queries minus covered.
    """
    covered = set()
    out_path = Path(out_path)
    fb = load_fallback()
    with out_path.open("w") as fh:
        for p, g, s in rows:
            covered.add(p)
            fh.write(f"{p}\t{g}\t{s:.6f}\n")
        queries = set(query_ids())
        missing = (queries - covered) if fallback_prots is None else fallback_prots
        nfb = 0
        for p in missing:
            for g, s in fb.get(p, []):
                fh.write(f"{p}\t{g}\t{s:.6f}\n")
                nfb += 1
    return len(covered), nfb


def run_cafaeval(pred_file, category, out_dir):
    """Run the LAFA cafaeval CLI for one category GT. Returns
    {mfo,bpo,cco: f_micro_w}. Reads evaluation_best_f_micro_w.tsv."""
    pred_file = Path(pred_file)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = out_dir / "pred_dir"
    pred_dir.mkdir(exist_ok=True)
    # cafaeval iterates a directory; symlink the single pred file in
    link = pred_dir / pred_file.name
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(pred_file.resolve())
    cmd = [CAFA, OBO, str(pred_dir), GT[category], "-ia", IA,
           "-out_dir", str(out_dir), "-toi", TOI,
           "-prop", "fill", "-norm", "cafa", "-no_orphans"]
    if category == "PK":
        cmd += ["-known", PK_KNOWN]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    res_tsv = out_dir / "evaluation_best_f_micro_w.tsv"
    if p.returncode != 0 or not res_tsv.exists():
        return {"error": (p.stderr or p.stdout)[-600:]}
    out = {}
    with res_tsv.open() as fh:
        header = fh.readline().rstrip("\n").split("\t")
        i_ns = header.index("ns")
        i_f = header.index("f_micro_w")
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            ns = parts[i_ns]
            if ns in NS_INV:
                out[NS_INV[ns]] = round(float(parts[i_f]), 4)
    return out


def score_all(pred_file, work_dir):
    work_dir = Path(work_dir)
    res = {}
    for cat in ["NK", "LK", "PK"]:
        res[cat] = run_cafaeval(pred_file, cat, work_dir / f"results_{cat}")
    return res
