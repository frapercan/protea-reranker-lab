"""Leakage confirmation: re-score pooled-per-aspect f_micro_w for champion
(1-distance) and reranked (reranker_score) on (a) the FULL eval and (b) the
CLEAN eval with all train-positive (protein,go) pairs removed. If the reranker
lift survives on clean, the 7.4% pair overlap is not driving it.
"""
import json, subprocess, tempfile
from pathlib import Path
import pandas as pd, numpy as np

S = "/tmp/claude-1000/-home-frapercan-Thesis2/afd2c43a-ede7-46dc-94dd-9745808d2194/scratchpad"
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
PROTEA_PY = "/home/frapercan/Thesis2/repositories/PROTEA/.venv/bin/python"
NS = {"bpo": "biological_process", "mfo": "molecular_function", "cco": "cellular_component"}
DRIVER = '''
import json, signal
from cafaeval.evaluation import cafa_eval
signal.signal(signal.SIGTERM, signal.SIG_DFL)
df, dfs = cafa_eval("{obo}", "{pred_dir}", "{gt}", ia="{ia}", prop="fill", norm="cafa",
    no_orphans=True, max_terms=500, th_step=0.001, n_cpu=1, weighted_only=False)
json.dump({{k: v.reset_index().to_dict("records") for k, v in dfs.items()}}, open("{out}", "w"), default=str)
'''


def fmw(pred_dir, gt, asp, td):
    out = Path(td) / "r.json"; drv = Path(td) / "d.py"
    drv.write_text(DRIVER.format(obo=OBO, pred_dir=pred_dir, gt=gt, ia=IA, out=str(out)))
    p = subprocess.run([PROTEA_PY, str(drv)], capture_output=True, text=True, timeout=1800)
    if p.returncode != 0:
        return None
    raw = json.loads(out.read_text())
    for rec in raw.get("f_micro_w", []):
        if (rec.get("ns") or rec.get("namespace") or "") == NS[asp] and rec.get("f_micro_w") is not None:
            return round(float(rec["f_micro_w"]), 4)
    return None


def pooled(df, score_col, tag):
    res = {}
    with tempfile.TemporaryDirectory() as td:
        for asp in ["mfo", "bpo", "cco"]:
            d = df[df.aspect == asp]
            if len(d) == 0:
                continue
            cd = Path(td) / f"{tag}_{asp}"; pdir = cd / "pred"; pdir.mkdir(parents=True)
            with (pdir / "p.tsv").open("w") as fh:
                for p_, g_, s_ in zip(d.protein_accession, d.go_term_id, d[score_col]):
                    fh.write(f"{p_}\t{g_}\t{s_:.6f}\n")
            gt = cd / "gt.tsv"
            dp = d[d.label > 0]
            with gt.open("w") as fh:
                for p_, g_ in zip(dp.protein_accession, dp.go_term_id):
                    fh.write(f"{p_}\t{g_}\n")
            res[asp] = fmw(str(pdir), str(gt), asp, str(cd))
    res["mean"] = round(float(np.mean([v for v in res.values() if isinstance(v, float)])), 4)
    return res


def main():
    ev = pd.read_parquet(f"{S}/rerank_out/eval_scores.parquet")
    ev["champ"] = 1.0 - ev["distance"]
    tr = pd.read_parquet(f"{S}/ds227230/train.parquet", columns=["protein_accession", "go_term_id", "label"])
    trpos = set(zip(tr[tr.label > 0].protein_accession, tr[tr.label > 0].go_term_id))
    pair = list(zip(ev.protein_accession, ev.go_term_id))
    ev["seen_pos_in_train"] = [p in trpos for p in pair]
    clean = ev[~ev.seen_pos_in_train].copy()
    print(f"full eval rows={len(ev)} pos={(ev.label>0).sum()} | clean rows={len(clean)} pos={(clean.label>0).sum()} "
          f"(removed {(ev.label>0).sum()-(clean.label>0).sum()} train-seen positives)")
    out = {}
    for name, d in [("FULL", ev), ("CLEAN_no_trainpos", clean)]:
        ch = pooled(d, "champ", f"{name}_ch")
        rr = pooled(d, "reranker_score", f"{name}_rr")
        out[name] = {"champion": ch, "reranked": rr,
                     "lift_mean": round(rr["mean"] - ch["mean"], 4)}
        print(f"\n[{name}] champion mean={ch['mean']} {ch}")
        print(f"[{name}] reranked mean={rr['mean']} {rr}")
        print(f"[{name}] LIFT mean = {out[name]['lift_mean']}")
    Path(f"{S}/leak_check_overlap.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
