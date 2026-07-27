"""True-frame cafaeval scoring harness, replicating score_the_extras_trueframe.py's DRIVER.

Board frame: lab obo + IA, prop=fill, norm=cafa, no_orphans, toi restriction.
PK adds exclude=groundtruth_PK_known.tsv (evaluation.nf:279). LK has no -known.
Decides by f_micro_w on biological_process, max over the threshold sweep.
"""
import json, subprocess, tempfile, collections
from pathlib import Path

ROOT = Path("/home/frapercan/Thesis2")
OBO = str(ROOT / "protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo")
IA_F = str(ROOT / "protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv")
GT_DIR = ROOT / "repositories/protea-reranker-lab/results/sparse_classifier/lafa_gt"
TOI = str(GT_DIR / "groundtruth_terms_of_interest.txt")
PY = str(ROOT / "repositories/PROTEA/.venv/bin/python")

# obo maps
def load_obo():
    par = collections.defaultdict(set); ns = {}; alt = {}
    cur = None
    for line in open(OBO):
        line = line.rstrip("\n")
        if line == "[Term]":
            cur = None
        elif line.startswith("id: GO:"):
            cur = line[4:]
        elif cur and line.startswith("namespace: "):
            ns[cur] = line[11:]
        elif cur and line.startswith("is_a: GO:"):
            par[cur].add(line[6:].split(" ! ")[0].strip())
        elif cur and line.startswith("relationship: part_of GO:"):
            par[cur].add(line[len("relationship: part_of "):].split(" ! ")[0].strip())
        elif cur and line.startswith("alt_id: GO:"):
            alt[line[8:]] = cur
    return par, ns, alt

DRIVER = '''
import json, signal
from cafaeval.evaluation import cafa_eval
signal.signal(signal.SIGTERM, signal.SIG_DFL)
kw=dict(ia="{ia}",prop="fill",norm="cafa",no_orphans=True,toi_file="{toi}",
        max_terms=None,th_step=0.01,n_cpu=4,weighted_only=False)
{exclude_line}
df,dfs=cafa_eval("{obo}","{pd}","{gt}",**kw)
out={{}}
for k,v in dfs.items(): out[k]=v.reset_index().to_dict(orient="records")
json.dump(out,open("{o}","w"),default=str)
'''

def score_cell(rows, gt_rows, known_file=None, timeout=7200):
    """rows: iterable of (protein, term, score). gt_rows: iterable of (protein, term).
    known_file: path to exclude tsv (PK) or None (LK). Returns f_micro_w on BP, max over thresholds."""
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "pd"; d.mkdir()
        with (d / "p.tsv").open("w") as fh:
            for p, g, v in rows:
                fh.write(f"{p}\t{g}\t{float(v):.6f}\n")
        gt = Path(td) / "gt.tsv"
        with gt.open("w") as fh:
            for p, g in gt_rows:
                fh.write(f"{p}\t{g}\n")
        raw = Path(td) / "r.json"; drv = Path(td) / "d.py"
        excl = f'kw["exclude"]="{known_file}"' if known_file else ""
        drv.write_text(DRIVER.format(obo=OBO, pd=str(d), gt=str(gt), ia=IA_F, toi=TOI,
                                     o=str(raw), exclude_line=excl))
        r = subprocess.run([PY, str(drv)], capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            return {"f_micro_w": None, "err": r.stderr[-2000:]}
        best = None
        data = json.loads(raw.read_text())
        for rec in data.get("f_micro_w", []):
            if (rec.get("ns") or "") == "biological_process" and rec.get("f_micro_w") is not None:
                if best is None or float(rec["f_micro_w"]) > best["f_micro_w"]:
                    best = {"f_micro_w": round(float(rec["f_micro_w"]), 5),
                            "pr": round(float(rec.get("pr", 0)), 5),
                            "rc": round(float(rec.get("rc", 0)), 5),
                            "tau": rec.get("tau")}
        return best or {"f_micro_w": None, "err": "no BP row"}

def read_gt(cell):
    """cell in {LK,PK}: returns list of (protein, term) for aspect P (BP)."""
    f = GT_DIR / f"groundtruth_{cell}.tsv"
    out = []
    with f.open() as fh:
        next(fh)
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3 and p[2] == "P":
                out.append((p[0], p[1]))
    return out

KNOWN = {"LK": None, "PK": str(GT_DIR / "groundtruth_PK_known.tsv"), "NK": None}
