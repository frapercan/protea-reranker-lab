"""Is the prior-knowledge channel thin BY DESIGN or BY IMPLEMENTATION?

The question, from the author: we lose LK-BPO and PK-BPO, the two cells where the
protein already knows something. What we own for "known terms -> new BP terms" is ONE
feature family, `association_*`, and it is built by `build_go_cooccurrence`: a
first-order co-occurrence COUNT. Measured worth: AUC ~0.60; LOFO +0.0037 on LK-BPO,
+0.0174 on PK-BPO. Meanwhile we are +0.072 / +0.076 behind TransFew on those cells.

Counting cannot express "THIS molecular function implies THIS process". It blurs
"co-occur because one implies the other" with "co-occur because both are frequent".
So: does a mechanism that reads the known SET, rather than marginal counts, extract
more from exactly the same information?

WHY PK AND NOT LK: the benchmark freezes t0 knowledge in `groundtruth_PK_known.tsv`
(EntryID/term/aspect). It covers PK targets by design; only 103 of 523 LK-BPO proteins
appear in it, and the t0 corpus needed for LK lives in the live DB. PK is fully frozen
(4,455 proteins), so PK is where this can be answered tonight, honestly. Verified while
scoping: of the 103 LK proteins present, the median count of known BP terms is 0 and
NONE has any, which confirms the LK definition (has MF/CC, lacks BP) from data.

ARMS. All predict a protein's NEW BP terms from its t0-known terms ALONE. No sequence,
no embedding, no neighbours. Every arm sees exactly the same training information, so
the comparison is a comparison of MECHANISMS.

  C  counting      what we ship: score(p,t) = f(co-occurrence counts of t with each
                   k in K(p)), max and sum, learned from the TRAIN split.
  S  set-level     Jaccard-kNN in KNOWN-TERM space: find the train proteins whose known
                   set most resembles K(p) and transfer their BP terms, similarity
                   weighted. Reads the set as a set; not a marginal.
  O  oracle        score = the true label. The ceiling of this channel: what ANY
                   function of nothing-but-prior-knowledge could reach.

READ IT AS: O tells you whether the channel exists at all. S minus C tells you whether
our counting implementation is leaving it on the table.

HONEST BOUNDS, stated before the numbers:
 * Both C and S learn from the SAME train-split pairs (K(p) -> BP(p)). Those BP terms
   are post-t0, so this is leakage-symmetric: fine for ranking two mechanisms against
   each other, NOT a production estimate. In production the co-occurrence table is
   built from the far larger t0 corpus. Read S-minus-C, not the absolutes.
 * The split is by PROTEIN, so no protein informs its own prediction.
 * AUC is reported because it is the natural per-candidate measure here, but the
   campaign's own rule stands: AUC ordered the reranker levers OPPOSITE to f_micro_w.
   So f_micro_w on the same harness is reported too, and it is the one that decides.
"""
import json, subprocess, tempfile, time, collections
from pathlib import Path
import numpy as np

W = Path("/home/frapercan/Thesis2/storage/cooc_experiment")
REL = Path("/home/frapercan/Thesis2/CAFA_forever/data/releases/Sep_2025_Mar_2026")
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
PY = "/home/frapercan/Thesis2/repositories/PROTEA/.venv/bin/python"
SEED = 42


def load_pairs(f, aspect=None):
    d = collections.defaultdict(set)
    for i, l in enumerate(open(f)):
        if i == 0 and l.startswith("EntryID"):
            continue
        p = l.rstrip("\n").split("\t")
        if len(p) < 2:
            continue
        if aspect and (len(p) < 3 or p[2] != aspect):
            continue
        d[p[0]].add(p[1])
    return d


known_all = {}
for i, l in enumerate(open(REL / "groundtruth_PK_known.tsv")):
    if i == 0:
        continue
    p = l.rstrip("\n").split("\t")
    if len(p) >= 3:
        known_all.setdefault(p[0], set()).add((p[1], p[2]))
truth_bp = load_pairs(REL / "groundtruth_PK.tsv", aspect="P")

prots = sorted(set(known_all) & set(truth_bp))
rng = np.random.default_rng(SEED)
rng.shuffle(prots)
cut = int(0.7 * len(prots))
TR, TE = set(prots[:cut]), set(prots[cut:])
print(f"proteins with frozen t0 knowledge AND new BP answers: {len(prots):,} "
      f"(train {len(TR):,} / test {len(TE):,})", flush=True)

K = {p: {t for t, a in known_all[p]} for p in prots}          # known terms, all aspects
BP = {p: truth_bp[p] for p in prots}                           # the new BP answers

# candidate BP vocabulary = the BP terms seen in TRAIN (never peek at test answers)
vocab = sorted({t for p in TR for t in BP[p]})
vidx = {t: i for i, t in enumerate(vocab)}
print(f"candidate BP vocabulary from TRAIN only: {len(vocab):,} terms", flush=True)

# ---- co-occurrence table from TRAIN pairs: count(k, t) --------------------------
cooc = collections.defaultdict(collections.Counter)
for p in TR:
    for k in K[p]:
        cooc[k].update(BP[p])
kfreq = collections.Counter(k for p in TR for k in K[p])
tfreq = collections.Counter(t for p in TR for t in BP[p])
n_tr = len(TR)
print(f"co-occurrence built: {len(cooc):,} known terms -> BP counters", flush=True)

# ---- Jaccard-kNN in known-term space -------------------------------------------
TRl = sorted(TR)
tr_sets = [K[p] for p in TRl]
inv = collections.defaultdict(list)
for j, s in enumerate(tr_sets):
    for k in s:
        inv[k].append(j)


MAXJ = 0.5   # NO near-duplicates: a neighbour more similar than this is excluded


def knn_transfer(kq, topn=30):
    """Transfer BP terms from train proteins whose KNOWN set resembles kq, EXCLUDING
    near-duplicates (Jaccard >= MAXJ), so nothing can be memorised."""
    hits = collections.Counter()
    for k in kq:
        for j in inv.get(k, ()):
            hits[j] += 1
    if not hits:
        return {}
    sims = []
    for j, inter in hits.items():
        u = len(kq) + len(tr_sets[j]) - inter
        if u:
            jac = inter / u
            if jac < MAXJ:            # drop near-duplicates
                sims.append((jac, j))
    sims.sort(reverse=True)
    sims = sims[:topn]
    tot = sum(s for s, _ in sims) or 1.0
    out = collections.Counter()
    for s, j in sims:
        for t in BP[TRl[j]]:
            out[t] += s
    return {t: v / tot for t, v in out.items()}


def score_arm(name, rows):
    """rows: (protein, term, score). Scored by the same cafaeval, full BP truth."""
    DRIVER = '''
import json, signal
from cafaeval.evaluation import cafa_eval
signal.signal(signal.SIGTERM, signal.SIG_DFL)
df,dfs=cafa_eval("{obo}","{pd}","{gt}",ia="{ia}",prop="fill",norm="cafa",
    no_orphans=True,max_terms=500,th_step=0.001,n_cpu=1,weighted_only=False)
out={{}}
for k,v in dfs.items(): out[k]=v.reset_index().to_dict(orient="records")
json.dump(out,open("{o}","w"),default=str)
'''
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "pd"
        d.mkdir()
        with (d / f"{name}.tsv").open("w") as fh:
            for p, t, v in rows:
                fh.write(f"{p}\t{t}\t{v:.6f}\n")
        gt = Path(td) / "gt.tsv"
        with gt.open("w") as fh:
            for p in TE:
                for t in BP[p]:
                    fh.write(f"{p}\t{t}\n")
        raw = Path(td) / "r.json"
        drv = Path(td) / "d.py"
        drv.write_text(DRIVER.format(obo=OBO, pd=str(d), gt=str(gt), ia=IA, o=str(raw)))
        r = subprocess.run([PY, str(drv)], capture_output=True, text=True, timeout=5400)
        if r.returncode != 0:
            print(f"  cafaeval FAILED {name}: {r.stderr[-300:]}", flush=True)
            return None
        best = None
        for rec in json.loads(raw.read_text()).get("f_micro_w", []):
            if (rec.get("ns") or "") == "biological_process" and rec.get("f_micro_w") is not None:
                best = rec
        return {k: round(float(best[k]), 4) for k in ("f_micro_w", "pr_micro_w", "rc_micro_w", "tau", "cov_max")} if best else None


def auc(y, s):
    y = np.asarray(y)
    s = np.asarray(s)
    if y.sum() == 0 or y.sum() == len(y):
        return None
    o = np.argsort(s)
    r = np.empty(len(s), float)
    r[o] = np.arange(1, len(s) + 1)
    pos = r[y == 1].sum()
    n1, n0 = y.sum(), (1 - y).sum()
    return float((pos - n1 * (n1 + 1) / 2) / (n1 * n0))


def top1_jac(kq):
    hits = collections.Counter()
    for k in kq:
        for j in inv.get(k, ()):
            hits[j] += 1
    best = 0.0
    for j, inter in hits.items():
        u = len(kq) + len(tr_sets[j]) - inter
        if u:
            best = max(best, inter / u)
    return best


TE = {p for p in TE if top1_jac(K[p]) < MAXJ}
print(f"TEST restricted to proteins with NO near-duplicate (top-1 Jaccard < {MAXJ}): {len(TE):,}", flush=True)

t0 = time.time()
res = {"note": "NO-DUPLICATE control: neighbours with Jaccard >= 0.5 excluded from transfer AND test proteins that have any such neighbour dropped. Nothing can be memorised. If S still beats C here, the cross-aspect signal is real and graded.",
       "n_train": len(TR), "n_test": len(TE), "vocab": len(vocab)}
rows_c, rows_s, rows_o, yv, sc_c, sc_s = [], [], [], [], [], []
for p in sorted(TE):
    kq = K[p]
    tru = BP[p]
    knn = knn_transfer(kq)
    # candidates: everything either mechanism proposes
    cnd = set(knn)
    for k in kq:
        cnd.update(cooc.get(k, {}))
    cnd &= set(vocab)
    if not cnd:
        continue
    for t in cnd:
        mx = max((cooc[k][t] / max(kfreq[k], 1) for k in kq if t in cooc.get(k, {})), default=0.0)
        sm = sum(cooc[k][t] for k in kq if t in cooc.get(k, {})) / max(len(kq), 1)
        prior = tfreq[t] / n_tr
        c = 0.7 * mx + 0.3 * min(sm / 10.0, 1.0)          # counting arm
        s = knn.get(t, 0.0)                                # set-level arm
        lab = 1 if t in tru else 0
        rows_c.append((p, t, c))
        rows_s.append((p, t, s))
        rows_o.append((p, t, float(lab)))
        yv.append(lab)
        sc_c.append(c)
        sc_s.append(s)

res["candidate_rows"] = len(yv)
res["positive_rate"] = round(float(np.mean(yv)), 4)
res["AUC_counting"] = round(auc(yv, sc_c) or 0, 4)
res["AUC_set_level"] = round(auc(yv, sc_s) or 0, 4)
print(f"rows={len(yv):,} pos_rate={res['positive_rate']}  "
      f"AUC counting={res['AUC_counting']}  AUC set-level={res['AUC_set_level']}  ({time.time()-t0:.0f}s)", flush=True)
json.dump(res, open(W / "cross_aspect_nodup.json", "w"), indent=1)

res["C_counting"] = score_arm("C", rows_c)
print(f"[C] counting (what we ship)  {res['C_counting']}", flush=True)
res["S_set_level"] = score_arm("S", rows_s)
print(f"[S] set-level Jaccard-kNN    {res['S_set_level']}", flush=True)
res["O_oracle_channel_ceiling"] = score_arm("O", rows_o)
print(f"[O] ORACLE channel ceiling   {res['O_oracle_channel_ceiling']}", flush=True)

fC = (res.get("C_counting") or {}).get("f_micro_w")
fS = (res.get("S_set_level") or {}).get("f_micro_w")
fO = (res.get("O_oracle_channel_ceiling") or {}).get("f_micro_w")
if fC and fS:
    res["set_level_minus_counting"] = round(fS - fC, 4)
json.dump(res, open(W / "cross_aspect_nodup.json", "w"), indent=1)
print("\n=== Is the prior-knowledge channel thin by DESIGN or by IMPLEMENTATION? ===", flush=True)
print(f"  C counting (what we ship) = {fC}", flush=True)
print(f"  S set-level (same info)   = {fS}", flush=True)
print(f"  O oracle (channel ceiling)= {fO}", flush=True)
if fC and fS:
    print(f"  -> our implementation leaves {fS-fC:+.4f} on the table", flush=True)
print("  Reminder: f_micro_w decides. AUC ordered the reranker levers the wrong way.", flush=True)
print("DONE", flush=True)
