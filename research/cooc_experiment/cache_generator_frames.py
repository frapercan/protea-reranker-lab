"""Cache the two generator frames and their labels once. Read-only, resumable, per-config.

WHY THIS IS ITS OWN SCRIPT. `generator_learned_codes.py` died twice with no error in its log, both
times during the six-PLM pull, which is five minutes of database reads it repeats on every attempt.
A run that has to redo its expensive setup before it can fail again is a run you cannot debug.

So: pull once, write each config's slice as its own `.npy`, and skip anything already on disk. A
crash now costs the config it was on, not the whole frame. Read-only against the live DB, batched,
no writes, no job dispatch.

WHAT IT WRITES, into `generator_frames/`:
  labels.npz         the v227-experimental, true-path-propagated label matrix (CSR) + the vocab
  accs.json          the 88,212 accessions, in the row order every frame shares
  learned_2048.npy   `d8979601`, the deployed k-WTA champion's codes
  raw_<cfg>.npy      one per PLM: ankh_base 768, esm2_3b 2560, ankh_large 1536, esm2_650m 1280,
                     esmc_600m 1152, prott5 1024. They concatenate to the 8320-d frame the shipped
                     `classifier_6plm_asl` eats.

LEAKAGE: labels come from `annotation_set c905dffa` = v227 = our t0, experimental evidence only,
exactly as `extract_base_plm.py` does. The evaluation window v227-v230 is strictly after it.
"""
import json, collections, time
from pathlib import Path
import numpy as np, yaml, psycopg2
from scipy.sparse import csr_matrix, save_npz

W = Path("/home/frapercan/Thesis2/storage/cooc_experiment")
OUT = W / "generator_frames"
OUT.mkdir(exist_ok=True)
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
V227 = "c905dffa-a5ce-430b-b17b-503e88666adb"
EXP = ('EXP', 'IDA', 'IMP', 'IPI', 'IGI', 'IEP', 'TAS', 'IC', 'HTP', 'HDA', 'HMP', 'HGI', 'HEP')
CFGS = [("learned_2048", "d8979601-ea59-4de1-9c16-21036ed67c36", 2048),
        ("raw_ankh_base", "08234f06-ba76-4d7d-aaec-ae601096b4fa", 768),
        ("raw_esm2_3b", "55e43f1c-1a3b-4b1d-88c0-26b433f5f673", 2560),
        ("raw_ankh_large", "238f79b1-3068-4c6f-9013-5cc52b4f662b", 1536),
        ("raw_esm2_650m", "c2e9dda3-e505-4170-b50d-435a451761ac", 1280),
        ("raw_esmc_600m", "2bf1e753-022f-44b8-a131-9a90acb4024e", 1152),
        ("raw_prott5", "084943c6-fec1-441d-bdc5-63b0268ada1b", 1024)]

par = collections.defaultdict(set); ns = {}; alt = {}
cur_ = None
for line in open(OBO):
    line = line.rstrip("\n")
    if line == "[Term]":
        cur_ = None
    elif line.startswith("id: GO:"):
        cur_ = line[4:]
    elif cur_ and line.startswith("namespace: "):
        ns[cur_] = line[11:]
    elif cur_ and line.startswith("is_a: GO:"):
        par[cur_].add(line[6:].split(" ! ")[0].strip())
    elif cur_ and line.startswith("relationship: part_of GO:"):
        par[cur_].add(line[len("relationship: part_of "):].split(" ! ")[0].strip())
    elif cur_ and line.startswith("alt_id: GO:"):
        alt[line[8:]] = cur_
AC = {}
def anc(t):
    if t in AC:
        return AC[t]
    o, st = set(), [t]
    while st:
        x = st.pop()
        for p in par.get(x, ()):
            if p not in o:
                o.add(p); st.append(p)
    AC[t] = o
    return o

cfg = yaml.safe_load(Path("/home/frapercan/Thesis2/repositories/PROTEA/protea/config/system.yaml").read_text())
url = None
def find(d):
    global url
    if isinstance(d, dict):
        for v in d.values():
            if isinstance(v, str) and v.startswith("postgresql"):
                url = v
            else:
                find(v)
    elif isinstance(d, list):
        for v in d:
            find(v)
find(cfg)
url = url.replace("postgresql+psycopg://", "postgresql://")
t0 = time.time()

if (OUT / "accs.json").exists() and (OUT / "labels.npz").exists():
    accs = json.load(open(OUT / "accs.json"))
    print(f"reusing accs.json + labels.npz: {len(accs):,} proteins", flush=True)
else:
    conn = psycopg2.connect(url); conn.set_session(readonly=True)
    cur = conn.cursor()
    cur.execute("""select protein_accession, go_id from protein_go_annotation pga
                   join go_term gt on gt.id = pga.go_term_id
                   where pga.annotation_set_id = %s and pga.evidence_code = any(%s)""",
                (V227, list(EXP)))
    lab = collections.defaultdict(set)
    n_direct = 0
    for a_, g_ in cur.fetchall():
        n_direct += 1
        g_ = alt.get(g_, g_)
        lab[a_] |= {g_} | anc(g_)
    conn.close()
    accs = sorted(lab)
    vocab = sorted({g for v in lab.values() for g in v if g in ns})
    vpos = {g: i for i, g in enumerate(vocab)}
    rows, cols = [], []
    for i, a_ in enumerate(accs):
        for g in lab[a_]:
            j = vpos.get(g)
            if j is not None:
                rows.append(i); cols.append(j)
    Y = csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(accs), len(vocab)))
    save_npz(OUT / "labels.npz", Y)
    json.dump(accs, open(OUT / "accs.json", "w"))
    json.dump(vocab, open(OUT / "vocab.json", "w"))
    print(f"v227 experimental: {len(accs):,} proteins, {n_direct:,} direct, {Y.nnz:,} propagated; "
          f"vocab {len(vocab):,}  ({time.time()-t0:.0f}s)", flush=True)

for name, cfg_id, dim in CFGS:
    f = OUT / f"{name}.npy"
    if f.exists():
        print(f"  {name:16s} already cached {np.load(f, mmap_mode='r').shape}", flush=True)
        continue
    conn = psycopg2.connect(url); conn.set_session(readonly=True)   # one connection per config:
    cur = conn.cursor()                                             # a dead config cannot poison the rest
    X = np.zeros((len(accs), dim), np.float32)
    pos = {a: i for i, a in enumerate(accs)}
    got = 0
    for s in range(0, len(accs), 2000):
        cur.execute("""select p.accession, e.embedding::text from protein p
                       join sequence_embedding e on e.sequence_id = p.sequence_id
                       where e.embedding_config_id = %s and p.accession = any(%s)""",
                    (cfg_id, accs[s:s + 2000]))
        for acc, emb in cur.fetchall():
            v = np.fromstring(emb.strip("[]"), sep=",", dtype=np.float32)
            if v.shape[0] == dim and acc in pos:
                X[pos[acc]] = v; got += 1
    conn.close()
    np.save(f, X)
    print(f"  {name:16s} {X.shape} coverage {(np.abs(X).sum(1) > 0).mean():.1%} "
          f"({got:,} rows)  ({time.time()-t0:.0f}s)", flush=True)
    del X

print(f"\ncached in {OUT}:", flush=True)
for p in sorted(OUT.iterdir()):
    print(f"  {p.name:20s} {p.stat().st_size/1e6:8.1f} MB", flush=True)
print("DONE", flush=True)
