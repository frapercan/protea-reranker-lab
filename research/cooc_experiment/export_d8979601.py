"""Export the deployed encoder's codes for every protein the joint model needs. Read-only.

WHY THIS EXISTS, AND WHAT IT CORRECTS. `joint_vs_scalar_temporal.py` reported that the joint model
could only train on **18.0%** of the pk-bpo rows because the protein vectors "cover only 8.2% of
train.parquet". That was measured off `storage/layer_ablation/query_d8979601.npy`, a 7,401-protein
array left over from the layer-ablation experiment, and it was **wrong about the cause**.

The vectors are not missing. `d8979601` (`learned-code:hard-neg:08234f06`) is the DEPLOYED encoder
and the live database holds **527,426** of its embeddings, covering **100.0%** of the 51,096 pk-bpo
training proteins. Nothing needs computing. What was missing was a way to get the signals out in a
usable shape, which is precisely what Track B's signal store is for, and this script is the
hand-rolled stand-in for it.

**The lesson is the campaign's, again: I inherited a premise from an artefact instead of asking the
system that produces it.** An hour of a re-embedding plan was written for data that already existed.

WHAT IT DOES. Reads `sequence_embedding` for config `d8979601` for the accessions the frozen
parquets name, and writes one float32 matrix plus an accession sidecar. **Read-only, batched, no
job dispatch, no writes, no stack interaction.** The connection is opened `set_session(readonly=True)`
so a mistake in this file cannot become a mistake in the database.
"""
import json, sys, time
from pathlib import Path
import numpy as np, yaml, psycopg2, pyarrow.parquet as pq

CFG = "d8979601-ea59-4de1-9c16-21036ed67c36"
DS = Path("/home/frapercan/Thesis2/repositories/protea-reranker-lab/datasets/protst-global-train227-test230")
OUT = Path("/home/frapercan/Thesis2/storage/cooc_experiment/d8979601_full")
OUT.mkdir(exist_ok=True)

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

want = set()
for f in ("train.parquet", "eval.parquet"):
    t = pq.read_table(DS / f, columns=["protein_accession"])
    want |= set(np.asarray(t.column("protein_accession").to_pylist()).tolist())
want = sorted(want)
print(f"accessions named by the frozen parquets: {len(want):,}", flush=True)

c = psycopg2.connect(url)
c.set_session(readonly=True)          # the live DB: read-only, by construction
cur = c.cursor()
t0 = time.time()
accs, vecs, dim = [], [], None
B = 4000
for s in range(0, len(want), B):
    chunk = want[s:s + B]
    cur.execute("""select p.accession, se.embedding::text
                   from protein p
                   join sequence_embedding se on se.sequence_id = p.sequence_id
                   where se.embedding_config_id = %s and p.accession = any(%s)""",
                (CFG, chunk))
    for acc, emb in cur.fetchall():
        v = np.fromstring(emb.strip("[]"), sep=",", dtype=np.float32)
        if dim is None:
            dim = v.shape[0]
            print(f"  embedding dim = {dim}", flush=True)
        if v.shape[0] != dim:
            continue
        accs.append(acc); vecs.append(v)
    print(f"  {s+len(chunk):>7,}/{len(want):,} requested | {len(accs):>7,} fetched  ({time.time()-t0:.0f}s)", flush=True)

# one accession can map to several protein rows through a shared sequence; keep the first
seen, keep = set(), []
for i, a in enumerate(accs):
    if a not in seen:
        seen.add(a); keep.append(i)
M = np.stack([vecs[i] for i in keep]).astype(np.float32)
A = [accs[i] for i in keep]
np.save(OUT / "codes.npy", M)
json.dump(A, open(OUT / "accs.json", "w"))
print(f"\nwrote {OUT}/codes.npy {M.shape} and accs.json ({len(A):,} accessions)  ({time.time()-t0:.0f}s)", flush=True)

# the number that motivated this: coverage of the cell the campaign is stuck on
t = pq.read_table(DS / "train.parquet", columns=["protein_accession", "category", "aspect"])
P = np.asarray(t.column("protein_accession").to_pylist())
C = np.asarray(t.column("category").to_pylist()); AS = np.asarray(t.column("aspect").to_pylist())
pk = set(P[(C == "pk") & (AS == "bpo")].tolist())
old = set(json.load(open("/home/frapercan/Thesis2/storage/layer_ablation/prot_go.json")))
print(f"\npk-bpo TRAIN protein coverage:")
print(f"  the leftover .npy the 18% figure came from: {len(pk & old):,}/{len(pk):,} = {len(pk & old)/len(pk):.1%}")
print(f"  this export:                                {len(pk & set(A)):,}/{len(pk):,} = {len(pk & set(A))/len(pk):.1%}")
print("DONE", flush=True)
