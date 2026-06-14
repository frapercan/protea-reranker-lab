"""M0 data extraction: per-protein Ankh-base embeddings + propagated t0 labels.

Training proteins = v227 experimental-annotated; eval = 7401 LAFA targets.
Vocab = TOI terms that appear in training labels. Saves /tmp/m0_data.npz.
"""
import numpy as np
import psycopg
from collections import defaultdict

DB = "postgresql://protea:protea@localhost:5432/protea"
EMB_CFG = "08234f06-ba76-4d7d-aaec-ae601096b4fa"
V227 = "c905dffa-a5ce-430b-b17b-503e88666adb"
EXP = ('EXP', 'IDA', 'IMP', 'IPI', 'IGI', 'IEP', 'TAS', 'IC', 'HTP', 'HDA', 'HMP', 'HGI', 'HEP')
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
TOI = "/home/frapercan/Thesis2/CAFA_forever/data/releases/Sep_2025_Mar_2026/groundtruth_terms_of_interest.txt"
TARGETS = "/home/frapercan/Thesis2/CAFA_forever/data/releases/Sep_2025_Mar_2026/groundtruth_targets.tsv"


def parents_map():
    parents = defaultdict(set); cur = None
    for line in open(OBO):
        line = line.strip()
        if line == "[Term]": cur = None
        elif line.startswith("id: GO:"): cur = line[4:]
        elif line.startswith("is_a:") and cur: parents[cur].add(line.split()[1])
        elif line.startswith("relationship: part_of") and cur:
            pr = line.split()
            if len(pr) >= 3: parents[cur].add(pr[2])
    return parents


def anc(t, parents, cache):
    if t in cache: return cache[t]
    out = set(); st = list(parents.get(t, ()))
    while st:
        a = st.pop()
        if a in out: continue
        out.add(a); st.extend(parents.get(a, ()))
    cache[t] = out; return out


def main():
    toi = set(l.strip() for l in open(TOI) if l.strip().startswith("GO:"))
    targets = set(l.strip() for l in open(TARGETS) if l.strip())
    print(f"TOI={len(toi)} targets={len(targets)}", flush=True)
    parents = parents_map(); cache = {}

    conn = psycopg.connect(DB)
    cur = conn.cursor()

    # training proteins + leaf experimental terms (v227)
    print("fetching training labels...", flush=True)
    cur.execute("""
        select a.protein_accession, g.go_id
        from protein_go_annotation a join go_term g on g.id=a.go_term_id
        where a.annotation_set_id=%s and a.evidence_code = any(%s)
          and coalesce(a.qualifier,'') not like '%%NOT%%' and g.aspect in ('F','P','C')
    """, (V227, list(EXP)))
    train_leaf = defaultdict(set)
    for acc, go in cur:
        train_leaf[acc].add(go)
    print(f"train proteins with exp labels={len(train_leaf)}", flush=True)

    # propagate + restrict to TOI; build vocab
    prop = {}
    vocab_counter = defaultdict(int)
    for acc, leaves in train_leaf.items():
        s = set(leaves)
        for t in leaves:
            s |= anc(t, parents, cache)
        s &= toi
        if s:
            prop[acc] = s
            for t in s:
                vocab_counter[t] += 1
    # vocab = TOI terms seen in >=1 training protein (drop ultra-rare <2 to bound)
    vocab = sorted([t for t, c in vocab_counter.items() if c >= 1])
    tidx = {t: i for i, t in enumerate(vocab)}
    print(f"vocab={len(vocab)} train proteins with prop labels={len(prop)}", flush=True)

    train_accs = sorted(prop.keys())
    # 7401 target accessions
    eval_accs = sorted(targets)

    # fetch embeddings for the union
    need = sorted(set(train_accs) | set(eval_accs))
    print(f"fetching embeddings for {len(need)} proteins...", flush=True)
    emb = {}
    B = 5000
    for i in range(0, len(need), B):
        chunk = need[i:i + B]
        cur.execute("""
            select p.accession, e.embedding::text
            from protein p join sequence s on s.id=p.sequence_id
            join sequence_embedding e on e.sequence_id=s.id
            where e.embedding_config_id=%s and p.accession = any(%s)
        """, (EMB_CFG, chunk))
        for acc, vec in cur:
            if acc not in emb:
                emb[acc] = np.fromstring(vec.strip("[]"), sep=",", dtype=np.float32)
        if i % 25000 == 0:
            print(f"  {i}/{len(need)} ({len(emb)} embedded)", flush=True)
    conn.close()
    print(f"embedded={len(emb)}", flush=True)

    # assemble train arrays (only proteins with embedding)
    tr = [a for a in train_accs if a in emb]
    Xtr = np.stack([emb[a] for a in tr]).astype(np.float32)
    # labels as ragged -> index arrays
    rows, cols = [], []
    for i, a in enumerate(tr):
        for t in prop[a]:
            j = tidx.get(t)
            if j is not None:
                rows.append(i); cols.append(j)
    ev = [a for a in eval_accs if a in emb]
    Xev = np.stack([emb[a] for a in ev]).astype(np.float32)
    print(f"Xtr={Xtr.shape} labels_nnz={len(rows)} Xev={Xev.shape} (eval missing emb={len(eval_accs)-len(ev)})", flush=True)

    np.savez_compressed("/tmp/m0_data.npz",
                        Xtr=Xtr, rows=np.array(rows, np.int32), cols=np.array(cols, np.int32),
                        tr_acc=np.array(tr), Xev=Xev, ev_acc=np.array(ev),
                        vocab=np.array(vocab))
    print("saved /tmp/m0_data.npz", flush=True)


if __name__ == "__main__":
    main()
