"""Append the 5 non-base PLMs to /tmp/m0_data.npz -> /tmp/m0v3_data.npz (8320-d).

Order (canonical PLM_CONCAT_ORDER after ankh_base): esm2_3b, ankh_large,
esm2_650m, esmc_600m, prott5. Reads the Ankh-base frame produced by
extract_base_plm.py and hstacks the rest, aligned to tr_acc / ev_acc.
"""
import numpy as np, psycopg
DB = "postgresql://protea:protea@localhost:5432/protea"
ORDER = [("55e43f1c-1a3b-4b1d-88c0-26b433f5f673", 2560),   # esm2_3b
         ("238f79b1-3068-4c6f-9013-5cc52b4f662b", 1536),   # ankh_large
         ("c2e9dda3-e505-4170-b50d-435a451761ac", 1280),   # esm2_650m
         ("2bf1e753-022f-44b8-a131-9a90acb4024e", 1152),   # esmc_600m
         ("084943c6-fec1-441d-bdc5-63b0268ada1b", 1024)]   # prott5
d = np.load("/tmp/m0_data.npz", allow_pickle=True)
tr_acc = [str(x) for x in d["tr_acc"]]; ev_acc = [str(x) for x in d["ev_acc"]]
need = sorted(set(tr_acc) | set(ev_acc))
conn = psycopg.connect(DB); cur = conn.cursor()
tr_parts = [d["Xtr"].astype(np.float32)]; ev_parts = [d["Xev"].astype(np.float32)]
for cfg, dim in ORDER:
    emb = {}; B = 5000
    for i in range(0, len(need), B):
        ch = need[i:i+B]
        cur.execute("select p.accession,e.embedding::text from protein p join sequence s on s.id=p.sequence_id "
                    "join sequence_embedding e on e.sequence_id=s.id where e.embedding_config_id=%s and p.accession=any(%s)",
                    (cfg, ch))
        for a, v in cur:
            if a not in emb:
                emb[a] = np.fromstring(v.strip("[]"), sep=",", dtype=np.float32)
    miss_tr = sum(1 for a in tr_acc if a not in emb); miss_ev = sum(1 for a in ev_acc if a not in emb)
    print(f"{cfg[:8]} dim {dim} n {len(emb)} miss_tr {miss_tr} miss_ev {miss_ev}", flush=True)
    Mtr = np.zeros((len(tr_acc), dim), np.float32)
    for i, a in enumerate(tr_acc):
        v = emb.get(a)
        if v is not None and len(v) == dim:
            Mtr[i] = v
    Mev = np.zeros((len(ev_acc), dim), np.float32)
    for i, a in enumerate(ev_acc):
        v = emb.get(a)
        if v is not None and len(v) == dim:
            Mev[i] = v
    tr_parts.append(Mtr); ev_parts.append(Mev); del emb
conn.close()
Xtr = np.hstack(tr_parts).astype(np.float32); Xev = np.hstack(ev_parts).astype(np.float32)
print("Xtr", Xtr.shape, "Xev", Xev.shape, flush=True)
OUT = "/home/frapercan/Thesis2/storage/struct_gate/m2frame.npz"
np.savez(OUT, Xtr=Xtr, rows=d["rows"], cols=d["cols"],
         tr_acc=d["tr_acc"], Xev=Xev, ev_acc=d["ev_acc"], vocab=d["vocab"])
print(f"saved {OUT}", flush=True)
