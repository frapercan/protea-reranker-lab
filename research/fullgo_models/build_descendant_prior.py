"""NEW PK evidence signal: descendant-of-known-t0 prior (DAG-structural).

descendant_prior[row] = 1.0 if the candidate term is a STRICT GO-DAG descendant of
any of the protein's known non-experimental t0 terms K(p) (i.e. ancestors(candidate)
intersect K(p), and candidate not itself in K(p)). Models annotation REFINEMENT:
PK proteins acquire more-specific (descendant) terms of what they already know.
Complementary to self_prior (candidate IN K(p)) and association (co-occurrence).
Leakage-clean: K(p) empty for NK -> 0; uses only pre-cutoff t0 non-exp annotations.

Reuses the de-risk K(p) data (selfprior_fix_data) + the same row-order alignment.
Outputs {split}_descprior.npz aligned to the v5 parquet row order."""
from __future__ import annotations

import io
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq
from minio import Minio

D = "/home/frapercan/Thesis2/storage/fullgo_models/selfprior_ia_experiment"
FIX = "/home/frapercan/Thesis2/storage/fullgo_models/selfprior_fix_data"
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
BUCKET = "protea"
BASE = "datasets/fullgo-union-SELECT-160-220-227-v5"
CLIENT = Minio("localhost:9000", access_key="minioadmin", secret_key="minioadmin", secure=False)
SPLITS = {"train": f"{BASE}/train.parquet", "eval": f"{BASE}/eval.parquet"}


def free_g():
    with open("/proc/meminfo") as fh:
        for ln in fh:
            if ln.startswith("MemAvailable"):
                return int(ln.split()[1]) / 1024 / 1024
    return 99.0


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] free={free_g():.1f}G  {m}", flush=True)


def parents_map():
    par = defaultdict(set)
    cur = None
    for line in open(OBO):
        line = line.strip()
        if line == "[Term]":
            cur = None
        elif line.startswith("id: GO:"):
            cur = line[4:]
        elif line.startswith("is_a:") and cur:
            par[cur].add(line.split()[1])
        elif line.startswith("relationship: part_of") and cur:
            p = line.split()
            if len(p) >= 3:
                par[cur].add(p[2])
    return par


_ANC_CACHE = {}


def anc(t, par):
    if t in _ANC_CACHE:
        return _ANC_CACHE[t]
    out = set()
    st = list(par.get(t, ()))
    while st:
        a = st.pop()
        if a in out:
            continue
        out.add(a)
        st.extend(par.get(a, ()))
    _ANC_CACHE[t] = out
    return out


def load_sets():
    ver2set = {}
    for ln in open(f"{FIX}/sets.csv"):
        ln = ln.strip()
        if ln:
            sid, ver = ln.split("|")
            ver2set[ver.strip()] = sid.strip()
    return ver2set


def load_goterm_map():
    m = {}
    for ln in open(f"{FIX}/goterm_map.tsv"):
        i = ln.find("\t")
        if i >= 0:
            m[ln[:i]] = ln[i + 1:].rstrip("\n")
    return m


def collect_pairs(ver2set):
    pairs = set()
    cache = {}
    for split, key in SPLITS.items():
        log(f"pass1 {split}")
        raw = CLIENT.get_object(BUCKET, key).read()
        cache[split] = raw
        pf = pq.ParquetFile(io.BytesIO(raw))
        for b in pf.iter_batches(batch_size=2_000_000, columns=["protein_accession", "snapshot_pair"]):
            for p, s in zip(b.column("protein_accession").to_pylist(), b.column("snapshot_pair").to_pylist()):
                sid = ver2set.get(s.split("-")[0].lstrip("v"))
                if sid is not None:
                    pairs.add((sid, p))
    log(f"pairs={len(pairs)}")
    return pairs, cache


def build_kp(pairs, gtmap):
    keep = {}
    n = kept = 0
    with open(f"{FIX}/nonexp_annotations.tsv") as fh:
        for ln in fh:
            n += 1
            i = ln.find("\t")
            j = ln.find("\t", i + 1)
            key = (ln[:i], ln[i + 1:j])
            if key not in pairs:
                continue
            goid = gtmap.get(ln[j + 1:].rstrip("\n"))
            if goid is None:
                continue
            keep.setdefault(key, set()).add(goid)
            kept += 1
            if n % 10_000_000 == 0:
                log(f"pass2 {n//1_000_000}M kept={kept} pairs={len(keep)}")
                if free_g() < 6.0:
                    log("ABORT pass2 RAM<6G")
                    sys.exit(2)
    log(f"pass2 done: kept {kept} across {len(keep)} pairs")
    return keep


def build_overlays(cache, ver2set, kp, par):
    sanity = {}
    for split, raw in cache.items():
        log(f"pass3 {split}")
        pf = pq.ParquetFile(io.BytesIO(raw))
        parts = []
        per_cat = {"nk": 0, "lk": 0, "pk": 0}
        tot = {"nk": 0, "lk": 0, "pk": 0}
        for b in pf.iter_batches(batch_size=2_000_000,
                                 columns=["protein_accession", "go_term_id", "snapshot_pair", "category"]):
            prot = b.column("protein_accession").to_pylist()
            goid = b.column("go_term_id").to_pylist()
            sp = b.column("snapshot_pair").to_pylist()
            cat = b.column("category").to_pylist()
            n = len(prot)
            dp = np.zeros(n, dtype=np.float32)
            for k in range(n):
                tot[cat[k]] += 1
                sid = ver2set.get(sp[k].split("-")[0].lstrip("v"))
                if sid is None:
                    continue
                known = kp.get((sid, prot[k]))
                if not known:
                    continue
                g = goid[k]
                if g in known:
                    continue  # that is self_prior, not descendant
                if anc(g, par) & known:
                    dp[k] = 1.0
                    per_cat[cat[k]] += 1
            parts.append(dp)
        arr = np.concatenate(parts)
        np.savez(f"{D}/{split}_descprior.npz", descendant_prior=arr)
        sanity[split] = {"rows": int(arr.size), "descprior_nonzero": int((arr > 0).sum()),
                         "per_cat_nonzero": per_cat, "per_cat_total": tot}
        log(f"pass3 {split}: descprior_nz={int((arr>0).sum())} per_cat={per_cat}")
    return sanity


def main():
    ver2set = load_sets()
    par = parents_map()
    log(f"OBO parents for {len(par)} terms")
    pairs, cache = collect_pairs(ver2set)
    gtmap = load_goterm_map()
    log(f"goterm_map {len(gtmap)}")
    kp = build_kp(pairs, gtmap)
    del gtmap
    sanity = build_overlays(cache, ver2set, kp, par)
    json.dump(sanity, open(f"{D}/descprior_sanity.json", "w"), indent=2)
    log("DONE build_descendant_prior")
    print(json.dumps(sanity, indent=2), flush=True)


if __name__ == "__main__":
    main()
