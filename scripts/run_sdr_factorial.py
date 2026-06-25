#!/usr/bin/env python
"""Factorial {substrate x aggregation} SDR correlation proxy, length-stratified (T-CIENCIA).

Two things the prior length-stratified proxy did NOT isolate:

  1. **Chunking: yes or no?**  mean vs chunk vs residue substrate, under the SAME aggregation.
  2. **How to aggregate?**     naive sparsify-then-bundle vs LEARNED attention-pool, same substrate.

This runs the full 3x2 factorial plus a dense-mean-cosine reference, on ankh-base, bucketed by
protein length, reporting Spearman(arm_similarity, GO_semantic_similarity) for Resnik AND Lin per
bucket. The decisive, never-tested cells are chunk/residue + LEARNED (theory says they win where
locality matters).

  substrate  x  aggregation
  ---------     -----------
  mean          naive  = k-WTA on the mean -> Tanimoto
  mean          learned= champion learned hard-neg k-WTA code (d8979601) -> Tanimoto  [NO retrain]
  chunk         naive  = per-chunk k-WTA -> vote bundle -> Tanimoto
  chunk         learned= attention-pool over chunks -> top-k real code -> cosine       [TRAINED]
  residue       naive  = per-residue top-k -> OR bundle -> Tanimoto
  residue       learned= attention-pool over residues -> top-k real code -> cosine     [TRAINED]
  (reference)   dense-mean-cosine

TRUNCATION CONTROL (critical): the ankh-base substrates have DIFFERENT coverage (mean config
08234f06 truncates >2048, chunk config 6542db1e at 4096, residue npy sample tops near 1959).
Comparing substrates on long proteins is therefore confounded by COVERAGE, not aggregation. To
isolate substrate + aggregation cleanly we RESTRICT the whole factorial to length <= 1959, so all
three substrates see the (near) full protein, and bucket short <=318 / medium 319-969 / long
970-1959. The very-long (>2048) giants (~1% of proteins) are DROPPED: they are an untested regime
that needs full-length re-extraction; no claim here generalises to them.

Naive arms reuse Tanimoto over k-WTA bitsets (sparse set-overlap). Learned arms use cosine over the
sparse real code (the champion's readout). dense-mean-cosine uses cosine over the raw mean. Within a
(bucket, metric) all arms are scored on the SAME protein pairs, so the deltas are clean.

Leakage-clean (frozen v227 t0 pool + the t0 OBO), READ-ONLY on the DB. Logs to MLflow experiment
``sdr-factorial-substrate-agg``.

Run with the GPU lab/PROTEA venv and MLflow live::

    export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
    python scripts/run_sdr_factorial.py
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from protea_reranker_lab.sdr import (
    GoDag,
    cosine_dense,
    information_content,
    kwta_binarise,
    lin_pairwise,
    propagate,
    resnik_pairwise,
    tanimoto_dense,
)
from protea_reranker_lab.sdr_pool import (
    PoolSpec,
    apply_attention_pool,
    fit_attention_pool,
    l2n,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [sdr-fact] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sdr-fact")

EXPERIMENT = "sdr-factorial-substrate-agg"

EMB_MEAN = "08234f06-ba76-4d7d-aaec-ae601096b4fa"      # ankh-base mean, 768d
EMB_CHUNK = "6542db1e-202a-4769-b933-2e0f85aa81e6"     # ankh-base per-chunk, 768d/chunk
EMB_LEARNED = "d8979601-ea59-4de1-9c16-21036ed67c36"   # champion learned hard-neg code, 2048d
ANN_SET = "c905dffa-a5ce-430b-b17b-503e88666adb"       # GOA v227, t0
RESDIR = "/home/frapercan/Thesis2/storage/fullgo_models/per_residue_v227"

# truncation-clean buckets: very-long (>1959) DROPPED (coverage-confounded, untested regime)
MAX_LEN = 1959
BUCKETS = [
    ("short", 0, 318),
    ("medium", 319, 969),
    ("long", 970, 1959),
]
ARM_ORDER = [
    "dense-mean-cosine",
    "mean|naive", "mean|learned",
    "chunk|naive", "chunk|learned",
    "residue|naive", "residue|learned",
]


def bucket_of(length: int) -> str | None:
    for name, lo, hi in BUCKETS:
        if lo <= length <= hi:
            return name
    return None  # >MAX_LEN -> excluded


def _parse_vec(text: str) -> np.ndarray:
    return np.fromstring(text.strip()[1:-1], sep=",", dtype=np.float32)


# ---------------------------------------------------------------------------
# Unified DB sample (mean / chunk / learned), restricted to L <= MAX_LEN
# ---------------------------------------------------------------------------
def load_db_sample(dsn: str, seed: int, per_bucket_cap: int, residue_accs: set[str] | None = None):
    """Load the mean/chunk/learned sample for L<=MAX_LEN, length-bucketed and capped.

    ``residue_accs`` (accessions that have a per-residue array on disk) are PRIORITISED inside
    each bucket so the residue arms get a usable N; the rest of each bucket is filled from the
    remaining pool. This keeps the substrate arms scored on a shared, residue-rich sample.
    """
    import psycopg2

    residue_accs = residue_accs or set()
    rng = np.random.default_rng(seed)
    conn = psycopg2.connect(dsn)
    conn.set_session(readonly=True)
    cur = conn.cursor()

    log.info("scanning lengths for v227-annotated proteins with length <= %d", MAX_LEN)
    cur.execute(
        """
        SELECT p.accession, p.sequence_id, length(s.sequence) AS seqlen
        FROM (SELECT DISTINCT protein_accession AS acc
              FROM protein_go_annotation WHERE annotation_set_id = %(ann)s) a
        JOIN protein p ON p.accession = a.acc
        JOIN sequence s ON s.id = p.sequence_id
        WHERE p.sequence_id IS NOT NULL AND length(s.sequence) <= %(maxlen)s
        """,
        {"ann": ANN_SET, "maxlen": MAX_LEN},
    )
    cand = cur.fetchall()
    log.info("  %d annotated proteins (<=%d) with a sequence", len(cand), MAX_LEN)

    acc_to_seq = {acc: sid for acc, sid, _ in cand}
    acc_len = {acc: int(ln) for acc, _, ln in cand}

    by_bucket: dict[str, list[str]] = {name: [] for name, _, _ in BUCKETS}
    for acc, _, ln in cand:
        b = bucket_of(int(ln))
        if b is not None:
            by_bucket[b].append(acc)
    sample: list[str] = []
    for name in by_bucket:
        accs_b = by_bucket[name]
        # prioritise residue-available accessions, then fill from the rest (both shuffled)
        with_res = [a for a in accs_b if a in residue_accs]
        without_res = [a for a in accs_b if a not in residue_accs]
        rng.shuffle(with_res)
        rng.shuffle(without_res)
        ordered = with_res + without_res
        take = ordered[:per_bucket_cap]
        n_res_taken = sum(1 for a in take if a in residue_accs)
        log.info("  bucket %-7s available=%d (residue=%d) -> taking %d (residue=%d)",
                 name, len(accs_b), len(with_res), len(take), n_res_taken)
        sample.extend(take)
    rng.shuffle(sample)
    log.info("  truncation-clean sample size: %d", len(sample))
    seq_ids = tuple(acc_to_seq[a] for a in sample)
    seq_to_acc = {acc_to_seq[a]: a for a in sample}
    sample_set = tuple(sample)

    def _pull_dense(cfg: str) -> dict[str, np.ndarray]:
        cur.execute(
            """SELECT sequence_id, embedding::text FROM sequence_embedding
               WHERE embedding_config_id = %s AND sequence_id IN %s""",
            (cfg, seq_ids),
        )
        out: dict[str, np.ndarray] = {}
        for sid, vtext in cur:
            a = seq_to_acc.get(sid)
            if a is not None:
                out[a] = _parse_vec(vtext)
        return out

    log.info("pulling mean vectors (%s)", EMB_MEAN)
    mean = _pull_dense(EMB_MEAN)
    log.info("pulling champion learned codes (%s)", EMB_LEARNED)
    learned = _pull_dense(EMB_LEARNED)

    log.info("pulling per-chunk vectors (%s)", EMB_CHUNK)
    cur.execute(
        """SELECT sequence_id, chunk_index_s, embedding::text FROM sequence_embedding
           WHERE embedding_config_id = %s AND sequence_id IN %s
           ORDER BY sequence_id, chunk_index_s""",
        (EMB_CHUNK, seq_ids),
    )
    chunk_tmp: dict[str, list[tuple[int, np.ndarray]]] = {}
    for sid, ci, vtext in cur:
        a = seq_to_acc.get(sid)
        if a is not None:
            chunk_tmp.setdefault(a, []).append((ci, _parse_vec(vtext)))
    chunks: dict[str, np.ndarray] = {}
    for a, lst in chunk_tmp.items():
        lst.sort(key=lambda t: t[0])
        chunks[a] = np.vstack([v for _, v in lst])

    log.info("pulling v227 leaf annotations")
    cur.execute(
        """SELECT pga.protein_accession, gt.go_id FROM protein_go_annotation pga
           JOIN go_term gt ON gt.id = pga.go_term_id
           WHERE pga.annotation_set_id = %s AND pga.protein_accession IN %s""",
        (ANN_SET, sample_set),
    )
    leaves: dict[str, list[str]] = {}
    for a, go in cur:
        leaves.setdefault(a, []).append(go)

    cur.close()
    conn.close()

    accs = [a for a in sample if a in mean and a in learned and a in chunks and a in leaves]
    log.info("  %d accessions with mean+chunk+learned+annotations", len(accs))
    return accs, acc_len, mean, chunks, learned, leaves


# ---------------------------------------------------------------------------
# Naive bundle representations
# ---------------------------------------------------------------------------
def chunk_naive_bundle(chunks: np.ndarray, k: int, d: int) -> np.ndarray:
    """Per-chunk k-WTA -> length-normalised top-k vote bundle (one uint8 SDR row)."""
    from protea_reranker_lab.sdr import kwta_active_set

    if chunks.shape[0] == 1:
        top = kwta_active_set(chunks[0], k)
    else:
        votes = np.zeros(d, dtype=np.int32)
        mass = np.zeros(d, dtype=np.float32)
        for chunk in chunks:
            act = kwta_active_set(chunk, k)
            votes[act] += 1
            mass[act] += np.abs(chunk[act])
        order = np.lexsort((mass, votes))[::-1]
        top = order[:k]
    out = np.zeros(d, dtype=np.uint8)
    out[top] = 1
    return out


def residue_naive_bundle(M: np.ndarray, k: int, d: int) -> np.ndarray:
    """Per-residue top-k by |magnitude| -> OR-bundle across residues (one uint8 SDR row)."""
    kk = min(k, d - 1)
    idx = np.argpartition(-np.abs(M), kk, axis=1)[:, :kk]
    bits = np.zeros(M.shape, dtype=np.uint8)
    np.put_along_axis(bits, idx, 1, axis=1)
    return bits.any(0).astype(np.uint8)


def learned_active_bitset(codes: np.ndarray) -> tuple[np.ndarray, int]:
    bits = (codes != 0.0).astype(np.uint8)
    k_active = int(np.median(bits.sum(axis=1))) or 1
    return bits, k_active


# ---------------------------------------------------------------------------
# Pair sampling + GO targets
# ---------------------------------------------------------------------------
def sample_pairs(n: int, n_pairs: int, rng: np.random.Generator):
    n_pairs = min(n_pairs, n * (n - 1) // 2)
    seen: set[tuple[int, int]] = set()
    pairs: list[tuple[int, int]] = []
    while len(pairs) < n_pairs:
        i, j = int(rng.integers(n)), int(rng.integers(n))
        if i == j:
            continue
        key = (i, j) if i < j else (j, i)
        if key in seen:
            continue
        seen.add(key)
        pairs.append(key)
    return pairs


def go_targets(closures, ic, best_ic, pairs):
    return {
        "resnik": np.asarray(resnik_pairwise(closures, ic, pairs), dtype=np.float64),
        "lin": np.asarray(lin_pairwise(closures, ic, pairs, best_ic), dtype=np.float64),
    }


def _rho(sim, go_vec) -> float:
    return float(spearmanr(sim, go_vec)[0])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(args: argparse.Namespace) -> dict:
    rng = np.random.default_rng(args.seed)
    dag = GoDag.from_obo(Path(args.obo).expanduser())

    res_on_disk = set(f[:-4] for f in os.listdir(RESDIR) if f.endswith(".npy"))
    accs, acc_len, mean, chunks, learned, leaves = load_db_sample(
        args.dsn, args.seed, args.per_bucket_cap, residue_accs=res_on_disk
    )
    closures_all = [propagate(leaves[a], dag) for a in accs]
    keep = [i for i, c in enumerate(closures_all) if c]
    accs = [accs[i] for i in keep]
    closures_all = [closures_all[i] for i in keep]
    log.info("%d proteins with a non-empty GO closure", len(accs))

    lengths = np.array([acc_len[a] for a in accs])
    d_mean = mean[accs[0]].shape[0]

    # ---- residue substrate availability (own npy sample, restricted to <=MAX_LEN) ----
    has_res = {a: (a in res_on_disk) for a in accs}
    n_res = sum(has_res.values())
    log.info("residue arrays available for %d / %d sample proteins", n_res, len(accs))

    # ---- champion learned-mean bitset (no retrain) ----
    learned_bits_all, k_learned = learned_active_bitset(np.vstack([learned[a] for a in accs]))

    # ---- TRAIN learned attention pools (chunk + residue) on the WHOLE clean sample ----
    pool_spec = PoolSpec(
        dict_dim=args.dict_dim, top_k=args.pool_top_k, attn_dim=args.attn_dim,
        epochs=args.pool_epochs, train_pairs=args.pool_train_pairs, seed=args.seed,
    )
    mean_matrix = np.vstack([mean[a] for a in accs]).astype(np.float32)

    chunk_pool_code = None
    if not args.skip_chunk_learned:
        log.info("training CHUNK attention-pool (n=%d, epochs=%d)", len(accs), pool_spec.epochs)
        chunk_units = [l2n(chunks[a]) for a in accs]
        enc_c = fit_attention_pool(chunk_units, closures_all, dag, pool_spec, mean_matrix)
        chunk_pool_code = apply_attention_pool(enc_c, chunk_units, pool_spec.top_k)
        del enc_c

    residue_pool_code = None
    res_idx = [i for i, a in enumerate(accs) if has_res[a]]
    if not args.skip_residue_learned and len(res_idx) >= args.min_bucket_n:
        log.info("training RESIDUE attention-pool (n=%d with residue arrays)", len(res_idx))
        res_units = [l2n(np.load(os.path.join(RESDIR, f"{accs[i]}.npy")).astype(np.float32))
                     for i in res_idx]
        res_clo = [closures_all[i] for i in res_idx]
        res_mean = mean_matrix[res_idx]
        enc_r = fit_attention_pool(res_units, res_clo, dag, pool_spec, res_mean)
        codes = apply_attention_pool(enc_r, res_units, pool_spec.top_k)
        residue_pool_code = {accs[res_idx[j]]: codes[j] for j in range(len(res_idx))}
        del enc_r

    # group indices by bucket
    buckets: dict[str, list[int]] = {name: [] for name, _, _ in BUCKETS}
    for i, ln in enumerate(lengths):
        b = bucket_of(int(ln))
        if b is not None:
            buckets[b].append(i)

    skipped: list[str] = []
    table: dict[str, dict] = {}
    for bname, idxs in buckets.items():
        n = len(idxs)
        entry: dict[str, object] = {"N": n}
        if n < args.min_bucket_n:
            log.warning("bucket %-7s N=%d below min (%d) -> skipped", bname, n, args.min_bucket_n)
            skipped.append(f"bucket:{bname}(N={n})")
            table[bname] = entry
            continue

        sub_acc = [accs[i] for i in idxs]
        sub_clo = [closures_all[i] for i in idxs]
        ic = information_content(sub_clo, dag)
        best_ic = [max((ic.get(t, 0.0) for t in c), default=0.0) for c in sub_clo]
        pairs = sample_pairs(n, args.n_pairs, rng)
        ii = np.array([p[0] for p in pairs])
        jj = np.array([p[1] for p in pairs])
        go = go_targets(sub_clo, ic, best_ic, pairs)
        log.info("bucket %-7s N=%d pairs=%d", bname, n, len(pairs))

        dense = np.vstack([mean[a] for a in sub_acc]).astype(np.float32)

        # dense-mean-cosine (reference)
        cos = cosine_dense(dense)[ii, jj]
        for m in ("resnik", "lin"):
            entry[f"dense-mean-cosine|{m}"] = _rho(cos, go[m])

        # mean|naive (k-WTA Tanimoto, swept k -> keep best per metric, report all)
        for k in args.kwta_k:
            msdr = kwta_binarise(dense, k)
            mtan = tanimoto_dense(msdr, k)[ii, jj]
            for m in ("resnik", "lin"):
                entry[f"mean|naive-k{k}|{m}"] = _rho(mtan, go[m])
            csdr = np.vstack([chunk_naive_bundle(chunks[a], k, d_mean) for a in sub_acc])
            ctan = tanimoto_dense(csdr, k)[ii, jj]
            for m in ("resnik", "lin"):
                entry[f"chunk|naive-k{k}|{m}"] = _rho(ctan, go[m])

        # mean|learned (champion code, fixed)
        lbits = learned_bits_all[idxs]
        ltan = tanimoto_dense(lbits, k_learned)[ii, jj]
        for m in ("resnik", "lin"):
            entry[f"mean|learned|{m}"] = _rho(ltan, go[m])

        # chunk|learned (attention pool, cosine over sparse real code)
        if chunk_pool_code is not None:
            cc = chunk_pool_code[idxs]
            ccos = cosine_dense(cc)[ii, jj]
            for m in ("resnik", "lin"):
                entry[f"chunk|learned|{m}"] = _rho(ccos, go[m])

        # residue arms (only proteins with a residue array): own sub-pairs within this bucket
        ridx_local = [li for li, i in enumerate(idxs) if has_res[accs[i]]]
        n_res_b = len(ridx_local)
        entry["N_residue"] = n_res_b
        if n_res_b >= args.min_bucket_n:
            r_sub_acc = [sub_acc[li] for li in ridx_local]
            r_clo = [sub_clo[li] for li in ridx_local]
            r_ic = information_content(r_clo, dag)
            r_bic = [max((r_ic.get(t, 0.0) for t in c), default=0.0) for c in r_clo]
            r_pairs = sample_pairs(n_res_b, args.n_pairs, rng)
            ri = np.array([p[0] for p in r_pairs])
            rj = np.array([p[1] for p in r_pairs])
            r_go = go_targets(r_clo, r_ic, r_bic, r_pairs)

            for k in args.kwta_k:
                rsdr = np.vstack([
                    residue_naive_bundle(
                        np.load(os.path.join(RESDIR, f"{a}.npy")).astype(np.float32), k, d_mean)
                    for a in r_sub_acc])
                rtan = tanimoto_dense(rsdr, int(rsdr.sum(1).mean()) or k)[ri, rj]
                for m in ("resnik", "lin"):
                    entry[f"residue|naive-k{k}|{m}"] = _rho(rtan, r_go[m])

            if residue_pool_code is not None:
                rc = np.vstack([residue_pool_code[a] for a in r_sub_acc])
                rcos = cosine_dense(rc)[ri, rj]
                for m in ("resnik", "lin"):
                    entry[f"residue|learned|{m}"] = _rho(rcos, r_go[m])
        else:
            log.warning("bucket %-7s residue N=%d below min -> residue arms skipped",
                        bname, n_res_b)
            skipped.append(f"residue:{bname}(N={n_res_b})")

        table[bname] = entry

    if args.skip_chunk_learned:
        skipped.append("chunk|learned(--skip-chunk-learned)")
    if args.skip_residue_learned:
        skipped.append("residue|learned(--skip-residue-learned)")

    return {
        "n_proteins": len(accs),
        "n_residue_proteins": n_res,
        "table": table,
        "kwta_k": list(args.kwta_k),
        "buckets": [b[0] for b in BUCKETS],
        "k_learned": k_learned,
        "pool_top_k": args.pool_top_k,
        "max_len": MAX_LEN,
        "skipped": skipped,
        "seed": args.seed,
    }


# ---------------------------------------------------------------------------
# Reporting (collapse k-sweep to best-k per arm|metric for the headline table)
# ---------------------------------------------------------------------------
def _best_over_k(entry: dict, prefix: str, metric: str) -> float | None:
    vals = [v for kk, v in entry.items()
            if isinstance(v, float) and kk.startswith(prefix + "-k") and kk.endswith("|" + metric)]
    if vals:
        return max(vals)
    direct = entry.get(f"{prefix}|{metric}")
    return direct if isinstance(direct, float) else None


def headline_cell(entry: dict, arm: str, metric: str) -> float | None:
    if arm == "dense-mean-cosine":
        v = entry.get(f"dense-mean-cosine|{metric}")
        return v if isinstance(v, float) else None
    sub, agg = arm.split("|")
    if agg == "naive":
        return _best_over_k(entry, f"{sub}|naive", metric)
    return entry.get(f"{sub}|learned|{metric}") if isinstance(
        entry.get(f"{sub}|learned|{metric}"), float) else None


def print_report(result: dict) -> None:
    buckets = result["buckets"]
    table = result["table"]
    print("\n" + "=" * 96)
    print(f"  SDR FACTORIAL  substrate x aggregation  (ankh-base, L<={result['max_len']}, "
          f"naive k best-of {result['kwta_k']})")
    print("  Spearman(arm_similarity, GO_semantic_similarity), per length bucket")
    print("=" * 96)

    for metric in ("resnik", "lin"):
        print(f"\n--- {metric.upper()} ---")
        hdr = f"{'arm':<22}" + "".join(
            f"{b+' (N)':>18}" for b in buckets)
        print(hdr)
        for arm in ARM_ORDER:
            cells = []
            for b in buckets:
                entry = table.get(b, {})
                v = headline_cell(entry, arm, metric)
                if "residue" in arm:
                    nb = entry.get("N_residue", 0)
                else:
                    nb = entry.get("N", 0)
                cells.append(f"{v:.4f} ({nb})" if isinstance(v, float) else f"{'-':>10} ({nb})")
            mark = "  <== NEW" if arm in ("chunk|learned", "residue|learned") else ""
            print(f"{arm:<22}" + "".join(f"{c:>18}" for c in cells) + mark)

    # isolation reads
    print("\n" + "-" * 96)
    print("  ISOLATION 1 - CHUNKING EFFECT (substrate at matched aggregation), delta vs mean")
    print("-" * 96)
    for metric in ("resnik", "lin"):
        for agg in ("naive", "learned"):
            row = f"  [{metric}/{agg:<7}] "
            for b in buckets:
                e = table.get(b, {})
                base = headline_cell(e, f"mean|{agg}", metric)
                parts = []
                for sub in ("chunk", "residue"):
                    v = headline_cell(e, f"{sub}|{agg}", metric)
                    if isinstance(v, float) and isinstance(base, float):
                        parts.append(f"{sub} {v - base:+.4f}")
                    else:
                        parts.append(f"{sub} n/a")
                row += f"{b}: " + ", ".join(parts) + "  |  "
            print(row)

    print("\n" + "-" * 96)
    print("  ISOLATION 2 - AGGREGATION EFFECT (learned - naive at matched substrate)")
    print("-" * 96)
    for metric in ("resnik", "lin"):
        for sub in ("mean", "chunk", "residue"):
            row = f"  [{metric}/{sub:<7}] "
            for b in buckets:
                e = table.get(b, {})
                nv = headline_cell(e, f"{sub}|naive", metric)
                lv = headline_cell(e, f"{sub}|learned", metric)
                if isinstance(nv, float) and isinstance(lv, float):
                    row += f"{b}: {lv - nv:+.4f}  |  "
                else:
                    row += f"{b}: n/a  |  "
            print(row)

    print("\n" + "-" * 96)
    print("  CHAMPION COMPARISON (best substrate+learned - mean|learned champion)")
    print("-" * 96)
    for metric in ("resnik", "lin"):
        row = f"  [{metric}] "
        for b in buckets:
            e = table.get(b, {})
            champ = headline_cell(e, "mean|learned", metric)
            best_name, best_v = None, None
            for sub in ("chunk", "residue"):
                v = headline_cell(e, f"{sub}|learned", metric)
                if isinstance(v, float) and (best_v is None or v > best_v):
                    best_v, best_name = v, sub
            if isinstance(champ, float) and best_v is not None:
                row += f"{b}: {best_name}|learned {best_v - champ:+.4f}  |  "
            else:
                row += f"{b}: n/a  |  "
        print(row)

    if result["skipped"]:
        print("\n  SKIPPED / low-N caveats: " + "; ".join(result["skipped"]))
    print(f"\n  NOTE: results hold only for sequences <= {result['max_len']}; the >2048 giants "
          "(~1% of proteins) are an untested regime requiring full-length re-extraction.")


# ---------------------------------------------------------------------------
# Artifacts + MLflow
# ---------------------------------------------------------------------------
def write_artifacts(result: dict, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    js = out_dir / "sdr_factorial.json"
    js.write_text(json.dumps(result, indent=2))

    rows = []
    for b in result["buckets"]:
        entry = result["table"].get(b, {})
        for metric in ("resnik", "lin"):
            for arm in ARM_ORDER:
                v = headline_cell(entry, arm, metric)
                if isinstance(v, float):
                    nb = entry.get("N_residue" if "residue" in arm else "N", 0)
                    rows.append((b, metric, arm, v, nb))
    csv = out_dir / "sdr_factorial_headline.csv"
    lines = ["bucket,metric,arm,spearman,N"]
    lines += [f"{b},{m},{a},{v:.6f},{n}" for b, m, a, v, n in rows]
    csv.write_text("\n".join(lines) + "\n")
    return [js, csv]


def log_to_mlflow(args, result, artifacts) -> str | None:
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        log.warning("MLFLOW_TRACKING_URI not set; skipping MLflow logging")
        return None
    try:
        os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
        os.environ.setdefault("AWS_ACCESS_KEY_ID", "minioadmin")
        os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minioadmin")
        import mlflow

        mlflow.set_experiment(EXPERIMENT)
        with mlflow.start_run(run_name="sdr-factorial-substrate-agg") as active:
            mlflow.log_params({
                "plm": "ankh-base", "emb_mean": EMB_MEAN, "emb_chunk": EMB_CHUNK,
                "emb_learned_champion": EMB_LEARNED, "annotation_set": ANN_SET,
                "window": "v227-t0", "max_len": MAX_LEN,
                "n_proteins": result["n_proteins"], "n_residue_proteins": result["n_residue_proteins"],
                "n_pairs_per_bucket": args.n_pairs, "kwta_k": ",".join(map(str, args.kwta_k)),
                "k_learned": result["k_learned"], "pool_top_k": result["pool_top_k"],
                "pool_epochs": args.pool_epochs, "buckets": ",".join(result["buckets"]),
                "seed": args.seed,
            })
            for b, entry in result["table"].items():
                for kk, vv in entry.items():
                    if isinstance(vv, (int, float)) and not isinstance(vv, bool):
                        key = f"{b}__{kk}".replace("|", "__").replace("-", "_").replace(".", "_")
                        mlflow.log_metric(key, float(vv))
            # headline cells (best-k collapsed) for easy reading in the UI
            for b in result["buckets"]:
                entry = result["table"].get(b, {})
                for metric in ("resnik", "lin"):
                    for arm in ARM_ORDER:
                        v = headline_cell(entry, arm, metric)
                        if isinstance(v, float):
                            key = f"HEADLINE__{b}__{arm}__{metric}".replace("|", "_").replace("-", "_")
                            mlflow.log_metric(key, v)
            if result["skipped"]:
                mlflow.set_tag("skipped", "; ".join(result["skipped"]))
            for p in artifacts:
                mlflow.log_artifact(str(p))
            run_id = active.info.run_id
        log.info("mlflow: run %s logged to %r", run_id, EXPERIMENT)
        return run_id
    except Exception as exc:  # pragma: no cover
        log.warning("mlflow logging failed (%s)", exc)
        return None


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dsn", default="host=localhost dbname=protea user=protea password=protea")
    p.add_argument("--obo", default="~/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo")
    p.add_argument("--per-bucket-cap", type=int, default=2000,
                   help="max DB proteins per length bucket")
    p.add_argument("--n-pairs", type=int, default=20_000, help="protein pairs per bucket")
    p.add_argument("--kwta-k", type=int, nargs="+", default=[64, 128])
    p.add_argument("--min-bucket-n", type=int, default=50)
    p.add_argument("--dict-dim", type=int, default=2048)
    p.add_argument("--pool-top-k", type=int, default=128)
    p.add_argument("--attn-dim", type=int, default=256)
    p.add_argument("--pool-epochs", type=int, default=120)
    p.add_argument("--pool-train-pairs", type=int, default=200_000)
    p.add_argument("--skip-chunk-learned", action="store_true")
    p.add_argument("--skip-residue-learned", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--no-mlflow", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    result = run(args)
    out_dir = Path(args.out_dir) if args.out_dir else Path.cwd() / "sdr_factorial_out"
    artifacts = write_artifacts(result, out_dir)
    run_id = None if args.no_mlflow else log_to_mlflow(args, result, artifacts)
    print_report(result)
    if run_id:
        print(f"\nMLflow run: {run_id}")
    print(f"artifacts -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
