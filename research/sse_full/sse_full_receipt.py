"""Assemble the SSE_FULL receipt (MD + JSON) from all stage outputs."""
import json
from pathlib import Path
ROOT = Path("/home/frapercan/Thesis2")
SSE = ROOT / "storage/sse_full"; REG = ROOT / "storage/regen_headline"
def rd(p):
    try: return json.load(open(p))
    except Exception: return {}
train = rd(SSE / "sse_full_train.json")
modeA = rd(SSE / "modeA_min.json")
mB_re = rd(SSE / "modeB_retrain.json")
mB_ca = rd(SSE / "modeB_cafa.json")

def verdict_A(key):
    c = modeA.get(key)
    if not c: return "n/a"
    best = None
    for K in ["top5", "top10", "top25"]:
        d = c.get("convert", {}).get(K, {}).get("delta_B")
        if d is not None and (best is None or d > best): best = d
    if best is None: return "n/a"
    if best > 0.001: return f"BETTER (+{best})"
    if best < -0.001: return f"WORSE ({best})"
    return f"TIE ({best})"

def verdict_B(key):
    c = mB_ca.get(key)
    if not c: return "n/a"
    bs = c.get("bootstrap", {})
    dm = bs.get("delta_mean"); ci = bs.get("ci95")
    dshuf = c.get("delta_sse_shuffled_vs_baseline")
    is_lk = key.startswith("lk")
    if dm is None or ci is None: return "n/a"
    sig = ci[0] > 0
    hurts = ci[1] < 0
    # shuffled-control gate: a clean lever must beat its own signal-destroyed control by a clear margin
    shuf_ok = (dshuf is None) or (abs(dshuf) < 0.5 * abs(dm) if dm != 0 else False)
    if not sig:
        return f"NO LEVER (tie, {dm:+} CI{ci})" if not hurts else f"HURTS ({dm:+} CI{ci})"
    # CI clears zero:
    if is_lk:
        return (f"SUSPECT-MEMORIZATION (+{dm} CI{ci}; LK has NO -known, so SSE re-derives already-known "
                f"annotations; shuffled control {dshuf:+} -> NOT a genuine lever)")
    if not shuf_ok:
        return f"SUSPECT (+{dm} CI{ci}; shuffled control {dshuf:+} nearly as large -> distributional, not signal)"
    return f"LEVER (+{dm} CI{ci}; shuffled {dshuf:+})"

out = {"config": train.get("config"), "train_aspects": train.get("aspects"),
       "modeA_generator": modeA, "modeB_retrain": mB_re, "modeB_cafa": mB_ca,
       "verdicts": {}}
cells = [f"{c}-{a}" for c in ["pk", "lk"] for a in ["mfo", "bpo", "cco"]]
Acells = [f"{c}-{a}" for c in ["PK", "LK"] for a in ["mfo", "bpo", "cco"]]
out["verdicts"]["modeA_vs_twotower"] = {k: verdict_A(k) for k in Acells}
out["verdicts"]["modeB_feature_lever"] = {k: verdict_B(k) for k in cells}
json.dump(out, open(REG / "SSE_FULL.json", "w"), indent=1, default=float)

L = []
L.append("# SSE Full-Corpus: Mode A (generator) + Mode B (reranker feature)\n")
L.append("Sparse Semantic-Entailment scaled to the full experimental corpus, all three aspects. "
         "Two-tower comparison (Mode A) and reranker-feature integration (Mode B), true-frame "
         "f_micro_w under the temporal gate (train<=v225, blind eval v227-v230).\n")
cfg = train.get("config", {})
L.append("## Config run\n")
L.append(f"- Corpus: {cfg.get('corpus')} (ESM2-3B mean, 2560-d); eval proteins held out of ALL "
         f"training: {cfg.get('n_eval_heldout')}")
L.append(f"- SDR N={cfg.get('N')} protein-bits={cfg.get('PK')} ensemble K={cfg.get('K_SEEDS')} "
         f"epochs<={cfg.get('EPOCHS')} axiom-weight={cfg.get('GAMMA_AX')}; MIN positives/term={cfg.get('MIN_SUB')}")
for asp, a in (train.get("aspects") or {}).items():
    L.append(f"- {asp.upper()}: corpus {a['n_corpus']} prot, vocab {a['n_terms']} terms, "
             f"axioms nf1={a['nf1']}/nf4={a['nf4']}, mean val AUC {a['mean_val_auc']:.4f}")
L.append("")
L.append("## Mode A -- SSE as generator vs the deployed two-tower (PK/LK x aspect)\n")
L.append("Best top-k delta of adding SSE out-of-pool proposals to the deployed two-tower+reranker pool "
         "(true-frame f_micro_w vs reproduced anchor). Positive = SSE adds true terms the two-tower missed.\n")
L.append("| cell | anchor | SSE sep-AUC | rr sep-AUC | best deltaB | rand-order | matched-vol | verdict |")
L.append("|---|---|---|---|---|---|---|---|")
for k in Acells:
    c = modeA.get(k, {})
    if not c: L.append(f"| {k} | - | - | - | - | - | - | n/a |"); continue
    sep = c.get("separability", {})
    conv = c.get("convert", {})
    bestK = max(conv, key=lambda K: (conv[K].get("delta_B") if conv[K].get("delta_B") is not None else -9),
                default=None) if conv else None
    cc = conv.get(bestK, {}) if bestK else {}
    L.append(f"| {k} | {c.get('anchor_reproduced')} | {sep.get('SSE_auc')} | {sep.get('reranker_auc')} "
             f"| {cc.get('delta_B')} ({bestK}) | {cc.get('delta_random')} | {cc.get('delta_uniform')} "
             f"| {verdict_A(k)} |")
L.append("")
L.append("_All three PK cells (honest `-known` frame): SSE generation is WORSE than the two-tower and "
         "tracks its own random-order control (no ranking signal; the loss is volume, matched-vol far "
         "worse). The small LK-BPO/LK-CCO Mode-A positives (+0.003/+0.023) are in the no-`-known` frame "
         "= the same memorization artifact seen in Mode B, not a genuine generation gain._\n")
L.append("## Mode B -- SSE as a reranker feature (retrained per-category LightGBM)\n")
L.append("| cell | deployed anchor | baseline | +sse | delta(+sse) | bootstrap CI95 | +sse_shuffled | verdict |")
L.append("|---|---|---|---|---|---|---|---|")
for k in cells:
    c = mB_ca.get(k, {})
    if not c: L.append(f"| {k} | - | - | - | - | - | - | n/a |"); continue
    arm = c.get("arms", {}); bs = c.get("bootstrap", {})
    L.append(f"| {k} | {c.get('deployed_anchor_ref')} | {arm.get('baseline',{}).get('f_micro_w')} "
             f"| {arm.get('sse',{}).get('f_micro_w')} | {c.get('delta_sse_vs_baseline')} "
             f"| {bs.get('ci95')} (fpos {bs.get('frac_positive')}) "
             f"| {c.get('delta_sse_shuffled_vs_baseline')} | {verdict_B(k)} |")
L.append("")
gains = (mB_re.get("sse_gain", {}) or {}).get("sse", {})
if gains:
    L.append("### SSE feature importance in the retrained reranker (+sse variant)")
    for cat, g in gains.items():
        L.append(f"- {cat}: sse_score gain rank {g.get('sse_gain_rank')}/{g.get('n_features')} "
                 f"(gain {g.get('sse_score_gain'):.0f})")
    L.append("")
L.append("## Leakage & honest-frame analysis (Mode B)\n")
L.append("The +sse VALIDATION proxy jumped hugely (nk-bpo 0.35->0.60, lk/pk mfo/cco +0.07..+0.08), but "
         "validation proteins are NOT eval-held-out, so their sse_score is IN-FIT (SSE saw their labels). "
         "On the CLEAN held-out TEST (eval v227-v230, eval proteins excluded from SSE training) the gains "
         "collapse:\n")
L.append("- **PK cells (honest frame, `-known` excluded): NO lever.** pk-mfo +0.003 (CI crosses 0), "
         "pk-bpo -0.001 (CI crosses 0), pk-cco +0.008 (CI crosses 0). SSE's +0.135 BP separability edge "
         "does NOT convert even when the reranker can weight it (sse_score ranks #1 by gain but adds nothing "
         "at the operating point).\n")
L.append("- **LK-MFO/LK-CCO show +0.10 but are SUSPECT-MEMORIZATION, not levers.** LK groundtruth has NO "
         "`-known` exclusion, so SSE re-derives a protein's ALREADY-KNOWN annotations from its embedding and "
         "those count as true positives. The shuffled-feature control confirms it: lk-cco shuffled +0.0000 "
         "(all signal is the protein-term SSE value = memorization), lk-mfo shuffled +0.039 (largely "
         "distributional). LK-BPO +0.006 is beaten by its own shuffled control (+0.011) = pure noise.\n")
L.append("- **BP specifically: SSE-as-a-feature is NOT a lever.** pk-bpo -0.001 (CI[-0.005,+0.003]); "
         "lk-bpo tie/noise. The campaign's first BP lever did NOT materialize.\n")
L.append("## Reproducibility note\n")
L.append("SSE is fully reproducible from frozen inputs (seeds fixed, models + meta saved under "
         "storage/sse_full/models/). The deployed two-tower's SVD/projection basis was never persisted, "
         "so its candidate generation is not re-derivable; SSE as a candidate source is auditable where "
         "the two-tower is not, independent of the head-to-head f_micro_w outcome.\n")
L.append("## Verdicts\n")
L.append("**(1) Mode A -- SSE better or worse than the two-tower as a candidate source (per aspect PK):**")
for a in ["mfo", "bpo", "cco"]:
    L.append(f"  - PK-{a.upper()}: {verdict_A('PK-'+a)}")
L.append("")
L.append("**(2) Mode B -- is SSE-as-a-feature a lever (esp BP):**")
for k in cells:
    L.append(f"  - {k}: {verdict_B(k)}")
L.append("")
(REG / "SSE_FULL.md").write_text("\n".join(L))
print("WROTE", REG / "SSE_FULL.md", "and SSE_FULL.json")
print("\n".join(L[:40]))
