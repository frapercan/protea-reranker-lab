"""Assemble result_9cell.json + SUMMARY.txt for Phase 1 (GCN joint scorer).

GATE: with the GCN label encoder + joint scorer producing a meaningful compatibility for
EVERY candidate, does re-running the Phase 0 union experiment now CONVERT PK-BP (fix the
-0.0266 regression, ideally toward TransFew 0.294)? Single-variable vs Phase 0: same union
pool, the candidate score is the joint scorer's per-cut compatibility instead of the frozen
two-tower logit.
"""
import os, json, glob

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE0 = os.path.join(os.path.dirname(HERE), "bp_structural_phase0")

CHAMPION_9 = {
    "nk": {"bpo": 0.33745, "mfo": 0.66368, "cco": 0.47696},
    "lk": {"bpo": 0.42807, "mfo": 0.56384, "cco": 0.46147},
    "pk": {"bpo": 0.21797, "mfo": 0.24163, "cco": 0.26554},
}
TRANSFEW = {"lk": 0.512, "pk": 0.294}
PHASE0_UNION = {"lk": 0.44135, "pk": 0.18971}      # Phase 0 union-pool board-faithful BP
CONVERT_EPS = 0.005


def _feat_imp():
    allp = os.path.join(HERE, "train_meta_all.json")
    if os.path.exists(allp):
        return json.load(open(allp))
    merged = {}
    for f in sorted(glob.glob(os.path.join(HERE, "train_meta_*_*.json"))):
        merged.update(json.load(open(f)))
    return merged


def main():
    sv = json.load(open(os.path.join(HERE, "score_variants.json")))
    cells = sv["cells"]
    es = {}
    if os.path.exists(os.path.join(HERE, "eval_standalone.json")):
        es = json.load(open(os.path.join(HERE, "eval_standalone.json")))

    union9 = json.loads(json.dumps(CHAMPION_9))
    union9["lk"]["bpo"] = cells["lk"]["union_pool"]["f_micro_w"]
    union9["pk"]["bpo"] = cells["pk"]["union_pool"]["f_micro_w"]

    verdict = {}
    for cat in ["lk", "pk"]:
        c = cells[cat]
        base_f = c["retrained_base"]["f_micro_w"]
        union_f = c["union_pool"]["f_micro_w"]
        d = round(union_f - base_f, 5)
        d_vs_p0 = round(union_f - PHASE0_UNION[cat], 5)
        converts = (union_f >= base_f + CONVERT_EPS) if cat == "pk" else (d >= CONVERT_EPS)
        verdict[cat] = {
            "champion_reproduced_f": c["champion_reproduced"]["f_micro_w"],
            "base_reproduces_champion": abs(base_f - CHAMPION_9[cat]["bpo"]) < 0.003,
            "base_f": base_f, "union_gcn_f": union_f,
            "delta_union_minus_base": d,
            "phase0_union_f": PHASE0_UNION[cat], "delta_vs_phase0_union": d_vs_p0,
            "transfew": TRANSFEW[cat], "gap_to_transfew": round(union_f - TRANSFEW[cat], 5),
            "rc_ceiling_base": c["retrained_base"]["rc_ceiling_tau0"],
            "rc_ceiling_union": c["union_pool"]["rc_ceiling_tau0"],
            "rc_realized_base": c["retrained_base"]["rc_realized_at_best"],
            "rc_realized_union": c["union_pool"]["rc_realized_at_best"],
            "joint_scoring_converts": bool(converts),
        }

    pk = verdict["pk"]; lk = verdict["lk"]
    pk_converts = pk["joint_scoring_converts"]
    if pk["union_gcn_f"] > PHASE0_UNION["pk"] and pk["union_gcn_f"] >= pk["base_f"] - 0.002:
        pk_status = ("RECOVERS: GCN joint scoring lifts PK-BP union above the Phase 0 "
                     "regression and back to/above champion")
    elif pk["union_gcn_f"] > PHASE0_UNION["pk"]:
        pk_status = ("PARTIAL: GCN joint scoring lifts PK-BP union above Phase 0's "
                     "diluted union but not yet to champion")
    else:
        pk_status = "NO: GCN joint scoring does not lift PK-BP union above Phase 0"

    headline = (f"JOINT SCORING {'CONVERTS' if pk_converts else 'DOES NOT FULLY CONVERT'} PK-BP: "
                f"champion {pk['base_f']:.5f} | phase0-union {PHASE0_UNION['pk']:.5f} | "
                f"union+GCN {pk['union_gcn_f']:.5f} | TransFew {TRANSFEW['pk']:.3f}")

    result = {
        "question": ("Does a learnable GCN GO-label encoder + joint protein<->term scorer "
                     "(meaningful compatibility for EVERY candidate, incl. clf-only BP terms) "
                     "fix the Phase 0 PK-BP precision-dilution regression when fused into the "
                     "union reranker, board-faithful, 227->230?"),
        "method": ("Single-variable vs Phase 0: SAME union pool; the candidate score is the "
                   "GCN joint scorer's per-cut compatibility (gcn_scorer.py) instead of the "
                   "frozen two-tower logit. PK union = percut full pool + gcn_score + gcn_rank; "
                   "LK union = Phase 0 pool with tt_clf_score := gcn_score. Board-faithful "
                   "cafa_eval (OBO/IA/TOI, prop=fill, norm=cafa, no_orphans, PK excludes "
                   "PK_known), InterPro2GO naive-max graft."),
        "temporal_honesty": ("GCN head trained on the v227 snapshot (test t0); per-cut honest "
                             "scoring at inference (per_cut/go_sparse_codes_v{N}.npz, frozen "
                             "v227 SVD basis). No future-snapshot code touches a training row."),
        "board_faithful_bp": {cat: {
            "champion_canonical": CHAMPION_9[cat]["bpo"],
            "retrained_base": cells[cat]["retrained_base"],
            "phase0_union": PHASE0_UNION[cat],
            "union_pool_gcn": cells[cat]["union_pool"],
            "transfew": TRANSFEW[cat],
            "delta_union_minus_base": verdict[cat]["delta_union_minus_base"],
            "delta_vs_phase0_union": verdict[cat]["delta_vs_phase0_union"],
            "gap_to_transfew": verdict[cat]["gap_to_transfew"],
        } for cat in ("lk", "pk")},
        "nine_cell_board_faithful": {
            "champion_graft_interpro": CHAMPION_9,
            "union_gcn_in_frame": union9,
            "note": ("Only LK-bpo and PK-bpo change; MF/CC/NK identical by construction."),
        },
        "standalone_scoring_eval": es,
        "feature_importance": _feat_imp(),
        "verdict_detail": verdict,
        "pk_status": pk_status,
        "headline": headline,
    }
    json.dump(result, open(os.path.join(HERE, "result_9cell.json"), "w"), indent=2)
    write_summary(result, verdict, pk_status, es)
    print(json.dumps(result["board_faithful_bp"], indent=2))
    print("HEADLINE:", headline)


def write_summary(result, verdict, pk_status, es):
    lk, pk = verdict["lk"], verdict["pk"]
    h2h = es.get("head_to_head_clf_present", {})
    L = [
        "BP STRUCTURAL LEVER -- PHASE 1: does a learnable GCN GO-label encoder + joint",
        "protein<->term scorer fix the Phase 0 PK-BP precision-dilution regression?",
        "=" * 78,
        "Offline lab only. Board-faithful cafa_eval (OBO, IA, TOI, prop=fill, norm=cafa,",
        "no_orphans, PK excludes PK_known; GT in lafa_gt/). LAFA 227->230 test. GCN label",
        "tower replaces the frozen GO codes (init = fused sparse functional codes, message-",
        "passed over the go_parents DAG); joint scorer = ProjHead protein tower + GCN label",
        "tower, trained end-to-end (ASL + DAG hinge). SAME union pool as Phase 0 -- the only",
        "change is the candidate SCORE (per-cut joint compatibility, gcn_scorer.py).",
        "",
        "STEP 0 SANITY -- champion reproduced (no-gcn base retrain on its own frame):",
        f"  LK-bpo base = {lk['base_f']:.5f}  (champion 0.42807)  reproduced={lk['base_reproduces_champion']}",
        f"  PK-bpo base = {pk['base_f']:.5f}  (champion 0.21631)  reproduced={pk['base_reproduces_champion']}",
        "",
        "RESULT (board-faithful BP, 227->230 test)",
        "-" * 78,
        "                champion   phase0-union   union+GCN    d(GCN-champ)  d(GCN-p0)   TransFew",
        f"  LK-bpo        {lk['base_f']:.5f}    {lk['phase0_union_f']:.5f}      {lk['union_gcn_f']:.5f}     "
        f"{lk['delta_union_minus_base']:+.5f}     {lk['delta_vs_phase0_union']:+.5f}    {lk['transfew']:.3f}",
        f"  PK-bpo        {pk['base_f']:.5f}    {pk['phase0_union_f']:.5f}      {pk['union_gcn_f']:.5f}     "
        f"{pk['delta_union_minus_base']:+.5f}     {pk['delta_vs_phase0_union']:+.5f}    {pk['transfew']:.3f}",
        "",
        "RECALL CEILING vs REALIZED (propagated rc_micro_w at the BP cell)",
        "-" * 78,
        f"  PK  rc_ceiling base->union: {pk['rc_ceiling_base']:.5f} -> {pk['rc_ceiling_union']:.5f}",
        f"      rc_realized@bestFmax  : {pk['rc_realized_base']:.5f} -> {pk['rc_realized_union']:.5f}",
        f"  LK  rc_ceiling base->union: {lk['rc_ceiling_base']:.5f} -> {lk['rc_ceiling_union']:.5f}",
        f"      rc_realized@bestFmax  : {lk['rc_realized_base']:.5f} -> {lk['rc_realized_union']:.5f}",
        "",
        "STANDALONE SCORING QUALITY (does the joint score separate true/false candidates?)",
        "-" * 78,
        f"  recall@100 (v227 held-out): GCN {es.get('recall_at_k_gcn',{}).get('100','?')} "
        f"vs two-tower {es.get('recall_at_k_two_tower',{}).get('100','?')}",
        f"  PK-BP true-vs-false ranking (clf-present rows, n={h2h.get('n','?')}):",
        f"     two-tower AUROC {h2h.get('two_tower_auroc','?')} AP {h2h.get('two_tower_ap','?')}",
        f"     GCN-joint AUROC {h2h.get('gcn_auroc','?')} AP {h2h.get('gcn_ap','?')}",
        f"  GCN on all in-vocab PK-BP candidates: AUROC "
        f"{es.get('gcn_all_invocab',{}).get('auroc','?')} AP {es.get('gcn_all_invocab',{}).get('ap','?')}",
        "",
        "VERDICT",
        "-" * 78,
        f"  {result['headline']}",
        f"  PK: {pk_status}",
        f"  joint_scoring_converts (PK) = {pk['joint_scoring_converts']}",
        "",
        "PER-CELL SHIPPABLE (best-of, board-faithful)",
        "-" * 78,
        f"  LK-bpo: best = {max(lk['base_f'], lk['union_gcn_f']):.5f} "
        f"({'union+GCN' if lk['union_gcn_f'] > lk['base_f'] else 'champion'})",
        f"  PK-bpo: best = {max(pk['base_f'], pk['union_gcn_f']):.5f} "
        f"({'union+GCN' if pk['union_gcn_f'] > pk['base_f'] else 'champion'})",
        "",
        "ARTIFACTS",
        "-" * 78,
        "  label_gcn.py, train_gcn.py (smoke+full), gcn_seed*.pt, gcn_metrics.json,",
        "  gcn_scorer.py, build_union_gcn.py, union_lk_gcn_{train,eval}.parquet,",
        "  train_and_emit_gcn.py, model_{lk,pk}_{base,union}.txt, rerank_bp_*.parquet,",
        "  score_all.py, score_variants.json, eval_standalone.py/json, finalize.py,",
        "  result_9cell.json, SUMMARY.txt.",
    ]
    open(os.path.join(HERE, "SUMMARY.txt"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
