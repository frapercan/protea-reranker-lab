"""Merge gate + board-faithful validation + robustness into one result_9cell.json
and write the full 9-cell board-faithful table (graft+InterPro baseline vs the
learned-gating BP cells; all other cells unchanged by construction)."""
import os, json
HERE = os.path.dirname(os.path.abspath(__file__))
SC = "/home/frapercan/Thesis2/repositories/protea-reranker-lab/results/sparse_classifier"

gate = json.load(open(os.path.join(HERE, "result_9cell.json")))
vbf = json.load(open(os.path.join(HERE, "valid_boardfaithful.json")))
rob = json.load(open(os.path.join(HERE, "robustness_check.json")))
ipr = json.load(open(os.path.join(SC, "interpro2go_test", "blend_9cell.json")))
base9 = ipr["naivemax_bponly_9cell"]  # graft+InterPro baseline (current board-faithful)

# build the learned-gating 9-cell = baseline with LK-bpo / PK-bpo replaced
learned9 = {c: {a: base9[c][a]["f_micro_w"] if isinstance(base9[c][a], dict) else base9[c][a]
                for a in ["bpo", "mfo", "cco"]} for c in ["nk", "lk", "pk"]}
learned9["lk"]["bpo"] = gate["test"]["lk"]["learned_lit"]["f_micro_w"]
learned9["pk"]["bpo"] = gate["test"]["pk"]["learned_lit"]["f_micro_w"]
base9flat = {c: {a: (base9[c][a]["f_micro_w"] if isinstance(base9[c][a], dict) else base9[c][a])
                 for a in ["bpo", "mfo", "cco"]} for c in ["nk", "lk", "pk"]}

out = {
    "question": "Does LEARNED GATING of literature features (text->GO max-cosine, "
                "S-PubMedBERT) into the BP meta-reranker (ADR-D43) close the BP gap vs "
                "TransFew, beyond the graft+InterPro baseline?",
    "protocol": "Union BP evidence pool (reranker UNION InterPro2GO) per category. "
                "Features: BASE (s_rerank, s_interpro, s_max, indicators, ia, depth, "
                "per-protein difficulty) + LIT (text max-cosine, within-protein text "
                "rank/percentile, top-1/3/5, text-minus-reranker rank gap, has_text). "
                "Gate = LightGBM trained on v225->v227 validation; include-literature "
                "DECISION by BOARD-FAITHFUL cafa_eval on a held-out 25% of validation "
                "proteins. Selected model applied to v227->v230 TEST, scored board-"
                "faithfully (OBO, IA, TOI, prop=fill, norm=cafa, no_orphans, PK excl "
                "PK_known). Literature routed into BP only.",
    "baselines_bp_fmax": {"graft_interpro_LK": base9flat["lk"]["bpo"],
                          "graft_interpro_PK": base9flat["pk"]["bpo"],
                          "transfew_LK": 0.512, "transfew_PK": 0.294},
    "validation_board_faithful_bpo": {c: vbf[c]["board_faithful_valid_bpo"] for c in vbf},
    "validation_decision": {c: {"chosen": vbf[c]["chosen"],
                                "literature_helps": vbf[c]["literature_helps"]} for c in vbf},
    "test_bpo": {
        "lk": gate["test"]["lk"],
        "pk": gate["test"]["pk"],
    },
    "robustness_regularized_monotone_gate_test_bpo": rob,
    "nine_cell_board_faithful": {
        "graft_interpro_baseline": base9flat,
        "learned_gating_applied_bp": learned9,
        "note": "MF/CC and NK cells identical by construction (literature routed to "
                "BP only). LK-bpo and PK-bpo are the only changed cells.",
    },
    "deltas_test_bp": {
        "LK_learned_lit_minus_graft": round(gate["test"]["lk"]["learned_lit"]["f_micro_w"]
                                            - base9flat["lk"]["bpo"], 5),
        "PK_learned_lit_minus_graft": round(gate["test"]["pk"]["learned_lit"]["f_micro_w"]
                                            - base9flat["pk"]["bpo"], 5),
    },
    "verdict": "NEGATIVE (transfer failure). Board-faithful validation selects the "
               "learned literature-gated gate (LK 0.585 vs naivemax 0.538; PK 0.777 vs "
               "0.751) and literature adds signal on validation, BUT the learned gate "
               "does NOT transfer to test: it REGRESSES every BP cell vs the parameter-"
               "free graft+InterPro naive-max (LK 0.428->0.385; PK 0.216->0.161). A "
               "heavily-regularized monotone gate also regresses (LK 0.396, PK 0.175). "
               "Literature features DO get gain in the gate, so the orthogonal signal "
               "is real and learnable, but it is not Fmax-convertible here. Root cause: "
               "literature/InterPro features exist only for the eval window, so the gate "
               "can only be trained on the single v225->v227 validation snapshot and "
               "overfits that window; the multi-snapshot-trained base reranker and the "
               "parameter-free naive-max combiner are more robust out-of-window. No "
               "injectable emitted (validation-backed but test-regressive).",
}
json.dump(out, open(os.path.join(HERE, "result_9cell.json"), "w"), indent=2)
print(json.dumps({"baselines": out["baselines_bp_fmax"],
                  "test_lk": {k: v["f_micro_w"] for k, v in gate["test"]["lk"].items() if isinstance(v, dict)},
                  "test_pk": {k: v["f_micro_w"] for k, v in gate["test"]["pk"].items() if isinstance(v, dict)},
                  "deltas": out["deltas_test_bp"]}, indent=2))
print("written result_9cell.json")
