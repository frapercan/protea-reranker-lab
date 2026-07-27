"""Assemble the ENTAIL_KWTA deliverable: (a) separability, (b) conversion vs anchor with CI + controls,
(c) the DeepGO-SE random-level diagnosis, and the one-line verdict."""
import json
from pathlib import Path

ROOT = Path("/home/frapercan/Thesis2")
KW = ROOT / "storage/deepgose_kwta"
FA = ROOT / "storage/deepgose_faithful"
EK = ROOT / "storage/entail_kwta"
REGEN = ROOT / "storage/regen_headline"
ANCHOR = {"LK-BPO": 0.31323, "PK-BPO": 0.14351}

mavg = json.load(open(KW / "measure_avg.json"))          # separability + generator conversion (true frame)
ci = json.load(open(EK / "convert_ci.json"))             # rescore + bootstrap CI (this run)
h2h = json.load(open(FA / "headtohead_result.json"))     # DeepGO-SE standalone true-frame
rob = json.load(open(FA / "robustness_checks.json"))     # standalone submission-construction sweep
lkleak = json.load(open(EK / "lk_leakage.json")) if (EK / "lk_leakage.json").exists() else None

def best_gen(cell):
    best = None
    for k, d in mavg[cell]["convert"].items():
        db = d.get("delta_B")
        if db is None: continue
        if best is None or db > best[1]: best = (k, db, d)
    return best

out = {"model": ci["model"], "anchor": ANCHOR, "a_separability": {}, "b_conversion": {}, "c_diagnostic": {}}

# ---------- (a) SEPARABILITY ----------
for cell in ["LK-BPO", "PK-BPO"]:
    s = mavg[cell]["separability"]
    out["a_separability"][cell] = {
        "SE_mean_auc": s["SE_mean_auc"], "reranker_mean_auc": s["reranker_mean_auc"],
        "SE_minus_reranker": s["SE_minus_reranker"], "n_proteins": s["n_proteins"],
        "vs_deepgose_se_0626": round(s["SE_mean_auc"] - 0.626, 4)}

# ---------- (b) CONVERSION ----------
for cell in ["LK-BPO", "PK-BPO"]:
    bg = best_gen(cell)
    row = {"anchor": ANCHOR[cell], "anchor_reproduced": mavg[cell]["A_pool_only"],
           "generator_best_K": bg[0], "generator_best_delta_B": bg[1],
           "generator_best_delta_uniform_control": bg[2].get("delta_uniform"),
           "generator_best_delta_random_control": bg[2].get("delta_random")}
    if cell in ci:
        c = ci[cell]
        row["rescore_SE"] = c.get("B_rescore_SE")
        if isinstance(c.get("B_rescore_SE"), dict):
            b = c["B_rescore_SE"]
            row["rescore_delta"] = b["delta"]; row["rescore_ci"] = [b.get("ci_lo"), b.get("ci_hi")]
            row["rescore_p_gt0"] = b.get("p_delta_gt0")
            row["rescore_random_control_delta"] = c["R_rescore_random"]["delta"]
        else:
            row["rescore_delta"] = c.get("delta")
        if "G_generator_top5" in c:
            g = c["G_generator_top5"]
            row["generator_top5_delta"] = g["delta"]; row["generator_top5_ci"] = [g["ci_lo"], g["ci_hi"]]
            row["generator_top5_p_gt0"] = g["p_delta_gt0"]
    if cell == "LK-BPO" and lkleak:
        row["lk_rescore_raw_no_exclusion_delta"] = lkleak["raw_no_exclusion"]["delta"]
        row["lk_rescore_leakage_clean_delta"] = lkleak["leakage_clean_v227known_excluded"]["delta"]
        row["lk_leakage_note"] = ("raw LK has NO -known exclusion; 100% of LK eval proteins are in SE "
            "training and 47% of LK gt BP terms were v227 training labels. The raw +delta is SE "
            "surfacing MEMORIZED terms. Under the leakage-clean frame (v227-known excluded, the LK "
            "analog of PK -known) the delta is the honest number.")
        # the leakage-clean delta is the deciding LK conversion number
        row["rescore_delta"] = lkleak["leakage_clean_v227known_excluded"]["delta"]
    out["b_conversion"][cell] = row

# ---------- (c) DIAGNOSTIC ----------
stand = rob.get("deepgose_standalone_true_frame_f_micro_w", {})
recall = rob.get("deepgose_candidate_recall_ceiling_IA_weighted_new_terms", {})
h2h_bp = h2h["variants"]["faithful_avg"]["cells"]["bpo"]["boot"]
out["c_diagnostic"] = {
    "deepgose_se_standalone_bpo_f_micro_w": round(h2h_bp["f_chal"], 5),
    "deepgose_se_standalone_delta_vs_anchor": round(h2h_bp["delta"], 5),
    "deepgose_se_standalone_ci": [round(h2h_bp["ci_lo"], 5), round(h2h_bp["ci_hi"], 5)],
    "proper_construction_ceiling_bpo": {
        "topK_sweep_avg": stand.get("topk_sweep_avg", {}).get("BP"),
        "min_combination_top500": stand.get("min_combination_top500", {}).get("BP"),
        "rankpct_top2000": stand.get("rankpct_top2000", {}).get("BP"),
        "ceiling": stand.get("ceiling", {}).get("BP")},
    "candidate_recall_ceiling_new_terms_bpo": recall.get("BP"),
    "diagnosis": (
        "The 0.011 random-level was the SUBMISSION CONSTRUCTION exposing a candidate-generation "
        "ceiling, NOT the entailment principle. On a RESTRICTED reachable candidate set the SE score "
        "separates true from false (PK-BPO AUC 0.6263 > reranker 0.491). But as a STANDALONE full-vocab "
        "submission the true NEW BP terms are largely OUTSIDE the model's high-scoring set: candidate "
        "recall ceiling is 5.7% at top500. Every construction we tried (per-protein topK 3..2000, "
        "min/avg ensemble combos, rankpct, floor 0.01) caps BP at 0.011-0.0126 -- barely above the "
        "global-tau full-vocab 0.011 -- because you cannot rank into the top what the model never "
        "surfaces. So: construction is bad AND the underlying full-vocab candidate set is thin; the "
        "0.626 separability is real but lives on a reachable set the standalone generator cannot build.")}

# ---------- VERDICT ----------
verdict = {}
for cell in ["LK-BPO", "PK-BPO"]:
    sep = out["a_separability"][cell]["SE_minus_reranker"]
    conv = out["b_conversion"][cell]
    gen = conv["generator_best_delta_B"]
    resc = conv.get("rescore_delta")
    separates = sep is not None and sep > 0.01
    converts = (gen is not None and gen > 0.0034) or (isinstance(resc, (int, float)) and resc > 0.0034)
    if separates and converts:
        label = "GO (SUSPECT - needs independent verifier)"
    elif separates and not converts:
        label = "SEPARATES-BUT-CAPPED (real separability, does NOT convert to f_micro_w -> the wall is fundamental)"
    else:
        label = "NO-GO (does not separate and does not convert)"
    verdict[cell] = {"separability_SE_minus_reranker": sep, "generator_best_delta": gen,
                     "rescore_delta": resc, "verdict": label}
out["verdict"] = verdict
out["headline"] = (
    "Clean semantic entailment over our learned k-WTA rep SEPARATES as well as DeepGO-SE (PK-BPO "
    f"AUC {out['a_separability']['PK-BPO']['SE_mean_auc']} vs reranker "
    f"{out['a_separability']['PK-BPO']['reranker_mean_auc']}; = DeepGO-SE's 0.626) but does NOT CONVERT "
    "(every true-frame f_micro_w delta vs the deployed anchor is negative, rescore + generator + "
    "controls, CI excludes 0 on the negative side). DeepGO-SE's random-level was the submission "
    "construction / candidate-generation ceiling, not the entailment principle. VERDICT = "
    "SEPARATES-BUT-CAPPED: the BP wall is FUNDAMENTAL (separability at the reachable tail does not "
    "buy f_micro_w at the metric's operating point), not the broken machinery.")

REGEN.mkdir(exist_ok=True)
json.dump(out, open(REGEN / "ENTAIL_KWTA.json", "w"), indent=1)
json.dump(out, open(EK / "ENTAIL_KWTA.json", "w"), indent=1)

def sd(x): return "n/a" if x is None else (f"{x:+.5f}" if isinstance(x, float) else str(x))

md = []
md.append("# Clean semantic entailment over our learned k-WTA representation: does the ONE mechanism "
          "that showed signal (DeepGO-SE's entailment score) SEPARATE and CONVERT when stripped of the "
          "broken machinery?\n")
md.append("**Headline.** " + out["headline"] + "\n")
md.append("## The model (minimal, fully controlled)\n")
m = out["model"]
md.append(f"- **Architecture**: {m['architecture']}")
md.append(f"- **Axioms**: {m['axioms']}")
md.append(f"- **Stripped**: no GAT/DGL ({m['no_gat']}), no MF-preds pipeline ({m['no_mfpreds']}), "
          f"no deepgo2 harness ({m['no_deepgo2_harness']})")
md.append(f"- **Temporal gate**: {m['temporal_gate']}")
md.append(f"- **Corpus/training**: {m['corpus']}\n")
md.append("The protein tower is a small MLP mapping the learned k-WTA code (d8979601, 2048-d) to a point "
          "in the GO-ball space; each GO term is a geometric region (center + radius) trained on the EL "
          "normal forms; the entailment score is the ensemble-consensus degree the protein's point is "
          "subsumed inside the term's ball. This IS DeepGO-SE's semantic-entailment principle, and "
          "nothing else.\n")

md.append("## (a) SEPARABILITY -- per-protein AUC on the reachable pool tail (diagnostic)\n")
md.append("| cell | SE AUC | reranker AUC | SE - reranker | vs DeepGO-SE 0.626 | n_proteins |")
md.append("|---|---|---|---|---|---|")
for cell in ["LK-BPO", "PK-BPO"]:
    s = out["a_separability"][cell]
    md.append(f"| {cell} | **{s['SE_mean_auc']}** | {s['reranker_mean_auc']} | {sd(s['SE_minus_reranker'])} | "
              f"{sd(s['vs_deepgose_se_0626'])} | {s['n_proteins']} |")
md.append("")
md.append("PK-BPO: clean entailment over k-WTA reaches **0.6263**, matching DeepGO-SE's 0.626 and beating "
          "the deployed reranker's 0.491 (chance). The separability signal is REAL and reproduced by the "
          "clean model. LK-BPO: a small edge (0.867 vs 0.829).\n")

md.append("## (b) CONVERSION -- true-frame f_micro_w vs the DEPLOYED anchor (DECIDES)\n")
md.append("Frame: prop=fill norm=cafa no_orphans toi, PK exclude=groundtruth_PK_known, cafa_eval under "
          "PROTEA/.venv, temporal gate v227->v230. Rescore = volume-matched rank-match of the reranker "
          "score multiset in SE order (isolates ranking). Generator = union top-k SE proposals into the "
          "pool. CIs are paired protein bootstrap (1000x, exact IA-weighted micro-F).\n")
md.append("| cell | anchor (reproduced) | RESCORE delta [CI] p>0 | rand-order ctrl | GENERATOR top5 delta [CI] | prior sweep best-K | matched-vol ctrl |")
md.append("|---|---|---|---|---|---|---|")
for cell in ["LK-BPO", "PK-BPO"]:
    c = out["b_conversion"][cell]
    resc_ci = c.get("rescore_ci"); resc = c.get("rescore_delta")
    resc_s = f"{sd(resc)} {resc_ci if resc_ci else ''} p>0={c.get('rescore_p_gt0','n/a')}"
    gt5 = c.get("generator_top5_delta"); gci = c.get("generator_top5_ci")
    gen_s = (f"{sd(gt5)} {gci}" if gt5 is not None else "n/a")
    md.append(f"| {cell} | {c['anchor']} ({c['anchor_reproduced']}) | {resc_s} | "
              f"{sd(c.get('rescore_random_control_delta','n/a') if isinstance(c.get('rescore_random_control_delta'),(int,float)) else None)} | {gen_s} | "
              f"{sd(c['generator_best_delta_B'])} ({c['generator_best_K']}) | "
              f"{sd(c['generator_best_delta_uniform_control'])} |")
md.append("")
md.append("(RESCORE and GENERATOR top5 with CI are this run's independent measurements via the exact "
          "IA-weighted micro-F decomposition + paired bootstrap; the prior-sweep best-K column is the "
          "corroborating true-frame generator sweep from deepgose_kwta/measure_avg.json, negative at "
          "every K. LK rescore delta is the LEAKAGE-CLEAN value; see below.)\n")
md.append("Every LEAKAGE-CLEAN conversion delta is NEGATIVE, in both modes and under both controls, with "
          "the PK CIs excluding zero (rescore [-0.042,-0.028], generator [-0.024,-0.015], p>0=0). The SE "
          "ordering does beat random-order rescore (PK -0.035 vs -0.098) -- the 0.626 separability is "
          "real -- but it still loses to the deployed reranker at the metric's operating point. The "
          "reachable tail is a tiny IA fraction at catastrophic precision (top25 added-true IA-precision "
          "~0.002-0.004, i.e. 100-300x more false IA than true), so a 0.626 AUC sits far below the "
          "anchor's threshold.\n")
if lkleak:
    md.append("### LK-BPO was a SUSPECT: +0.0207 was memorization, not conversion\n")
    md.append(f"The raw LK rescore looked POSITIVE (+{lkleak['raw_no_exclusion']['delta']}). LK, unlike "
              "PK, has no `-known` exclusion, yet 100% of LK eval proteins are in the SE training corpus "
              "and 47% of LK ground-truth BP terms were already v227 training labels. So SE was reordering "
              "the pool to surface terms it had MEMORIZED for these exact proteins, and LK let them count. "
              "Building the LK analog of PK's `-known` (each LK protein's v227 t0 BP labels, propagated, "
              "excluded at scoring) collapses it:\n")
    md.append("| LK-BPO frame | anchor | SE rescore | delta |")
    md.append("|---|---|---|---|")
    md.append(f"| raw (no exclusion, leakage-exposed) | {lkleak['raw_no_exclusion']['anchor']} | "
              f"{lkleak['raw_no_exclusion']['rescore_SE']} | **+{lkleak['raw_no_exclusion']['delta']}** |")
    lc = lkleak['leakage_clean_v227known_excluded']
    md.append(f"| leakage-clean (v227-known excluded) | {lc['anchor']} | {lc['rescore_SE']} | "
              f"**{sd(lc['delta'])}** |")
    md.append("")
    md.append("Under the leakage-clean frame LK behaves exactly like PK (-0.044 vs PK -0.035). The one "
              "positive this whole experiment produced was a temporal-gate hole in the LK cell, and it "
              "closes.\n")

md.append("## (c) DeepGO-SE random-level DIAGNOSIS -- model or submission construction?\n")
d = out["c_diagnostic"]
md.append(f"- DeepGO-SE (faithful, standalone full-vocab, global tau) PK-BPO f_micro_w = "
          f"**{d['deepgose_se_standalone_bpo_f_micro_w']}** (delta {sd(d['deepgose_se_standalone_delta_vs_anchor'])} "
          f"vs anchor, CI {d['deepgose_se_standalone_ci']}).")
md.append(f"- Proper-construction ceiling (per-protein topK sweep 3..2000): BP tops at "
          f"~{max(d['proper_construction_ceiling_bpo']['topK_sweep_avg'].values())}; "
          f"min-combo top500 {d['proper_construction_ceiling_bpo']['min_combination_top500']}; "
          f"rankpct {d['proper_construction_ceiling_bpo']['rankpct_top2000']}. Ceiling "
          f"{d['proper_construction_ceiling_bpo']['ceiling']}.")
md.append(f"- Candidate recall ceiling (fraction of true NEW BP terms in DeepGO-SE top-K): "
          f"{d['candidate_recall_ceiling_new_terms_bpo']}.")
md.append("")
md.append("**Diagnosis.** " + d["diagnosis"] + "\n")

md.append("## VERDICT\n")
for cell in ["LK-BPO", "PK-BPO"]:
    v = verdict[cell]
    md.append(f"- **{cell}**: separability SE-reranker {sd(v['separability_SE_minus_reranker'])}, "
              f"rescore delta {sd(v['rescore_delta'])}, generator best delta {sd(v['generator_best_delta'])} "
              f"-> **{v['verdict']}**")
md.append("")
md.append("**One line.** Clean semantic entailment over our learned representation SEPARATES as well as "
          "DeepGO-SE (0.626) but does NOT CONVERT to f_micro_w -> **SEPARATES-BUT-CAPPED**; and "
          "DeepGO-SE's random-level failure was the **submission construction / candidate-generation "
          "ceiling, not the machinery and not the entailment principle** -- the BP wall is FUNDAMENTAL "
          "(a separability/precision limit at the reachable tail), which is why isolating the clean "
          "principle does not move it.\n")

open(REGEN / "ENTAIL_KWTA.md", "w").write("\n".join(md))
open(EK / "ENTAIL_KWTA.md", "w").write("\n".join(md))
print("wrote ENTAIL_KWTA.{md,json} to", REGEN)
print(json.dumps(verdict, indent=1))
