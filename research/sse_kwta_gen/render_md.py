"""Render SSE_KWTA_GEN.md from SSE_KWTA_GEN.json (generalization-capacity receipt)."""
import json
from pathlib import Path
ROOT = Path("/home/frapercan/Thesis2")
R = json.load(open(ROOT / "storage/regen_headline/SSE_KWTA_GEN.json"))
def f(x, n=4):
    return "n/a" if x is None else (f"{x:.{n}f}" if isinstance(x, float) else str(x))
L = []
m = R["model"]
L += ["# Generalization capacity of a single-model, accept-all Sparse Semantic-Entailment (SSE) over the DEPLOYED frozen k-WTA representation",
      "",
      "**What this measures.** How well entailment (sparse set-containment) over our FROZEN champion "
      "k-WTA protein codes (d8979601, 2048-d, 128 active) RANKS true GO terms on the blind v227->v230 "
      "eval, decomposed across three generalization regimes. Threshold-free ranking metrics "
      "(AUROC / AUPR / protein-centric Fmax); the calibrated f_micro_w is a footnote, not the verdict "
      "(accept-all floods it by construction).",
      "",
      "## Model + frozen input frames",
      f"- Protein rep: **{m['protein_rep']}**",
      f"- Train codes: `{m['train_codes']}`",
      f"- Eval codes: `{m['eval_codes']}` -- SAME d8979601 encoder (both 2048-d, exactly 128 active); aligned by the `accs` sidecar.",
      f"- Term tower: {m['term_tower']}",
      f"- K = {m['K']}",
      f"- Temporal gate: {m['temporal_gate']}",
      "",
      "**Regimes** (over eval positive (protein,term) pairs; negatives = vocab terms neither true nor known@t0):",
      "- **R1 known@t0** -- annotation already true at t0 (v227); excluded by the honest `-known` frame = memorization / in-distribution-recall UPPER REFERENCE.",
      "- **R2 new / seen-term** -- new in v227->v230, term train-support >= S_HI (well-represented) = LK-like.",
      "- **R3 new / novel-term** -- new in v227->v230, term train-support < S_HI (rare/deep novel term) = the wall. Out-of-vocab novel terms are UNREACHABLE by accept-all (recall-ceiling loss).",
      ""]
# per-aspect degradation tables
L += ["## The 3-regime ranking decomposition (K=1, per aspect)", ""]
for asp in ["bpo", "mfo", "cco"]:
    a = R["aspects"].get(asp)
    if not a: continue
    L += [f"### {asp.upper()}  (vocab {a['n_vocab']}, targets {a['n_targets']}, S_HI={a['S_HI']})",
          "",
          "| regime | n_prot | AUROC (per-prot) | AUPR (per-prot) | Fmax | AUROC micro | rand AUROC |",
          "|---|---|---|---|---|---|---|"]
    for name, key in [("R1 known (memorization ref)", "R1_known_memorization"),
                      ("R2 new / seen-term", "R2_new_seen_term"),
                      ("R3 new / novel-term (wall)", "R3_new_novel_term")]:
        rg = a["regimes"][key]; k1 = rg["K1"]; rc = rg["random_control"]
        L.append(f"| {name} | {k1['n_prot']} | {f(k1.get('auroc_perprot'))} | {f(k1.get('aupr_perprot'))} | "
                 f"{f(k1['fmax']['fmax'])} | {f(k1.get('auroc_micro'))} | {f(rc.get('auroc_perprot'))} |")
    # degradation gap
    r1 = a["regimes"]["R1_known_memorization"]["K1"].get("auroc_perprot")
    r2 = a["regimes"]["R2_new_seen_term"]["K1"].get("auroc_perprot")
    r3 = a["regimes"]["R3_new_novel_term"]["K1"].get("auroc_perprot")
    if None not in (r1, r2, r3):
        L += ["", f"**Degradation (AUROC):** R1 {f(r1)} -> R2 {f(r2)} (Δ {f(r2-r1)}) -> R3 {f(r3)} (Δ {f(r3-r2)}). "
              f"The R1->R3 collapse of {f(r1-r3)} AUROC IS the generalization gap.", ""]
    # recall ceiling
    rcl = a["recall_ceiling"]; im = rcl["regime_ia_mass"]
    L += ["**Accept-all recall ceiling (IA-mass).** "
          f"reachable (in-vocab) = **{f(rcl['acceptall_reachable_frac'],3)}** of true IA-mass; "
          f"the rest is out-of-vocab novel terms (unreachable) -- a DATA/reachability property, not the model.",
          "",
          f"- regime IA-mass: R1 {f(im['R1_known'],0)} | R2 {f(im['R2_seen'],0)} | R3 in-vocab {f(im['R3_novel_invocab'],0)} | "
          f"R3 out-of-vocab UNREACHABLE {f(im['R3_novel_oov_UNREACHABLE'],0)}",
          f"- deployed pool reachability of NEW true IA-mass: PK {f(rcl.get('pool_reachability_PK_new'),3)} | LK {f(rcl.get('pool_reachability_LK_new'),3)}",
          ""]
# comparisons
vr = R.get("vs_reranker_PK_bpo_tail", {})
L += ["## Controls + honest comparisons", "",
      "### vs the deployed reranker ordering (PK-BPO reachable tail)",
      f"- n_proteins {vr.get('n_proteins')}; reranker per-protein AUC **{f(vr.get('reranker_auc_mean'))}** vs "
      f"single-model SSE AUC **{f(vr.get('sse_k1_auc_mean'))}** (SSE - reranker = {f(vr.get('sse_minus_reranker'))}).",
      f"- {vr.get('note','')}",
      "",
      "### K=1 (accept-all) vs K=5 MIN ensemble (same frozen rep)"]
for asp in ["bpo", "mfo", "cco"]:
    a = R["aspects"].get(asp)
    if not a: continue
    row = []
    for key in ["R1_known_memorization", "R2_new_seen_term", "R3_new_novel_term"]:
        k1 = a["regimes"][key]["K1"].get("auroc_perprot"); k5 = a["regimes"][key]["K5min"].get("auroc_perprot")
        row.append(f"{key.split('_')[0]} K1 {f(k1)}/K5 {f(k5)}")
    L.append(f"- {asp.upper()}: " + " | ".join(row))
# containment
cs = R.get("containment_semantics", {})
L += ["", "### Containment semantics on the frozen rep", "",
      "| aspect | term-axiom containment | random floor | frac fully-contained | frozen-rep parent>=child coverage |",
      "|---|---|---|---|---|"]
for asp in ["bpo", "mfo", "cco"]:
    c = cs.get(asp)
    if not c: continue
    L.append(f"| {asp.upper()} | {f(c.get('term_axiom_mean_containment'),3)} | {f(c.get('random_floor'),3)} | "
             f"{f(c.get('term_axiom_frac_full'),3)} | {f(c.get('frozen_hierarchy_monotone_frac'),3)} |")
L += ["", f"_{cs.get('caveat','')}_", ""]
# footnote
ft = R.get("acceptall_fmicrow_footnote", {})
L += ["### Footnote: accept-all calibrated f_micro_w (PK -known) -- confirms flood, NOT the verdict",
      f"- anchor reproduced: {ft.get('anchor_reproduced')}",
      f"- accept-all SSE f_micro_w: {ft.get('acceptall_sse_f_micro_w')}",
      f"- delta vs anchor: {ft.get('delta_vs_anchor')}",
      f"- {ft.get('note','')}", ""]
# verdict placeholder filled by hand below in receipt if needed
open(ROOT / "storage/regen_headline/SSE_KWTA_GEN.md", "w").write("\n".join(str(x) for x in L))
print("wrote SSE_KWTA_GEN.md")
