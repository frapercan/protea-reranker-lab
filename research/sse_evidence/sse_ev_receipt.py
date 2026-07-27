"""Assemble the SSE-evidence receipt: regen_headline/SSE_EVIDENCE.{md,json}."""
import json
from pathlib import Path
ROOT = Path("/home/frapercan/Thesis2")
OUT = ROOT / "storage/sse_evidence"; RH = ROOT / "storage/regen_headline"
def rd(p, d=None):
    try: return json.load(open(p))
    except Exception: return d if d is not None else {}
build = rd(OUT / "build_report.json")
meas = rd(OUT / "measure.json")
wall = rd(OUT / "wall_by_evidence.json")
trains = {a: rd(OUT / f"train_{a}.json") for a in ["full", "noiea", "exp"]}

J = {"corpus_frame": {
        "corpus": "cooc_experiment/generator_frames raw_esm2_3b (ESM2-3B mean, 2560-d)",
        "why": "the exact frame the base full-corpus SSE used; fixed vocab = base per-aspect SSE meta terms "
               "(mfo 808 / bpo 3970 / cco 653); rows = base eligibility (eval LK+PK proteins held out of all training)",
        "K": 1, "temporal_gate": "train<=v225, blind eval v227->v230, eval proteins excluded from training",
        "evidence_source": "storage/gaf_cache/goa_uniprot_all.gaf.225.gz (v225=train cutoff), streamed, "
                           "restricted to the 88,212 corpus accessions"},
     "build_report": build, "train": trains, "measure": meas, "wall_by_evidence": wall}
json.dump(J, open(RH / "SSE_EVIDENCE.json", "w"), indent=1, default=float)

def g(d, *ks, default=None):
    for k in ks:
        if not isinstance(d, dict): return default
        d = d.get(k, {})
    return d if d != {} else default

L = []
L.append("# Evidence-aware Sparse Semantic-Entailment (SSE) + the BP wall characterised BY evidence\n")
L.append("Frozen v225 evidence (`goa_uniprot_all.gaf.225.gz`), leakage-safe (pre-cutoff = legitimate INPUT; "
         "test-side v227->v230 evidence NEVER used as input). Corpus = `generator_frames raw_esm2_3b` "
         "(ESM2-3B mean, 2560-d); fixed vocab = base SSE meta terms; K=1; eval proteins held out. Compute "
         "`repositories/PROTEA/.venv`.\n")

L.append("## Evidence tiers\n")
L.append("EXP {EXP,IDA,IPI,IMP,IGI,IEP,HTP,HDA,HMP,HGI,HEP} / PHY {IBA,IBD,IKR,IRD} / "
         "COMP {ISS,ISO,ISA,ISM,IGC,RCA} / AUTH {TAS,NAS,IC} / ELEC {IEA}. "
         "de-circularisation arms are SUBSETS of the exact base positive set: **noiea** = A-positive with any "
         "non-electronic tier support; **exp** = A-positive with experimental (EXP) tier support.\n")

L.append("## (a) De-circularised training corpus -- geometry + separability\n")
L.append("| aspect | A-positives | GAF-supported | noiea (drop IEA) | exp-only | full valAUC | noiea valAUC | exp valAUC |")
L.append("|---|--:|--:|--:|--:|--:|--:|--:|")
for asp in ["mfo","bpo","cco"]:
    b = g(build,"aspects",asp,default={})
    vf = g(trains,"full","aspects",asp,"val_auc"); vn = g(trains,"noiea","aspects",asp,"val_auc"); ve = g(trains,"exp","aspects",asp,"val_auc")
    L.append(f"| {asp} | {b.get('A_positives','?')} | {b.get('gaf_support_frac','?')} | "
             f"{b.get('noiea_frac_of_full','?')} | {b.get('exp_frac_of_full','?')} | "
             f"{round(vf,4) if vf else '?'} | {round(vn,4) if vn else '?'} | {round(ve,4) if ve else '?'} |")
L.append("")

L.append("### Ranking metric (per-protein separability AUC over deployed pool candidates; SSE vs deployed reranker)\n")
L.append("| cell | arm | SSE per-prot AUC | reranker AUC | SSE pooled AUC | n_pos |")
L.append("|---|---|--:|--:|--:|--:|")
for cell in ["pk","lk"]:
    for asp in ["bpo","mfo","cco"]:
        key = f"{cell}-{asp}"
        for arm in ["full","noiea","exp"]:
            r = g(meas,"arms",arm,key,default=None)
            if not r: continue
            L.append(f"| {key} | {arm} | {r.get('sse_perprot_auc')} | {r.get('rr_perprot_auc')} | "
                     f"{r.get('sse_pooled_auc')} | {r.get('n_pos')} |")
L.append("")

L.append("### Calibrated wall check -- SSE as pool-reranker f_micro_w (true frame, PK -known) vs deployed anchor\n")
L.append("| cell | arm | SSE pool-reranker f | deployed anchor |")
L.append("|---|---|--:|--:|")
for cell,asp in [("pk","bpo"),("lk","bpo")]:
    key=f"{cell}-{asp}"
    for arm in ["full","noiea","exp"]:
        w = g(meas,"arms",arm,"calibrated_wall",key,default=None)
        if not w: continue
        L.append(f"| {key} | {arm} | {w.get('sse_poolreranker_f')} | {w.get('deployed_anchor_ref')} |")
L.append(f"\nDeployed anchor reproduced as pool-reranker (deployed scores over the same pool): "
         f"{g(meas,'anchor_poolreranker',default={})}\n")

L.append("## (b) Per-protein t0 characterisation-quality calibration prior (on full-arm SSE)\n")
L.append("Prior = v225 KNOWN evidence summary per protein [exp_frac, nonelec_frac, log-count, log-breadth, has_exp]. "
         "Per-protein AUC is invariant to a per-protein prior, so this is measured where a prior CAN move the metric: "
         "cross-protein POOLED AUC + calibrated f_micro_w.\n")
L.append("| cell | pooled AUC SSE | +prior | delta AUC | f SSE-cal | f +prior | delta f |")
L.append("|---|--:|--:|--:|--:|--:|--:|")
for cell,asp in [("pk","bpo"),("lk","bpo")]:
    r = g(meas,"arm_b_calibration",f"{cell}-{asp}",default=None)
    if not r: continue
    L.append(f"| {cell}-{asp} | {r.get('pooled_auc_sse_only')} | {r.get('pooled_auc_sse_plus_prior')} | "
             f"{r.get('delta_pooled_auc')} | {r.get('f_sse_only_calibrated')} | {r.get('f_sse_plus_prior_calibrated')} | {r.get('delta_f')} |")
L.append("")

L.append("## (c) Evidence-typed containment (term v225 experimental-fraction scaling of coverage, full arm)\n")
L.append("| cell | per-prot AUC raw | evtyped | delta | pooled raw | evtyped | pool-reranker f |")
L.append("|---|--:|--:|--:|--:|--:|--:|")
for cell,asp in [("pk","bpo"),("lk","bpo")]:
    r = g(meas,"arm_c_evtyped",f"{cell}-{asp}",default=None)
    if not r: continue
    L.append(f"| {cell}-{asp} | {r.get('sse_perprot_auc')} | {r.get('sse_c_perprot_auc')} | {r.get('delta_perprot_auc')} | "
             f"{r.get('sse_pooled_auc')} | {r.get('sse_c_pooled_auc')} | {r.get('sse_c_poolreranker_f')} |")
L.append("")

L.append("## The BP wall characterised BY evidence (analysis-only)\n")
L.append("**Data caveat:** the v227->v230 annotating evidence of each true term is NOT on disk (only the v225 GAF). "
         "Term-level (1) + per-protein (3) are v225 corpus proxies; donor (2) is the KNN neighbour's pre-cutoff "
         "evidence (provenance-confirmed, NOT the annotation's own). eval.parquet `evidence_code` = donor, not label.\n")
for cell in ["pk","lk"]:
    c = g(wall,"cells",f"{cell}-bpo",default=None)
    if not c: continue
    L.append(f"### {cell.upper()}-BPO  (targets {c.get('n_targets_scored')}, with-unreachable {c.get('n_with_unreachable')})\n")
    im = c.get("IA_mass",{})
    L.append(f"- IA mass: true {im.get('true')}, missing {im.get('missing')}, unreachable {im.get('unreach')}")
    tc = c.get("term_class_IA_frac",{})
    L.append(f"- **unreachable** term-class (IA-frac): {tc.get('unreach')}")
    L.append(f"- **missing** term-class (IA-frac): {tc.get('missing')}")
    L.append(f"- unreachable IA-weighted mean exp-frac {c.get('unreachable_IAweighted_mean_exp_frac')} "
             f"/ elec-frac {c.get('unreachable_IAweighted_mean_elec_frac')}")
    L.append(f"- missing IA-weighted mean exp-frac {c.get('missing_IAweighted_mean_exp_frac')} "
             f"/ elec-frac {c.get('missing_IAweighted_mean_elec_frac')}")
    L.append(f"- donor evidence of REACHABLE true terms (IA-frac): {c.get('donor_evidence_reachable_IA_frac')}")
    pp = c.get("perprotein_t0_exp_frac_mean",{})
    L.append(f"- per-protein t0 exp-frac: unreachable-carrying {pp.get('unreachable_carrying')} "
             f"vs fully-reachable {pp.get('fully_reachable')}\n")

L.append("## Verdict\n")
L.append("**Can encoding evidence help? No encoding converts on the metric; only the per-protein calibration "
         "prior (b) moves any cross-protein signal, and it does not convert.** "
         "(a) De-circularisation is a NO-GO and a null result at the geometry level: at the propagated-label "
         "level only ~2% of BP positives are IEA-only and ~8% are non-experimental, so dropping IEA (noiea) or "
         "keeping only experimental support (exp) leaves val-AUC essentially unchanged (bpo 0.937/0.935/0.924) "
         "and only REMOVES separability (pk-bpo pooled AUC 0.613->0.601->0.574) and calibrated f_micro_w "
         "(0.084->0.080->0.070). The learned entailment geometry is NOT materially driven by circular IEA "
         "co-annotation, and de-circularising it only deletes signal. "
         "(b) The per-protein t0 characterisation-quality prior is the ONLY evidence encoding that moves a "
         "cross-protein metric: pk-bpo pooled AUC 0.613->0.661 (+0.048). But it does NOT convert -- calibrated "
         "f_micro_w falls 0.077->0.063 (-0.015). This is the crux the campaign keeps hitting: a +0.048 "
         "cross-protein-ranking gain that the IA-weighted micro-F does not reward, because the metric rewards "
         "cross-protein CALIBRATION under accept-all flooding, not ranking. "
         "(c) Evidence-typed containment (scaling coverage by term v225 experimental-fraction) collapses "
         "separability to chance (pk-bpo per-prot AUC 0.611->0.491; lk-bpo 0.838->0.452): experimentally-typed "
         "terms are not the high-recall ones, so the scaling deletes the signal. All three arms leave SSE far "
         "below the deployed anchor on f_micro_w (pk-bpo SSE 0.084 vs anchor-pool 0.118 / full anchor 0.144).\n")
L.append("**What the unreachable tail's evidence composition says about the frontier:** the ~45% of PK-BPO true "
         "IA mass that no t0 signal reaches is EXPERIMENTAL-type biology, not a pipeline-reachable IEA/ISS slice. "
         "By v225 term-profile the unreachable mass is 43% experimental-dominant terms + 24% mixed + 23% "
         "absent-in-v225, and only ~9% IEA-dominant (IA-weighted mean experimental-fraction 0.51 vs electronic "
         "0.24); LK-BPO is the same direction (exp-frac 0.46). Decisively, the proteins carrying unreachable mass "
         "are BETTER experimentally characterised at t0 than the fully-reachable ones (t0 exp-frac 0.465 vs 0.375 "
         "PK; 0.29 vs 0.21 LK), so this is NOT a poorly-annotated-protein artefact. The frontier is genuinely new "
         "experimental annotation on already well-studied proteins, with no electronic/homology footprint at t0 -- "
         "which is exactly why encoding, de-circularising, or reweighting the t0 evidence cannot break the wall. "
         "The small IEA-dominant slice (~9%) is the only 'pipeline-reachable' remainder and is far too small to "
         "move the cell. (Caveat: v227->v230 annotating evidence is not on disk; this uses v225 corpus proxies "
         "and provenance-confirmed donor evidence, stated above.)\n")
open(RH / "SSE_EVIDENCE.md","w").write("\n".join(L))
print("WROTE", RH / "SSE_EVIDENCE.md", "and SSE_EVIDENCE.json")
print("\n".join(L))
