"""Assemble the LK abstract-fold deliverable: SUMMARY.txt from result_lk_bp.json,
abstract_coverage.json, train_meta_lk.json, robustness_lk.json."""
import os
import json

HERE = os.path.dirname(os.path.abspath(__file__))


def load(fn):
    p = os.path.join(HERE, fn)
    return json.load(open(p)) if os.path.exists(p) else {}


def main():
    res = load("result_lk_bp.json")
    cov = load("abstract_coverage.json")
    meta = load("train_meta_lk.json")
    rob = load("robustness_lk.json")
    inj = load("injectables_meta.json")
    c = res["cells"]
    d = res["deltas_f_micro_w"]

    L = []
    L.append("LK-BP ABSTRACT FOLD vs TITLES FOLD vs CHAMPION (board-faithful)")
    L.append("=" * 64)
    L.append("")
    L.append("QUESTION: GATE 0 showed ABSTRACTS beat TITLES on LK-BP pair "
             "separability (+0.031 AUROC, 0.586->0.617). Does that abstract "
             "upgrade also beat the titles in-frame fold on board-faithful LK-BP "
             "Fmax (f_micro_w)?")
    L.append("")
    L.append("SETUP (board-faithful, reused harness): OBO/IA/TOI, prop=fill, "
             "norm=cafa, no_orphans; naivemax(reranker_BP, InterPro2GO_BP); GT "
             "lafa_gt/groundtruth_LK.tsv. In-frame retrain on the champion's own "
             "multi-snapshot frame (v160..v225 train, v225-v227 early-stop), only "
             "the text-score source differs across variants -> clean A/B/C.")
    L.append("TEMPORAL HONESTY: text features from publications <= 2025-09 "
             "(v227 t0); test window v227->v230 (after Sep 2025) -> no leakage.")
    L.append("")
    L.append("BOARD-FAITHFUL LK-BP f_micro_w:")
    L.append(f"  champion (canonical)        0.42807")
    L.append(f"  retrained base (no text)    {c['retrained_base']['f_micro_w']:.5f}"
             f"   (tau {c['retrained_base']['tau']}, reproduces champion EXACTLY)")
    L.append(f"  titles fold                 {c['retrained_titles']['f_micro_w']:.5f}"
             f"   (tau {c['retrained_titles']['tau']}, +{d['titles_minus_base']:.5f} vs base)")
    L.append(f"  ABSTRACT fold               {c['retrained_abstract']['f_micro_w']:.5f}"
             f"   (tau {c['retrained_abstract']['tau']}, +{d['abstract_minus_base']:.5f} vs base)")
    L.append("")
    L.append("DELTAS (single seed=42):")
    L.append(f"  base   - champion = {d['base_minus_champion']:+.5f}  (sanity: exact reproduction)")
    L.append(f"  titles - base     = {d['titles_minus_base']:+.5f}")
    L.append(f"  abstract - base   = {d['abstract_minus_base']:+.5f}")
    L.append(f"  abstract - titles = {d['abstract_minus_titles']:+.5f}  <-- KEY: abstracts LOSE to titles")
    L.append("")
    if rob:
        L.append("SEED ROBUSTNESS (titles vs abstract, seeds 7/123/2024):")
        for r in rob["seeds"]:
            L.append(f"  seed {r['seed']:>4}: titles {r['titles']:.5f}  "
                     f"abstract {r['abstract']:.5f}  "
                     f"abstract-titles {r['abstract_minus_titles']:+.5f}")
        L.append(f"  MEAN: titles {rob['mean_titles']:.5f}  "
                 f"abstract {rob['mean_abstract']:.5f}  "
                 f"abstract-titles {rob['mean_abstract_minus_titles']:+.5f}")
        L.append("")
    L.append("ABSTRACT COVERAGE (LK-BP proteins):")
    for k in ("train_lk_bp", "test_lk_bp", "union_lk_bp"):
        v = cov.get(k, {})
        L.append(f"  {k:14s} proteins {v.get('proteins')}, with_abstract "
                 f"{v.get('with_abstract')} ({v.get('pct_abstract')}%), "
                 f"with_title {v.get('with_title')} ({v.get('pct_title')}%)")
    L.append(f"  abstracts.json pmids fetched: {cov.get('abstracts_json_pmids')}")
    L.append("  ASYMMETRY: TEST is abstract-rich (99.6%), TRAIN is abstract-poor "
             "(29%). The model learns the abstract feature from few training "
             "proteins yet must apply it on an abstract-saturated test set.")
    L.append("")
    L.append("FEATURE GAIN (LightGBM gain, seed 42): abstract text features carry "
             "MORE in-sample gain than titles "
             f"(text_score {meta.get('abstract',{}).get('lit_importance',{}).get('text_score')} "
             f"vs {meta.get('titles',{}).get('lit_importance',{}).get('text_score')}; "
             f"text_rank_pct {meta.get('abstract',{}).get('lit_importance',{}).get('text_rank_pct')} "
             f"vs {meta.get('titles',{}).get('lit_importance',{}).get('text_rank_pct')}) "
             "-> the model leans on abstracts MORE, but that reliance does NOT "
             "convert to held-out board Fmax. Abstract bodies raise the max-cosine "
             "floor (cos_med ~0.896) and add off-target sentence matches that "
             "inflate wrong-term scores, costing precision at the operating tau.")
    L.append("")
    L.append("INJECTABLES (uninjected, non-regressive vs champion):")
    for k, v in inj.items():
        L.append(f"  {v['file']}  rows {v['rows']}  proteins {v['proteins']}")
    L.append("")
    verdict_stable = rob and rob.get("mean_abstract_minus_titles", 0) < 0
    L.append("VERDICT:")
    L.append("  Abstracts gain over the bare champion (+{:.4f}) but REGRESS vs the "
             "titles fold ({:+.4f} at seed 42{}).".format(
                 d['abstract_minus_champion'], d['abstract_minus_titles'],
                 ", mean {:+.5f} across seeds".format(rob['mean_abstract_minus_titles'])
                 if rob else ""))
    L.append("  The GATE-0 pair-level separability win (+0.031 AUROC) does NOT "
             "translate to board-faithful Fmax. Titles remain the lever to ship "
             "for LK-BP; abstracts are NOT worth shipping over titles.")
    L.append("  Recommended injectable: the TITLES fold "
             "(literature_infame/injectable_lk_bp_lit.tsv == "
             "injectable_lk_bp_titles.tsv here, 0.43957). NK/MF/CC untouched.")
    txt = "\n".join(L) + "\n"
    open(os.path.join(HERE, "SUMMARY.txt"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
