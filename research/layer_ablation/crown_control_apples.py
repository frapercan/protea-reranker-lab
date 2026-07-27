"""The apples-to-apples control: train the head on 100,000 proteins, score it on the
SAME 15,000-protein reference the served champion was scored on.

Why this exists. The crown trained every arm on the 15,000 reference that sits on disk,
and its L48 control reached mean9 f_micro_w 0.1425 against the champion's 0.2150. The
champion's own metadata declares `reference_n: 100000`, so the control was starved 6.7x.

Rerunning the lab harness at ref_n=100000 does not settle it, because in that harness
`ref_n` sets BOTH the encoder's training pool AND the kNN reference used for GO transfer
(`_load_data`), so its 0.1961 is measured against a different, larger reference and its
per-cell numbers are not comparable to the champion's (it beats the champion on all three
PK cells and collapses on CCO: that is the reference moving, not the encoder).

Here the two are separated. The encoder trains on the 100k pool pulled from the database,
exactly as `_train_encoder` does, and then encodes the crown's own 7,401 queries and
15,000 reference proteins. Scoring is the crown's protocol, unchanged: cosine top-30 GO
vote into the 15k reference, cafaeval f_micro_w over nine cells.

If it lands near 0.2150, the training-pool explanation is proven and the crown's arm
ordering stands as a controlled contrast on a starved substrate. If it does not, the
recipe differs in some other way and every crown delta needs re-reading. Both answers are
results; neither is assumed.

Read-only against the database.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import psycopg2

sys.path.insert(0, "/home/frapercan/Thesis2/repositories/protea-reranker-lab/src")
from protea_reranker_lab.encoder_ablation import (  # noqa: E402
    ArmSpec, EncoderAblationSpec, GoDag, _load_data, _train_encoder, l2n, pull_mean,
)
from cafaeval.evaluation import cafa_eval  # noqa: E402

W = Path("/home/frapercan/Thesis2/storage/layer_ablation")
REL = Path("/home/frapercan/Thesis2/CAFA_forever/data/releases/Sep_2025_Mar_2026")
OBO = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/go-basic.obo"
IA = "/home/frapercan/Thesis2/protea-lafa-knn/lafa_t0_Sep_2025/IA.tsv"
TOI = str(REL / "groundtruth_terms_of_interest.txt")
NS2A = {"biological_process": "bpo", "molecular_function": "mfo", "cellular_component": "cco"}
CATS = {"NK": (str(REL / "groundtruth_NK.tsv"), None),
        "LK": (str(REL / "groundtruth_LK.tsv"), None),
        "PK": (str(REL / "groundtruth_PK.tsv"), str(REL / "groundtruth_PK_known.tsv"))}
KNN = 30
CHAMPION = {"nk-bpo": 0.16646, "nk-cco": 0.31222, "nk-mfo": 0.34481,
            "lk-bpo": 0.21742, "lk-cco": 0.31982, "lk-mfo": 0.27577,
            "pk-bpo": 0.06447, "pk-cco": 0.13840, "pk-mfo": 0.09567}


def knn_cells(Qc: np.ndarray, Rc: np.ndarray, qaccs: list[str], ref_go: list[list[str]]) -> dict:
    """The crown's own scoring protocol, unchanged."""
    d = tempfile.mkdtemp()
    try:
        Qn, Rn = l2n(Qc), l2n(Rc)
        with open(os.path.join(d, "p.tsv"), "w") as w:
            for i in range(0, len(qaccs), 500):
                sims = Qn[i:i + 500] @ Rn.T
                for r, srow in enumerate(sims):
                    idx = np.argpartition(-srow, KNN)[:KNN]
                    votes: dict[str, float] = defaultdict(float)
                    for n in idx:
                        s = float(srow[n])
                        if s <= 0:
                            continue
                        for go in ref_go[n]:
                            votes[go] += s
                    if votes:
                        mx = max(votes.values())
                        for go, v in votes.items():
                            w.write(f"{qaccs[i + r]}\t{go}\t{v / mx:.6f}\n")
        cells = {}
        for cat, (gt, known) in CATS.items():
            _, best = cafa_eval(OBO, d, gt, ia=IA, no_orphans=True, norm="cafa", prop="fill",
                                exclude=known, toi_file=TOI, th_step=0.01, n_cpu=8)
            for _, row in best["f_micro_w"].reset_index().iterrows():
                a = NS2A.get(row["ns"])
                if a:
                    cells[f"{cat.lower()}-{a}"] = round(float(row["f_micro_w"]), 5)
        return cells
    finally:
        shutil.rmtree(d, ignore_errors=True)


def main() -> None:
    qry_accs = json.load(open(W / "emb_ankh_base/meta.json"))["accs"]
    ref_accs = json.load(open(W / "ref_emb/meta.json"))["accs"]
    ref_go_map = json.load(open(W / "ref_go.json"))
    assert not (set(qry_accs) & set(ref_accs)), "query and reference overlap: self-retrieval leakage"

    spec = EncoderAblationSpec(
        name="crown-control-apples",
        embedding_config_id="08234f06-ba76-4d7d-aaec-ae601096b4fa",  # ankh-base, last layer
        band="v227", ref_n=100_000, knn=KNN, epochs=150, train_pairs=300_000, seed=42,
    )
    dag = GoDag.from_obo(Path(OBO))
    rng = np.random.default_rng(spec.seed)

    # The training pool. _load_data excludes the queries it is given, so hand it the union of
    # both evaluation sets: neither the queries nor the 15k reference may train the encoder.
    banned = sorted(set(qry_accs) | set(ref_accs))
    R, _Q_unused, ref_clo, _ = _load_data(spec, dag, banned, rng)
    print(f"  training pool: {len(ref_clo):,} annotated proteins, dim {R.shape[1]}", flush=True)
    assert len(ref_clo) > 50_000, f"training pool collapsed to {len(ref_clo)}"

    # Pull the two evaluation sets from the same source, so nothing but the pool size moves.
    conn = psycopg2.connect(spec.dsn)
    cur = conn.cursor()
    q_emb = pull_mean(cur, qry_accs, spec.embedding_config_id)
    r_emb = pull_mean(cur, ref_accs, spec.embedding_config_id)
    cur.close()
    conn.close()
    q_keep = [a for a in qry_accs if a in q_emb]
    r_keep = [a for a in ref_accs if a in r_emb]
    print(f"  queries with embeddings: {len(q_keep):,}/{len(qry_accs):,} | "
          f"reference: {len(r_keep):,}/{len(ref_accs):,}", flush=True)

    # _train_encoder encodes whatever it is handed as Q, so hand it both sets at once.
    Q = np.vstack([np.vstack([q_emb[a] for a in q_keep]),
                   np.vstack([r_emb[a] for a in r_keep])]).astype(np.float32)
    arm = ArmSpec(name="learned-k128-hardneg", kind="learned", dict_dim=2048, top_k=128,
                  objective="hard-neg")
    _Rc, Qc = _train_encoder(R, Q, ref_clo, dag, arm, spec)
    Qcodes, Rcodes = Qc[:len(q_keep)], Qc[len(q_keep):]
    print(f"  codes: query {Qcodes.shape}  reference {Rcodes.shape}", flush=True)

    cells = knn_cells(Qcodes, Rcodes, q_keep, [ref_go_map.get(a, []) for a in r_keep])
    mean9 = sum(cells.values()) / 9
    champ9 = sum(CHAMPION.values()) / 9
    crown15k = json.load(open(W / "crown_result.json"))["mean9_f_micro_w"]["L48"]

    out = {
        "question": "trained on 100k, scored on the champion's own 15k reference: does the "
                    "control reproduce the served champion?",
        "config": {"ref_n": spec.ref_n, "trained_on": len(ref_clo), "epochs": spec.epochs,
                   "train_pairs": spec.train_pairs, "seed": spec.seed, "knn": KNN,
                   "base": "08234f06 ankh-base last layer mean pool", "objective": "hard-neg"},
        "cells": cells, "mean9": round(mean9, 5),
        "champion_cells": CHAMPION, "champion_mean9": round(champ9, 5),
        "crown_control_15k_mean9": crown15k,
        "gap_to_champion": round(champ9 - mean9, 5),
        "fraction_of_gap_closed": round((mean9 - crown15k) / (champ9 - crown15k), 4),
        "per_cell_gap_to_champion": {k: round(CHAMPION[k] - cells[k], 5) for k in cells},
    }
    (W / "crown_control_apples_result.json").write_text(json.dumps(out, indent=2))
    print("\n" + json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
