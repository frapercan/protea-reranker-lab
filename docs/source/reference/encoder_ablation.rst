protea_reranker_lab.encoder_ablation
====================================

Learned-encoder ablation study: over a single PLM's mean-pooled embeddings,
compare the raw dense representation against an unsupervised PCA projection and
a supervised, GO-aligned learned projection (``Linear(d -> dict)`` + top-k real,
trained so cosine matches Lin GO-similarity). Every arm is scored by the same
KNN GO-transfer plus cafaeval pipeline on the official LAFA frame
(IA-weighted ``f_micro_w``), so the dense-vs-PCA-vs-learned deltas are clean.

Run it with the thin entry ``scripts/run_encoder_ablation.py`` (defaults to
``esm2_150m``, the smallest PLM, for fast iteration). OBO/IA resolve through
:func:`protea_reranker_lab.band_registry_bridge.resolve_band_artifacts`; cafaeval
runs in the PROTEA venv via the shared
:func:`protea_reranker_lab.universal_runner._run_cafaeval` recipe. Set
``MLFLOW_TRACKING_URI`` to log the per-arm ``f_micro_w`` to the ``encoder-ablation``
experiment.

.. automodule:: protea_reranker_lab.encoder_ablation
   :members:
   :undoc-members:
   :show-inheritance:
