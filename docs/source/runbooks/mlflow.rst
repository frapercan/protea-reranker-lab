MLflow tracking for lab experiments
===================================

The lab logs its tracked experiments (native-booster training, the SDR-A
correlation study, and any future runner that opts in) to a self-hosted MLflow
server. This runbook covers where the server lives, how to start and stop it, the
environment variables a run needs, the experiment-naming convention, and how a run
shows up in the UI.

Where the server lives
~~~~~~~~~~~~~~~~~~~~~~~~

The MLflow deployment is not part of this repository. It is a light, supervised
service in the Thesis2 root that reuses PROTEA's existing Postgres and MinIO (fully
isolated from the PROTEA application DB and bucket):

``~/Thesis2/storage/mlflow/``

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Artefact
     - Location
   * - Tracking URI / UI
     - ``http://127.0.0.1:5000``
   * - Backend store (params/metrics)
     - Postgres DB ``mlflow`` in ``protea-postgres-1``
   * - Artifact store (files/plots)
     - MinIO bucket ``s3://mlflow/`` at ``http://localhost:9000``
   * - Start script
     - ``~/Thesis2/storage/mlflow/start-server.sh``
   * - Stop script
     - ``~/Thesis2/storage/mlflow/stop-server.sh``
   * - Logs
     - ``~/Thesis2/storage/mlflow/logs/mlflow-server.log``

Start, stop, health
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   bash ~/Thesis2/storage/mlflow/start-server.sh   # backgrounds, writes mlflow-server.pid
   bash ~/Thesis2/storage/mlflow/stop-server.sh     # stops it

   curl -s http://127.0.0.1:5000/health             # OK
   curl -s http://127.0.0.1:5000/version            # 3.14.0

The server is RAM-conscious (single uvicorn worker, async job runner disabled);
roughly 450 MB RSS, safe to run alongside training.

Required environment variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A run logs to MLflow only when ``MLFLOW_TRACKING_URI`` is set. Artifact upload
(plots, summary tables) additionally needs the S3 endpoint and credentials, since
artifacts go to the MinIO bucket:

.. code-block:: bash

   export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
   export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
   # the lab runners default these to minioadmin/minioadmin if unset:
   export AWS_ACCESS_KEY_ID=minioadmin
   export AWS_SECRET_ACCESS_KEY=minioadmin

.. note::

   ``mlflow-skinny`` (installed via the ``native`` extra) can hit a protobuf
   version clash on import. If you see a ``Descriptors cannot be created
   directly`` error, set ``PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`` in the
   environment before running. Artifact upload also requires ``boto3`` (shipped in
   the ``native`` extra); without it params and metrics still log but file upload
   fails with a ``No module named 'boto3'`` error.

All logging in the lab is best-effort and env-gated: if ``MLFLOW_TRACKING_URI``
is unset, unreachable, or errors, the run proceeds unchanged. MLflow can never break
an experiment.

Experiment naming
~~~~~~~~~~~~~~~~~

One experiment per study, named for what it measures. Current experiments:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Experiment
     - What
   * - ``native-reranker-boosters``
     - per-category LightGBM booster training
   * - ``sdr-a-correlation``
     - SDR-A overlap-vs-GO-semantic correlation (T-CIENCIA)

A new study picks a new lowercase, hyphenated name describing the question, not a
version token.

How a run shows up
~~~~~~~~~~~~~~~~~~

In the UI (``http://127.0.0.1:5000``) the hierarchy is **experiment, then run, then
params / metrics / artifacts**:

- **Params**: the run configuration (for SDR-A: ``plm``, ``K``, ``window``,
  ``n_proteins``, ``n_pairs``, ``kwta_k``, ``semantic_metric``, ``seed``).
- **Metrics**: the scalar readouts (for SDR-A: ``spearman_cosine_go`` and
  ``spearman_tanimoto_go`` logged with a ``step`` equal to the k-WTA ``k``, plus a
  flat ``spearman_tanimoto_go_k<k>`` per k and ``best_tanimoto_rho``). The booster
  job streams per-iteration eval AUC the same way.
- **Artifacts**: the summary table (CSV), the JSON result, and the scatter plot,
  uploaded to the MinIO ``mlflow`` bucket and browsable from the run page.
- **Tags**: free-form context such as ``slice`` and ``verdict``.

Running the SDR-A study
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
   export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
   export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
   poetry run python scripts/run_sdr_a_correlation.py

The runner defaults the bundle and OBO paths to the frozen v227 reference pool and
the t0 ontology snapshot, so no flags are needed. It logs to experiment
``sdr-a-correlation``. The ``no-mlflow`` flag writes the local artifacts only, and
the ``semantic-metric lin`` flag swaps the Lin metric for the default Resnik one
(both flags take the usual leading double dash; ``run_sdr_a_correlation.py -h``
lists the full set).
