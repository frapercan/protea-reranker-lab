GPU torch for the lab experiments
=================================

The lab's default install is **CPU torch**. ``pyproject.toml`` pins torch
to the ``pytorch-cpu`` source on purpose: CI has no GPU and the core
LightGBM reranker training is CPU bound, so the default environment stays
light. The research experiments that use torch (the SDR encoders, learned
poolings, and contrastive trainings) are the exception: they want the GPU.

Why this runbook exists
-----------------------

The lab venv is the only environment that carries the experiment
dependencies (``psycopg2`` for the read-only DB pull, ``scipy``, and the
``protea_reranker_lab.sdr`` package). The deploy venv has CUDA torch but
not those dependencies, so torch trainings cannot simply run there. The
result, if nothing is done, is that every SDR training runs on the CPU.

Flip torch to GPU
-----------------

Run the override after every ``poetry install`` or ``poetry update``:

.. code-block:: bash

   bash scripts/install_gpu_torch.sh            # default cu128, this repo's venv

It reinstalls ``torch`` and ``torchvision`` from the CUDA wheel index,
pulling the ``nvidia-*`` runtime packages torch needs. Confirm it worked:

.. code-block:: bash

   .venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
   # expect: 2.x.y+cu128 True

Hard rule
---------

Never recover with ``poetry install --sync``. The ``--sync`` flag forces
the lock and wipes the CUDA torch this script installs, dropping every
experiment back to the CPU. Use plain ``poetry install`` then re-run
``scripts/install_gpu_torch.sh``. The default cu128 build matches NVIDIA
driver 570 and 580 series; override with ``CUDA_VARIANT`` for others.
