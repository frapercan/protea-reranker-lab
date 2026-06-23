#!/usr/bin/env bash
# Override the default CPU torch installed by ``poetry install`` with the
# CUDA build, so the lab's torch-based experiments (SDR encoders, learned
# poolings, contrastive trainings) run on the GPU instead of the CPU.
#
# ``pyproject.toml`` pins torch to the ``pytorch-cpu`` source on purpose:
# GitHub CI runners and the slim images have no GPU, and the lab's core
# job (LightGBM reranker training) is CPU-bound, so the default install
# must stay light. Hosts that run the GPU experiments need to flip torch
# back to the CUDA wheel after each ``poetry install`` / ``poetry update``.
#
# Usage:
#   bash scripts/install_gpu_torch.sh                 # default cu128, lab venv
#   CUDA_VARIANT=cu121 bash scripts/install_gpu_torch.sh
#   VENV_PATH=/path/to/.venv bash scripts/install_gpu_torch.sh
#
# Default cu128 is compatible with NVIDIA driver 570 and 580 series; older
# variants stay reachable via the CUDA_VARIANT override above.
#
# Why no ``--no-deps``: torch on Linux ships its CUDA runtime through the
# ``nvidia-*`` PyPI packages (nvidia-cudnn-cu12, nvidia-cublas-cu12, ...).
# Without those, ``import torch`` fails at load time with
# ``ImportError: libcudnn.so.9: cannot open shared object file``. Letting
# pip resolve the runtime deps is the correct fix; unrelated bumps (sympy,
# networkx, MarkupSafe) stay inside the permissive ranges in pyproject.
#
# NEVER use ``poetry install --sync`` to recover: it forces the lock and
# wipes the CUDA torch this script installs. Use plain ``poetry install``
# then re-run this script.
set -euo pipefail

CUDA_VARIANT="${CUDA_VARIANT:-cu128}"
INDEX_URL="https://download.pytorch.org/whl/${CUDA_VARIANT}"

# Resolve the target venv. Default to this repo's poetry env; allow an
# explicit VENV_PATH for smoke tests or a sibling checkout.
VENV_PATH="${VENV_PATH:-}"
if [[ -z "${VENV_PATH}" ]]; then
    if ! command -v poetry >/dev/null 2>&1; then
        echo "poetry is required on PATH (or set VENV_PATH)" >&2
        exit 1
    fi
    VENV_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && poetry env info --path 2>/dev/null || true)"
fi

if [[ -z "${VENV_PATH}" || ! -x "${VENV_PATH}/bin/pip" ]]; then
    echo "no usable virtualenv at '${VENV_PATH}' (run 'poetry install' or set VENV_PATH)" >&2
    exit 1
fi

echo ">>> overriding torch + torchvision with ${CUDA_VARIANT} wheels (with deps) into ${VENV_PATH}"
"${VENV_PATH}/bin/pip" install --upgrade --force-reinstall \
    --index-url "${INDEX_URL}" \
    torch torchvision

echo ">>> installed:"
"${VENV_PATH}/bin/pip" list --format=columns \
    | grep -Ei '^(torch|torchvision|triton|nvidia-)' \
    || true

echo ">>> import torch self-check:"
"${VENV_PATH}/bin/python" - <<'PY'
import torch
print(f"torch {torch.__version__} cuda_available={torch.cuda.is_available()}")
PY
