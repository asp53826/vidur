#!/bin/bash
# One-time environment setup on Sapelo2. Run on a login node:
#   bash validation/slurm/setup_env.sh
#
# Creates a virtualenv with vLLM (for the GPU runs) and Vidur's own
# requirements (for the simulator, which runs on CPU).
set -euo pipefail

# Pick a Python 3.10-3.12 module: list them with `ml spider Python`, then
# export PYTHON_MODULE=<name> before running this script.
if [[ -n "${PYTHON_MODULE:-}" ]]; then
  ml "$PYTHON_MODULE"
fi

VENV=${VENV:-/scratch/${USER:-$(id -un)}/venvs/vidur-spec}
REPO=$(cd "$(dirname "$0")/../.." && pwd)

python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip
# leave vLLM unpinned so you get a current release, but record what you got
python -m pip install vllm aiohttp
python -m pip install -r "$REPO/requirements.txt"
python -m pip install pytest

python -c "import vllm; print('vllm', vllm.__version__)" | tee "$REPO/validation/vllm_version.txt"
echo
echo "Done. Before submitting jobs:"
echo "  1. Accept the Llama 2 license on Hugging Face for meta-llama/Llama-2-7b-hf"
echo "     (and Llama-2-70b-hf for e2), then: export HF_TOKEN=..."
echo "  2. export HF_HOME=/scratch/${USER:-$(id -un)}/hf   # model weights are large"
