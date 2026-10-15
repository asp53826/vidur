#!/bin/bash
# Serve one experiment configuration with vLLM and replay workloads against it.
#
#   sbatch validation/slurm/run_experiment.sbatch EXPERIMENT_JSON WORKLOAD [WORKLOAD ...]
#
# For tensor-parallel experiments ask for more GPUs on the command line,
# e.g. sbatch --gres=gpu:A100:4 ... for e2. Results land in
# validation/results/<experiment name>/<workload name>/.
#
#SBATCH --job-name=vidur-spec-val
#SBATCH --partition=gpu_p
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=validation/logs/%x-%j.out

set -euo pipefail

EXPERIMENT=$1; shift
WORKLOADS=("$@")
REPO=${SLURM_SUBMIT_DIR:-$(pwd)}
cd "$REPO"

VENV=${VENV:-/scratch/${USER:-$(id -un)}/venvs/vidur-spec}
if [[ -n "${PYTHON_MODULE:-}" ]]; then ml "$PYTHON_MODULE"; fi
# shellcheck disable=SC1091
if [[ -f "$VENV/bin/activate" ]]; then source "$VENV/bin/activate"; fi
export HF_HOME=${HF_HOME:-/scratch/${USER:-$(id -un)}/hf}

NAME=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['name'])" "$EXPERIMENT")
MODEL=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['model'])" "$EXPERIMENT")
OUT=validation/results/$NAME
mkdir -p "$OUT" validation/logs

PORT=$((8000 + RANDOM % 1000))
SERVE_CMD=$(python validation/launch_vllm.py "$EXPERIMENT" "$PORT")
if [[ -n "${FAKE_VLLM:-}" ]]; then
  # dry run of this script without a GPU, against the harness's fake server
  SERVE_CMD="python validation/tests/fake_vllm_server.py --port $PORT"
fi

{
  echo "experiment: $EXPERIMENT"
  echo "host: $(hostname)  job: ${SLURM_JOB_ID:-local}"
  nvidia-smi --query-gpu=name,memory.total,driver_version,power.limit --format=csv || true
  python -c "import vllm; print('vllm', vllm.__version__)" || echo "vllm not importable"
  echo "serve: $SERVE_CMD"
} | tee "$OUT/environment.txt"

eval "$SERVE_CMD" > "$OUT/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true; wait $SERVER_PID 2>/dev/null || true' EXIT

echo "waiting for vLLM on port $PORT"
for _ in $(seq 1 "${STARTUP_POLLS:-180}"); do
  if curl -sf "http://localhost:$PORT/health" > /dev/null; then break; fi
  if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "vLLM exited during startup; tail of server.log:"; tail -40 "$OUT/server.log"; exit 1
  fi
  sleep "${STARTUP_POLL_S:-10}"
done
curl -sf "http://localhost:$PORT/health" > /dev/null || { echo "vLLM never became healthy"; exit 1; }

# one short throwaway run so CUDA graph capture and warmup don't land in the data
python validation/make_workload.py synthetic --qps 4 --num-requests 16 --seed 999 \
  --out "$OUT/warmup.jsonl" > /dev/null
python validation/replay.py --workload "$OUT/warmup.jsonl" --out-dir "$OUT/_warmup" \
  --base-url "http://localhost:$PORT" --model "$MODEL" > /dev/null

for W in "${WORKLOADS[@]}"; do
  RUN="$OUT/$(basename "$W" .jsonl)"
  echo "replaying $W -> $RUN"
  python validation/replay.py --workload "$W" --out-dir "$RUN" \
    --base-url "http://localhost:$PORT" --model "$MODEL"
  # vLLM idles between runs; give the queue a moment to drain fully
  sleep 5
done

echo "done: $OUT"
