#!/bin/bash
# CPU-only: build Vidur inputs from every measured run, simulate, and score.
# Run from the repository root (an interactive CPU job is plenty):
#
#   bash validation/slurm/simulate_and_report.sh validation/experiments/e1_llama2_7b_a100_ngram_k4.json
#
# The first Vidur run trains the execution-time models (~15 min on 2 cores);
# later runs reuse ./cache.
set -euo pipefail

EXPERIMENT=$1
WEIGHTS=${WEIGHTS:-"0.25 0.5 0.75"}
NAME=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['name'])" "$EXPERIMENT")
export WANDB_MODE=disabled

for RUN in "validation/results/$NAME"/*/; do
  [[ -f "$RUN/measured.csv" ]] || continue
  [[ "$(basename "$RUN")" == _warmup ]] && continue
  echo "== $RUN"
  python validation/build_sim_inputs.py --experiment "$EXPERIMENT" --run-dir "$RUN" --out-dir "$RUN/sim_inputs"
  # shellcheck disable=SC2086
  python validation/run_sim.py --sim-inputs "$RUN/sim_inputs" --out-dir "$RUN/sim" --cache-dir cache --weights $WEIGHTS
  python validation/report.py --run-dir "$RUN" --sim-inputs "$RUN/sim_inputs" --sim-dir "$RUN/sim" --out-dir "$RUN/report"
done
