#!/bin/bash
# Check that with speculative decoding off, this branch simulates exactly what
# upstream Vidur does: same config on both, request_metrics.csv compared byte
# for byte. First run trains predictor models (~15 min on 2 cores); both
# checkouts share ./cache so the second is quick.
#
#   bash scripts/check_matches_upstream.sh [UPSTREAM_REF]   # default abae7f6
set -euo pipefail

REF=${1:-abae7f6}
ROOT=$(git rev-parse --show-toplevel)
WORK=$(mktemp -d)
trap 'git -C "$ROOT" worktree remove --force "$WORK/upstream" 2>/dev/null || true; rm -rf "$WORK"' EXIT

git -C "$ROOT" worktree add -q "$WORK/upstream" "$REF"
export WANDB_MODE=disabled

simulate() {
  (cd "$1" && python -m vidur.main \
    --replica_config_device a100 --replica_config_model_name meta-llama/Llama-2-7b-hf \
    --request_generator_config_type synthetic --synthetic_request_generator_config_num_requests 128 \
    --length_generator_config_type trace --trace_request_length_generator_config_max_tokens 4096 \
    --trace_request_length_generator_config_trace_file ./data/processed_traces/splitwise_conv.csv \
    --interval_generator_config_type poisson --poisson_request_interval_generator_config_qps 2 \
    --replica_scheduler_config_type sarathi --sarathi_scheduler_config_batch_size_cap 256 \
    --sarathi_scheduler_config_chunk_size 512 \
    --metrics_config_cache_dir "$ROOT/cache" --metrics_config_output_dir "$2" \
    --no-metrics_config_store_plots --no-metrics_config_enable_chrome_trace > "$2.log" 2>&1)
}

simulate "$WORK/upstream" "$WORK/out_upstream"
simulate "$ROOT" "$WORK/out_branch"

A=$(ls "$WORK"/out_upstream/*/request_metrics.csv)
B=$(ls "$WORK"/out_branch/*/request_metrics.csv)
if cmp -s "$A" "$B"; then
  echo "identical: $(wc -l < "$B") lines of request metrics match upstream $REF"
else
  echo "DIFFERENT from upstream $REF"; diff <(head -5 "$A") <(head -5 "$B") || true; exit 1
fi
