# Validating speculative decoding against vLLM on Sapelo2

The question: when Vidur predicts TTFT, TPOT and end-to-end latency for a
vLLM deployment with speculative decoding, how far off is it, and which
verify-attention model (see `docs/speculative_decoding.md`) matches real
kernels?

The method: send the same timed requests to a real vLLM server and to the
simulator, with acceptance calibrated from the server's own counters, and
compare request by request. Acceptance is an input; latency is the
prediction being tested.

```
make_workload.py ──► workload.jsonl ──► replay.py ──► measured.csv + server_metrics.json
   (seeded arrivals,                     (against vLLM                    │
    exact lengths)                        on a GPU node)                  ▼
                                                              build_sim_inputs.py
                                                         (trace + calibrated acceptance)
                                                                          │
                                       report.md ◄── report.py ◄── run_sim.py (Vidur, CPU)
```

## Experiments

| file | what it tests | GPUs |
|---|---|---|
| `e0_llama2_7b_a100_baseline` | Vidur's error with no speculative decoding. Everything else is read against this. | 1 A100 |
| `e1_llama2_7b_a100_ngram_k{2,4,6}` | Verify-step cost and the proposal-miss model, with a nearly free proposer. | 1 A100 |
| `e2_llama2_70b_tp4_a100_draft7b_k4` | Draft-model cost and draft memory accounting. | 4 A100 |

Llama-2 is used because Vidur ships A100 profiles for both 7B and 70B, so no
profiling is needed to start. Sapelo2's A100s are 80 GB, matching Vidur's
`a100` profile.

## Steps

All commands from the repository root.

**1. Environment (once, on a login node).**

```bash
ml spider Python                      # pick a 3.10-3.12 module
export PYTHON_MODULE=<that module>
bash validation/slurm/setup_env.sh
export HF_TOKEN=...                   # after accepting the Llama 2 license on Hugging Face
export HF_HOME=/scratch/$USER/hf
```

**2. Workloads.** Same files for every experiment, so runs are comparable.

```bash
for qps in 1 3 6; do
  python validation/make_workload.py synthetic --qps $qps --num-requests 300 \
    --out validation/workloads/synthetic_qps$qps.jsonl
done
```

Synthetic prompts are random token ids, so n-gram acceptance comes only from
the model repeating itself. For realistic acceptance, add a text workload from
a dataset where lookups hit (summarization, code editing, RAG):

```bash
python validation/make_workload.py text --text-jsonl /path/to/prompts.jsonl \
  --qps 3 --out validation/workloads/text_qps3.jsonl
```

**3. Dry run without a GPU** (catches path and environment problems before
you wait in the queue):

```bash
mkdir -p validation/logs
FAKE_VLLM=1 STARTUP_POLL_S=1 bash validation/slurm/run_experiment.sbatch \
  validation/experiments/e0_llama2_7b_a100_baseline.json validation/workloads/synthetic_qps1.jsonl
rm -rf validation/results   # dry-run output is meaningless
```

**4. GPU runs.**

```bash
mkdir -p validation/logs
W="validation/workloads/synthetic_qps1.jsonl validation/workloads/synthetic_qps3.jsonl validation/workloads/synthetic_qps6.jsonl"
sbatch validation/slurm/run_experiment.sbatch validation/experiments/e0_llama2_7b_a100_baseline.json $W
for k in 2 4 6; do
  sbatch validation/slurm/run_experiment.sbatch validation/experiments/e1_llama2_7b_a100_ngram_k$k.json $W
done
sbatch --gres=gpu:A100:4 validation/slurm/run_experiment.sbatch \
  validation/experiments/e2_llama2_70b_tp4_a100_draft7b_k4.json $W
```

Each job writes `validation/results/<experiment>/<workload>/` with
`measured.csv`, `server_metrics.json`, and the server log. `environment.txt`
records the GPU, driver, power limit and vLLM version: put those in any
write-up.

**5. Simulate and score (CPU).**

```bash
bash validation/slurm/simulate_and_report.sh validation/experiments/e0_llama2_7b_a100_baseline.json
bash validation/slurm/simulate_and_report.sh validation/experiments/e1_llama2_7b_a100_ngram_k4.json
```

Each run gets `report/report.md`: measured vs simulated p50/p90/p99 for TTFT,
TPOT and E2E, per-request MAPE, throughput, and the simulator's mean
acceptance length next to vLLM's (they should match closely; if not, the
calibration is off, not the timing model).

## Reading the results

1. **Start with e0.** That is the floor. If Vidur is 15% off on TPOT with no
   speculation at all, a 17% error with speculation means the new modeling
   added about 2 points, not 17.
2. **e1: which bound?** Compare `decode_shared_kv`, `decode_per_token`, and
   the interpolated points. Expect them to agree at low load and small k and
   to separate as both grow. The weight that best fits TPOT across loads and
   k is the result; if it moves a lot with load, neither bound captures the
   kernel, and verify-shaped attention needs its own profile.
3. **Check the gap notes.** The report prints vLLM's mean draft length. If
   it's well under k, the simulator is charging for proposals vLLM never
   made (a known limitation, in `docs/speculative_decoding.md`).
4. **e2** adds the draft model and tensor parallelism. Vidur's
   `a100_pairwise_nvlink` network profile assumes two NVLinked pairs; check
   the actual topology with `nvidia-smi topo -m` on the node. If it differs,
   profile collectives with Vidur's profiler (`docs/profiling.md`) before
   trusting e2's numbers.

## Things that can bite

- **Prefix caching** is on by default in vLLM v1 and Vidur doesn't model it.
  `launch_vllm.py` turns it off.
- **KV cache size.** vLLM sizes its cache from free memory at startup.
  `build_sim_inputs.py` passes vLLM's reported `num_gpu_blocks` to Vidur so
  both schedule against the same memory; it warns if vLLM didn't report it.
- **Client lag.** `replay_meta.json` reports how late requests were sent. If
  the max is above ~50 ms the client couldn't keep up; give the job more CPUs.
- **Failed requests** are dropped from the comparison and counted in
  `replay_meta.json`. More than a handful means the run isn't comparable.
- **HTTP and tokenization overhead** show up in measured TTFT but not in
  Vidur's. Expect a small constant TTFT offset.
