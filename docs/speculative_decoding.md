# Speculative decoding in Vidur

This branch adds speculative decoding to Vidur's simulation of a vLLM v1-style
engine. With `num_speculative_tokens=0` (the default) it behaves exactly like
upstream: a full simulation writes a byte-identical `request_metrics.csv`
(checked against commit `abae7f6` by `scripts/check_matches_upstream.sh`).
One deliberate difference: decode-attention lookups past the profiled batch
size, which raised `KeyError` upstream, now extrapolate linearly.

```bash
python -m vidur.main \
  --replica_scheduler_config_type sarathi \
  --speculative_decoding_config_num_speculative_tokens 4 \
  --speculative_decoding_config_acceptance_rate 0.7 \
  ...   # the usual model, device and workload flags
```

Outputs gain `speculative_decoding_metrics.csv` (per request) and
`speculative_decoding_summary.json`, which use vLLM's definitions of mean
acceptance length and draft acceptance rate so the two can be compared
directly.

## What a decode step does

A decoding sequence with proposals feeds the newest token plus `k` proposed
tokens through the target model in one forward pass. Proposals are accepted
left to right until the first rejection, and the target contributes one more
token, so the step emits between 1 and `k + 1` tokens.

Vidur's `Batch` used one number per request for both "tokens computed" and
"tokens of progress". Those now come apart: `num_tokens` (k + 1 for a verify
step) drives the cost model, and `num_advance_tokens` (accepted + 1) drives
the request forward.

## What is modeled

| piece | how | where |
|---|---|---|
| Tokens per step | Truncated geometric over conditional acceptance probabilities p₁..p_k: constant α, a per-position list, or a per-request rate from a Beta distribution. A deterministic expected-value mode exists for analysis. | `entities/speculative_decoding.py` |
| Steps without proposals | `proposal_probability`: n-gram lookup only proposes when it finds a match, and vLLM runs those steps as plain decodes. | same |
| Token budget | Verify tokens count against Sarathi's `chunk_size`, and are trimmed when the budget runs out, as vLLM v1 trims spec tokens. Proposals never run past the end of a request. | `sarathi_replica_scheduler.py` |
| KV memory | Blocks for every proposal are reserved before the step (num_processed + num_tokens − 1 slots). Slots of rejected tokens stay allocated and are reused, as vLLM's lookahead slots are. | same |
| Linear layers, KV writes | Fall out of Vidur's existing token-count features once a verify step carries k + 1 tokens. | unchanged |
| Verify attention | Bracketed, see below. | `sklearn_execution_time_predictor.py` |
| Proposer | `ngram` / `fixed`: a + b·(drafting sequences) ms. `draft_model`: k forward passes of a second, independently trained predictor for the draft model; pass 1 mirrors the target batch (so the draft's KV stays in sync), passes 2..k add one token per sequence. | same |
| Draft memory | Draft weights and the draft's KV per request come out of the KV budget. | `memory_planner.py` |

## The open question: verify attention

Vidur profiles two attention shapes: decode (one query token per sequence)
and prefill chunks (one sequence at a time, many query tokens). A verify step
is neither: every sequence in the batch has k + 1 query tokens against its
own, usually long, KV cache.

Mapping it to the prefill-chunk model doesn't work. Vidur combines several
chunks by summing their KV lengths, and a batch of 64 sequences with 1,000
tokens of context each sums to 64,000, far outside the profiled range (and
not what the kernel does).

So the cost is bracketed instead:

- **`decode_shared_kv` (lower bound).** Each sequence's KV is read once and
  the extra query tokens are free. This is the memory-bound regime decode
  attention usually sits in, and for small k it may be close to right.
- **`decode_per_token` (upper bound).** Every query token costs as much as a
  full decode query, as if the KV were read k + 1 times.
- **`interpolated`.** `lower + w · (upper − lower)`, with `w` fitted against
  measurements.

The bounds diverge as load and k grow (more sequences, more query tokens per
sequence). Where real kernels fall between them, and whether that depends on
the attention backend, is what the validation harness measures. If neither
bound fits, the next step is profiling verify-shaped attention directly
(Vidur's attention profiler with q > 1 per decode sequence) and training a
third model on it.

### How much the choice matters

`scripts/spec_decode_sweep.py` runs Llama-2-7B on Vidur's A100 profile
(Splitwise conversation lengths, 300 requests, Sarathi with a 2,048-token
budget) across load, k and acceptance rate, under both bounds. These are the
simulator's predictions, not measurements.

![Predicted TPOT speedup](assets/spec_decode_sweep/tpot_speedup.png)

Median TPOT speedup over no speculation (full table in
`assets/spec_decode_sweep/results.csv`):

| load | k | α | KV read once | per-token cost |
|---|---|---|---|---|
| 1 req/s | 4 | 0.8 | 3.7× | 2.7× |
| 1 req/s | 8 | 0.6 | 2.6× | 1.4× |
| 3 req/s | 4 | 0.6 | 3.0× | 0.74× |
| 6 req/s | 4 | 0.6 | 2.0× | 0.68× |
| 6 req/s | 8 | 0.8 | 6.3× | 1.0× |

At light load the bounds agree that speculation helps and differ on how much.
At 3 req/s, where every configuration in the table still runs in steady state,
they disagree about the sign: with k = 4 and α = 0.6, one predicts
speculative decoding cuts median TPOT by 3× and the other predicts it makes
TPOT about 1.35× worse. Whether to turn speculation on at a given load is the
decision a capacity planner would use the simulator for, and it hinges on a
kernel property nobody has measured for this setup.

Two caveats on reading the numbers:

- Speedups above the mean acceptance length (e.g. 5.1× at 3 req/s, k = 4,
  α = 0.8, where acceptance length is 3.3) are a load effect, not a bug:
  finishing requests sooner shrinks the running batch and the queue, so every
  step gets cheaper.
- 6 req/s is past this replica's capacity even without speculation: the
  simulated baseline's median TTFT climbs from 0.15 s in the first quarter of
  requests to 5.5 s in the last. Ratios there depend on run length and show
  direction only. The 3 req/s row is steady state (median TTFT stays flat
  across the run under both bounds).

## Calibrating acceptance from vLLM

Acceptance is an input, not a prediction: it depends on the model pair and
the text, not on the serving system. The validation harness reads vLLM's
counters for a run and converts them:

- vLLM's per-position acceptance (`vllm:spec_decode_num_accepted_tokens_per_pos`
  divided by `num_drafts`) is the *unconditional* probability that positions
  1..i were all accepted. `conditional_from_unconditional` turns it into
  p_i = a_i / a_{i−1}.
- vLLM only counts a step as a draft when the proposer returned tokens. Every
  output token after the first comes from a decode step, so
  `steps_without_proposals = (Σ output_tokens − requests) − num_drafts − num_accepted`,
  which gives `proposal_probability`.

## Known limitations

- **A step proposes k tokens or none.** vLLM's n-gram proposer can return
  fewer than k tokens when the match is near the end of the context. vLLM exports the
  mean draft length; the report prints it so the gap is visible. Shorter
  drafts mean the simulator over-charges verify cost.
- **Short drafts fold into acceptance.** vLLM's per-position counters can't
  tell "position i wasn't proposed" from "position i was rejected", so the
  calibrated acceptance rates absorb both.
- **No rejection-sampler or bookkeeping cost.** Small on GPU, but not zero.
- **Draft pass 1 is approximate.** It uses the target batch's token layout;
  vLLM's draft-model proposer feeds the accepted tokens, which differ step to
  step.
- **Sarathi scheduler only, pipeline parallelism 1.** Other schedulers refuse
  a speculative config instead of ignoring it. vLLM's docs list speculative
  decoding and pipeline parallelism as not composable (as of 0.15).
- **Profiles predate current vLLM kernels.** Vidur's profiling data were
  collected with Sarathi-Serve-era kernels. The baseline experiment (k = 0)
  measures how much error that alone causes on a given GPU and vLLM version;
  speculative results should be read relative to it.

## Tests

```bash
python -m pytest tests            # acceptance model, scheduler, cost model
python -m pytest validation/tests # harness, against a fake vLLM server
VIDUR_CACHE_DIR=cache python -m pytest validation/tests   # + real Vidur end to end
```

The scheduler tests drive `SarathiReplicaScheduler` directly (no event loop,
no trained predictor) and check: exact token accounting, k + 1 tokens per
step at α = 1, budget trimming, KV slots covering every verify step, recovery
from preemption under memory pressure, and that `proposal_probability = 0`
reproduces plain decoding batch for batch.
