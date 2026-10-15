"""Build a request workload to replay against vLLM.

Two kinds:

  synthetic  random token-id prompts. Lengths are drawn from a Vidur trace
             (prompt, output) so the shape matches what Vidur ships with.
             Exact prompt length, and with ignore_eos, exact output length.
  text       real prompts from a JSONL file. Use a dataset where n-gram
             lookup can hit (summarization, code editing, RAG) to get
             realistic acceptance. Output length is still capped by the trace
             draw; prompt length is whatever the tokenizer makes of the text.

Arrivals are Poisson at --qps with a fixed seed, so the same workload file
drives every server configuration.

Example:
  python validation/make_workload.py synthetic --qps 2 --num-requests 300 \\
      --out validation/workloads/synthetic_qps2.jsonl
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import WorkloadRequest, write_workload  # noqa: E402

DEFAULT_TRACE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "data",
    "processed_traces",
    "splitwise_conv.csv",
)


def sample_lengths(trace_file, n, rng, max_total_tokens, max_prompt_tokens=None):
    df = pd.read_csv(trace_file)
    df = df[["num_prefill_tokens", "num_decode_tokens"]].astype(int).clip(lower=1)
    rows = df.sample(n=n, replace=True, random_state=int(rng.integers(2**31))).values
    lengths = []
    for prompt, output in rows:
        if max_prompt_tokens:
            prompt = min(prompt, max_prompt_tokens)
        # leave room for the output, as Vidur's trace loader does
        prompt = max(1, min(prompt, max_total_tokens - output))
        output = max(1, min(output, max_total_tokens - prompt))
        lengths.append((int(prompt), int(output)))
    return lengths


def poisson_arrivals(n, qps, rng):
    gaps = rng.exponential(1.0 / qps, size=n)
    gaps[0] = 0.0
    return np.cumsum(gaps)


def build_synthetic(args, rng):
    lengths = sample_lengths(
        args.lengths_trace, args.num_requests, rng, args.max_total_tokens
    )
    arrivals = poisson_arrivals(args.num_requests, args.qps, rng)
    requests = []
    for i, ((prompt_len, output_len), t) in enumerate(zip(lengths, arrivals)):
        # skip the lowest ids, which are special tokens in Llama vocabularies
        ids = rng.integers(args.min_token_id, args.vocab_size, size=prompt_len)
        requests.append(
            WorkloadRequest(
                request_index=i,
                arrival_s=float(t),
                max_tokens=output_len,
                prompt_token_ids=ids.tolist(),
                ignore_eos=True,
            )
        )
    return requests


def build_text(args, rng):
    prompts = []
    with open(args.text_jsonl) as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line)[args.text_field])
    if not prompts:
        raise SystemExit(f"no prompts found in {args.text_jsonl}")
    order = rng.permutation(len(prompts))
    picks = [prompts[order[i % len(prompts)]] for i in range(args.num_requests)]
    lengths = sample_lengths(
        args.lengths_trace, args.num_requests, rng, args.max_total_tokens
    )
    arrivals = poisson_arrivals(args.num_requests, args.qps, rng)
    return [
        WorkloadRequest(
            request_index=i,
            arrival_s=float(t),
            max_tokens=output_len if args.output_tokens is None else args.output_tokens,
            prompt=text,
            ignore_eos=not args.stop_at_eos,
        )
        for i, (text, (_, output_len), t) in enumerate(zip(picks, lengths, arrivals))
    ]


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("kind", choices=["synthetic", "text"])
    p.add_argument("--out", required=True)
    p.add_argument("--qps", type=float, required=True)
    p.add_argument("--num-requests", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lengths-trace", default=DEFAULT_TRACE)
    p.add_argument(
        "--max-total-tokens",
        type=int,
        default=4096,
        help="prompt + output cap; must not exceed vLLM's --max-model-len or Vidur's profiled range",
    )
    p.add_argument("--vocab-size", type=int, default=32000)
    p.add_argument("--min-token-id", type=int, default=100)
    p.add_argument("--text-jsonl")
    p.add_argument("--text-field", default="prompt")
    p.add_argument(
        "--output-tokens",
        type=int,
        default=None,
        help="fixed output length for text prompts",
    )
    p.add_argument(
        "--stop-at-eos", action="store_true", help="let text requests end at EOS"
    )
    args = p.parse_args(argv)

    if args.kind == "text" and not args.text_jsonl:
        p.error("text workloads need --text-jsonl")

    rng = np.random.default_rng(args.seed)
    requests = (
        build_synthetic(args, rng)
        if args.kind == "synthetic"
        else build_text(args, rng)
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    write_workload(args.out, requests)
    span = requests[-1].arrival_s
    print(
        f"wrote {len(requests)} requests over {span:.1f}s ({len(requests) / max(span, 1e-9):.2f} req/s) to {args.out}"
    )


if __name__ == "__main__":
    main()
