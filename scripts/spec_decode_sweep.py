"""Simulator-only sweep of speculative decoding settings.

These are Vidur's *predictions*, not measurements. For each load level,
proposal count k and acceptance rate, it runs both verify-attention bounds,
so the plot shows a band rather than a line: the width of the band is how
much the answer depends on the one modeling choice the GPU validation is
meant to settle.

  python scripts/spec_decode_sweep.py --out-dir docs/assets/spec_decode_sweep
"""

import argparse
import glob
import itertools
import json
import os
import subprocess
import sys

import pandas as pd

BASE = [
    "--replica_config_device",
    "a100",
    "--replica_config_model_name",
    "meta-llama/Llama-2-7b-hf",
    "--cluster_config_num_replicas",
    "1",
    "--replica_config_tensor_parallel_size",
    "1",
    "--replica_config_num_pipeline_stages",
    "1",
    "--request_generator_config_type",
    "synthetic",
    "--length_generator_config_type",
    "trace",
    "--trace_request_length_generator_config_max_tokens",
    "4096",
    "--trace_request_length_generator_config_trace_file",
    "./data/processed_traces/splitwise_conv.csv",
    "--interval_generator_config_type",
    "poisson",
    "--replica_scheduler_config_type",
    "sarathi",
    "--sarathi_scheduler_config_chunk_size",
    "2048",
    "--sarathi_scheduler_config_batch_size_cap",
    "128",
    "--no-metrics_config_store_plots",
    "--no-metrics_config_enable_chrome_trace",
]


def run(out_dir, cache_dir, num_requests, qps, k, alpha, model):
    name = f"qps{qps:g}_k{k}_a{alpha:g}_{model}"
    run_dir = os.path.join(out_dir, "runs", name)
    if not glob.glob(os.path.join(run_dir, "*", "request_metrics.csv")):
        os.makedirs(run_dir, exist_ok=True)
        args = BASE + [
            "--synthetic_request_generator_config_num_requests",
            str(num_requests),
            "--poisson_request_interval_generator_config_qps",
            str(qps),
            "--metrics_config_output_dir",
            run_dir,
            "--metrics_config_cache_dir",
            cache_dir,
        ]
        if k:
            args += [
                "--speculative_decoding_config_num_speculative_tokens",
                str(k),
                "--speculative_decoding_config_acceptance_rate",
                str(alpha),
                "--speculative_decoding_config_verify_attention_model",
                model,
            ]
        with open(os.path.join(run_dir, "vidur.log"), "w") as log:
            subprocess.run(
                [sys.executable, "-m", "vidur.main", *args],
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=dict(os.environ, WANDB_MODE="disabled"),
            )
    result = glob.glob(os.path.join(run_dir, "*", "request_metrics.csv"))[0]
    df = pd.read_csv(result)
    tpot = (df.request_e2e_time - df.prefill_e2e_time) / (
        df.request_num_decode_tokens - 1
    )
    summary_file = os.path.join(
        os.path.dirname(result), "speculative_decoding_summary.json"
    )
    mal = None
    if os.path.exists(summary_file):
        with open(summary_file) as f:
            mal = json.load(f)["mean_acceptance_length"]
    return {
        "qps": qps,
        "k": k,
        "alpha": alpha if k else None,
        "verify_attention_model": model if k else None,
        "tpot_p50_ms": tpot.median() * 1e3,
        "tpot_p99_ms": tpot.quantile(0.99) * 1e3,
        "ttft_p50_ms": df.prefill_e2e_time.median() * 1e3,
        "ttft_p99_ms": df.prefill_e2e_time.quantile(0.99) * 1e3,
        "e2e_p50_s": df.request_e2e_time.median(),
        "e2e_p99_s": df.request_e2e_time.quantile(0.99),
        "mean_acceptance_length": mal,
    }


def plot(df, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    qps_values = sorted(df.qps.unique())
    alphas = sorted(df.alpha.dropna().unique())
    fig, axes = plt.subplots(
        1, len(qps_values), figsize=(4.2 * len(qps_values), 3.6), sharey=True
    )
    colors = ["#2f6db3", "#d1495b", "#3c9d5d", "#8a5bb5"]
    for ax, qps in zip(axes, qps_values):
        base = df[(df.qps == qps) & (df.k == 0)].tpot_p50_ms.iloc[0]
        for color, alpha in zip(colors, alphas):
            sub = df[(df.qps == qps) & (df.alpha == alpha)]
            lo = sub[sub.verify_attention_model == "decode_shared_kv"].sort_values("k")
            hi = sub[sub.verify_attention_model == "decode_per_token"].sort_values("k")
            ks = [0] + list(lo.k)
            ax.fill_between(
                ks,
                [1] + list(base / hi.tpot_p50_ms),
                [1] + list(base / lo.tpot_p50_ms),
                color=color,
                alpha=0.25,
                linewidth=0,
            )
            ax.plot(
                ks,
                [1] + list(base / lo.tpot_p50_ms),
                color=color,
                linewidth=1.5,
                label=f"α = {alpha:g}",
            )
            ax.plot(
                ks,
                [1] + list(base / hi.tpot_p50_ms),
                color=color,
                linewidth=1.5,
                linestyle="--",
            )
        ax.axhline(1, color="#888", linewidth=0.8)
        ax.set_title(f"{qps:g} req/s")
        ax.set_xlabel("proposals per step (k)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("predicted median TPOT speedup")
    axes[0].legend(
        frameon=False,
        fontsize=8,
        title="solid: KV read once\ndashed: per-token cost",
        title_fontsize=7,
    )
    fig.suptitle("Vidur prediction, Llama-2-7B on A100 (not measured)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "tpot_speedup.png"), dpi=150)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out-dir", default="docs/assets/spec_decode_sweep")
    p.add_argument("--cache-dir", default="cache")
    p.add_argument("--num-requests", type=int, default=300)
    p.add_argument("--qps", nargs="+", type=float, default=[1, 3, 6])
    p.add_argument("--k", nargs="+", type=int, default=[2, 4, 8])
    p.add_argument("--alpha", nargs="+", type=float, default=[0.6, 0.8])
    args = p.parse_args()

    rows = []
    for qps in args.qps:
        rows.append(
            run(args.out_dir, args.cache_dir, args.num_requests, qps, 0, 0, "none")
        )
        for k, alpha, model in itertools.product(
            args.k, args.alpha, ["decode_shared_kv", "decode_per_token"]
        ):
            rows.append(
                run(
                    args.out_dir,
                    args.cache_dir,
                    args.num_requests,
                    qps,
                    k,
                    alpha,
                    model,
                )
            )
            print(json.dumps(rows[-1]), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, "results.csv"), index=False)
    plot(df, args.out_dir)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
