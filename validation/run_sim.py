"""Run Vidur on the inputs build_sim_inputs.py produced.

For speculative runs, the verify-attention model is the open question, so this
runs both bounds and, optionally, interpolated points between them. Each
variant lands in its own directory under --out-dir with a variant.json naming
it, which report.py reads.

Run from the repository root (Vidur resolves profiling data relative to it).
"""

import argparse
import glob
import json
import os
import subprocess
import sys


def run_variant(sim_args, name, extra, out_root, cache_dir, seed):
    out_dir = os.path.join(out_root, name)
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "vidur.main",
        *sim_args,
        *extra,
        "--seed",
        str(seed),
        "--metrics_config_output_dir",
        out_dir,
        "--metrics_config_cache_dir",
        cache_dir,
        "--no-metrics_config_store_plots",
        "--no-metrics_config_enable_chrome_trace",
    ]
    env = dict(os.environ, WANDB_MODE="disabled")
    print(f"[{name}] running Vidur", flush=True)
    with open(os.path.join(out_dir, "vidur.log"), "w") as log:
        subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT, env=env)

    # MetricsConfig writes into a timestamped subdirectory; take the newest
    runs = sorted(glob.glob(os.path.join(out_dir, "*", "request_metrics.csv")))
    if not runs:
        raise RuntimeError(
            f"Vidur produced no request_metrics.csv; see {out_dir}/vidur.log"
        )
    with open(os.path.join(out_dir, "variant.json"), "w") as f:
        json.dump(
            {
                "name": name,
                "extra_args": extra,
                "result_dir": os.path.dirname(runs[-1]),
            },
            f,
            indent=2,
        )
    return out_dir


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--sim-inputs", required=True, help="build_sim_inputs.py output directory"
    )
    p.add_argument("--out-dir", required=True)
    p.add_argument("--cache-dir", default="cache")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--attention-models",
        nargs="+",
        default=["decode_shared_kv", "decode_per_token"],
    )
    p.add_argument(
        "--weights",
        nargs="*",
        type=float,
        default=[],
        help="interpolated weights to also run",
    )
    args = p.parse_args(argv)

    with open(os.path.join(args.sim_inputs, "sim_args.json")) as f:
        sim_args = json.load(f)
    with open(os.path.join(args.sim_inputs, "calibration.json")) as f:
        calibration = json.load(f)

    if not calibration.get("num_speculative_tokens"):
        run_variant(sim_args, "baseline", [], args.out_dir, args.cache_dir, args.seed)
        return

    flag = "--speculative_decoding_config_verify_attention_model"
    for model in args.attention_models:
        run_variant(
            sim_args, model, [flag, model], args.out_dir, args.cache_dir, args.seed
        )
    for w in args.weights:
        run_variant(
            sim_args,
            f"interpolated_w{w:g}",
            [
                flag,
                "interpolated",
                "--speculative_decoding_config_verify_attention_weight",
                str(w),
            ],
            args.out_dir,
            args.cache_dir,
            args.seed,
        )


if __name__ == "__main__":
    main()
