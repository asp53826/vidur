"""Score Vidur's predictions against what vLLM measured.

For every simulated variant, requests are matched one to one with the
measured run. Reported per metric (TTFT, TPOT, end-to-end latency):
median / p90 / p99 on both sides, the signed relative error of each
percentile, and the per-request mean absolute percentage error. Throughput
and, for speculative runs, mean acceptance length are compared too.

Writes report.json and report.md to --out-dir.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import mape, percentile, relative_error, tpot  # noqa: E402

PERCENTILES = (50, 90, 99)


def measured_frame(run_dir, sim_inputs):
    measured = pd.read_csv(os.path.join(run_dir, "measured.csv"))
    order = pd.read_csv(os.path.join(sim_inputs, "trace_order.csv"))
    df = order.merge(measured, on="request_index", how="left")
    df["ttft"] = df["ttft_s"]
    df["e2e"] = df["e2e_s"]
    df["tpot"] = [tpot(e, t, int(n)) for e, t, n in zip(df["e2e"], df["ttft"], df["completion_tokens"])]
    df["end"] = df["sent_s"] + df["e2e"]
    return df


def simulated_frame(result_dir, measured):
    sim = pd.read_csv(os.path.join(result_dir, "request_metrics.csv"))
    sim = sim.rename(columns={"Request Id": "sim_request_id"})
    sim = sim.sort_values("sim_request_id").reset_index(drop=True)
    df = measured[["sim_request_id", "request_index", "completion_tokens"]].merge(
        sim, on="sim_request_id", how="left"
    )
    df["ttft"] = df["prefill_e2e_time"]
    df["e2e"] = df["request_e2e_time"]
    df["tpot"] = [tpot(e, t, int(n)) for e, t, n in zip(df["e2e"], df["ttft"], df["request_num_decode_tokens"])]
    return df


def throughput(tokens, starts, ends):
    span = np.nanmax(ends) - np.nanmin(starts)
    return float(np.nansum(tokens) / span) if span > 0 else float("nan")


def compare(measured, simulated, sim_trace):
    out = {}
    for metric in ("ttft", "tpot", "e2e"):
        m, s = measured[metric].to_numpy(float), simulated[metric].to_numpy(float)
        entry = {"mape": mape(s, m)}
        for q in PERCENTILES:
            mq, sq = percentile(m, q), percentile(s, q)
            entry[f"p{q}"] = {"measured": mq, "simulated": sq, "rel_error": relative_error(sq, mq)}
        out[metric] = entry

    measured_tput = throughput(measured["completion_tokens"], measured["sent_s"], measured["end"])
    arrivals = sim_trace["arrived_at"].to_numpy(float)
    sim_ends = arrivals + simulated["e2e"].to_numpy(float)
    sim_tput = throughput(sim_trace["num_decode_tokens"], arrivals, sim_ends)
    out["output_throughput_tok_s"] = {
        "measured": measured_tput,
        "simulated": sim_tput,
        "rel_error": relative_error(sim_tput, measured_tput),
    }
    return out


def fmt_pct(x):
    return "n/a" if x is None or not np.isfinite(x) else f"{x * 100:+.1f}%"


def fmt_ms(x):
    return "n/a" if x is None or not np.isfinite(x) else f"{x * 1e3:.1f}"


def markdown(results, run_dir, calibration, server):
    lines = [f"# Vidur vs vLLM: {os.path.basename(os.path.normpath(run_dir))}", ""]
    if calibration.get("num_speculative_tokens"):
        lines += [
            f"Speculative decoding, k = {calibration['num_speculative_tokens']}. "
            f"Measured mean acceptance length {calibration['measured_mean_acceptance_length']:.3f}, "
            f"proposal share {calibration['proposal_probability']:.3f}, "
            f"mean draft length {calibration['measured_mean_draft_length']:.2f}.",
            "",
        ]
    lines += [
        "| variant | metric | measured p50 / p90 / p99 (ms) | simulated p50 / p90 / p99 (ms) | error p50 / p90 / p99 | MAPE |",
        "|---|---|---|---|---|---|",
    ]
    for name, r in results.items():
        for metric in ("ttft", "tpot", "e2e"):
            e = r[metric]
            lines.append(
                f"| {name} | {metric.upper()} | "
                + " / ".join(fmt_ms(e[f"p{q}"]["measured"]) for q in PERCENTILES)
                + " | "
                + " / ".join(fmt_ms(e[f"p{q}"]["simulated"]) for q in PERCENTILES)
                + " | "
                + " / ".join(fmt_pct(e[f"p{q}"]["rel_error"]) for q in PERCENTILES)
                + f" | {fmt_pct(e['mape']).lstrip('+')} |"
            )
    lines += ["", "| variant | throughput measured (tok/s) | simulated | error | sim mean acceptance length |", "|---|---|---|---|---|"]
    for name, r in results.items():
        t = r["output_throughput_tok_s"]
        mal = r.get("sim_mean_acceptance_length")
        lines.append(
            f"| {name} | {t['measured']:.0f} | {t['simulated']:.0f} | {fmt_pct(t['rel_error'])} | "
            + ("n/a" if mal is None else f"{mal:.3f}")
            + " |"
        )
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, help="replay output (measured.csv)")
    p.add_argument("--sim-inputs", required=True)
    p.add_argument("--sim-dir", required=True, help="run_sim.py output")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args(argv)

    measured = measured_frame(args.run_dir, args.sim_inputs)
    sim_trace = pd.read_csv(os.path.join(args.sim_inputs, "trace.csv"))
    with open(os.path.join(args.sim_inputs, "calibration.json")) as f:
        calibration = json.load(f)
    with open(os.path.join(args.run_dir, "server_metrics.json")) as f:
        server = json.load(f)

    results = {}
    for variant_file in sorted(glob.glob(os.path.join(args.sim_dir, "*", "variant.json"))):
        with open(variant_file) as f:
            variant = json.load(f)
        simulated = simulated_frame(variant["result_dir"], measured)
        missing = simulated["e2e"].isna().sum()
        if missing:
            print(f"WARNING: {variant['name']}: {missing} requests missing from Vidur output", file=sys.stderr)
        r = compare(measured, simulated, sim_trace)
        summary = os.path.join(variant["result_dir"], "speculative_decoding_summary.json")
        if os.path.exists(summary):
            with open(summary) as f:
                r["sim_mean_acceptance_length"] = json.load(f)["mean_acceptance_length"]
        results[variant["name"]] = r

    if not results:
        raise SystemExit(f"no variants found under {args.sim_dir}")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "report.json"), "w") as f:
        json.dump({"calibration": calibration, "results": results}, f, indent=2)
    md = markdown(results, args.run_dir, calibration, server)
    with open(os.path.join(args.out_dir, "report.md"), "w") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
