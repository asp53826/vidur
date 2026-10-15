"""Turn one measured vLLM run into Vidur inputs.

The simulator replays exactly what the server saw: each request's actual
send time, prompt length, and output length (from vLLM's usage counts, not
the workload's targets). Server settings come from the experiment file, and
the KV cache size from what vLLM reported, so both sides schedule against the
same memory.

Acceptance is calibrated from vLLM's counters for the same run:
  - per-position acceptance, converted to the conditional form Vidur uses
  - the share of decode steps that carried proposals at all (n-gram lookups
    can miss, and vLLM doesn't count a step without proposals as a draft)

That calibration is what makes this a test of the *timing* model: acceptance
is an input, latency is the prediction.

Outputs in --out-dir: trace.csv, calibration.json, sim_args.json
"""

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import SpecCounters  # noqa: E402
from vidur.entities.speculative_decoding import (  # noqa: E402
    conditional_from_unconditional,
)


def load_experiment(path):
    with open(path) as f:
        return json.load(f)


def build_trace(measured: pd.DataFrame) -> pd.DataFrame:
    ok = measured[(measured["error"].fillna("") == "") & measured["completion_tokens"].notna()]
    trace = pd.DataFrame(
        {
            "request_index": ok["request_index"].astype(int),
            "arrived_at": ok["sent_s"] - ok["sent_s"].min(),
            "num_prefill_tokens": ok["prompt_tokens"].astype(int),
            "num_decode_tokens": ok["completion_tokens"].astype(int),
        }
    )
    # Vidur numbers requests in file order; keep that order = arrival order
    return trace.sort_values(["arrived_at", "request_index"]).reset_index(drop=True)


def calibrate(trace: pd.DataFrame, spec: SpecCounters, k: int) -> dict:
    """Acceptance parameters that reproduce the counters vLLM logged."""
    if not k:
        return {"num_speculative_tokens": 0}

    unconditional = spec.unconditional_acceptance()
    unconditional = (unconditional + [0.0] * k)[:k]
    conditional = conditional_from_unconditional(unconditional)

    # Every output token after the first comes from a decode step. A step with
    # proposals emits 1 + accepted; a step without emits 1. So
    #   tokens_after_first = steps_without + num_drafts + num_accepted
    tokens_after_first = int((trace["num_decode_tokens"] - 1).clip(lower=0).sum())
    steps_without = tokens_after_first - spec.num_drafts - spec.num_accepted_tokens
    steps_without = max(0.0, steps_without)
    total_steps = spec.num_drafts + steps_without
    proposal_probability = spec.num_drafts / total_steps if total_steps else 1.0

    return {
        "num_speculative_tokens": k,
        "acceptance_rate_per_position": conditional,
        "unconditional_acceptance_per_position": unconditional,
        "proposal_probability": proposal_probability,
        "measured_mean_acceptance_length": spec.mean_acceptance_length,
        "measured_draft_acceptance_rate": spec.draft_acceptance_rate,
        "measured_mean_draft_length": spec.mean_draft_length,
        "measured_num_drafts": spec.num_drafts,
        "estimated_decode_steps": total_steps,
    }


def vidur_args(experiment: dict, trace_path: str, num_blocks, calibration: dict) -> list:
    max_tokens = experiment["max_model_len"]
    args = [
        "--replica_config_model_name", experiment["model"],
        "--replica_config_device", experiment["device"],
        "--replica_config_network_device", experiment.get("network_device", "a100_pairwise_nvlink"),
        "--replica_config_tensor_parallel_size", str(experiment.get("tensor_parallel_size", 1)),
        "--replica_config_num_pipeline_stages", "1",
        "--cluster_config_num_replicas", "1",
        "--request_generator_config_type", "trace_replay",
        "--trace_request_generator_config_trace_file", trace_path,
        "--trace_request_generator_config_max_tokens", str(max_tokens),
        "--replica_scheduler_config_type", "sarathi",
        "--sarathi_scheduler_config_chunk_size", str(experiment["max_num_batched_tokens"]),
        "--sarathi_scheduler_config_batch_size_cap", str(experiment["max_num_seqs"]),
        "--sarathi_scheduler_config_block_size", str(experiment.get("block_size", 16)),
        "--random_forrest_execution_time_predictor_config_prediction_max_prefill_chunk_size",
        str(max(4096, experiment["max_num_batched_tokens"])),
        "--random_forrest_execution_time_predictor_config_prediction_max_batch_size",
        str(max(128, experiment["max_num_seqs"])),
        "--random_forrest_execution_time_predictor_config_prediction_max_tokens_per_request",
        str(max(4096, max_tokens)),
    ]
    if num_blocks:
        args += ["--sarathi_scheduler_config_num_blocks", str(num_blocks)]

    spec_cfg = experiment.get("speculative_config")
    k = calibration.get("num_speculative_tokens", 0)
    if spec_cfg and k:
        method = spec_cfg["method"]
        proposer = {"ngram": "ngram", "draft_model": "draft_model"}.get(method, "fixed")
        args += [
            "--speculative_decoding_config_num_speculative_tokens", str(k),
            "--speculative_decoding_config_proposer", proposer,
            "--speculative_decoding_config_acceptance_model", "per_position",
            "--speculative_decoding_config_acceptance_rate_per_position",
            *[f"{p:.6f}" for p in calibration["acceptance_rate_per_position"]],
            "--speculative_decoding_config_proposal_probability",
            f"{calibration['proposal_probability']:.6f}",
        ]
        if proposer == "draft_model":
            args += [
                "--speculative_decoding_config_draft_model_name", spec_cfg["model"],
                "--speculative_decoding_config_draft_tensor_parallel_size",
                str(spec_cfg.get("draft_tensor_parallel_size", 1)),
            ]
        overheads = experiment.get("proposer_overheads_ms", {})
        if "fixed" in overheads:
            args += ["--speculative_decoding_config_proposer_fixed_overhead_ms", str(overheads["fixed"])]
        if "per_seq" in overheads:
            args += ["--speculative_decoding_config_proposer_overhead_ms_per_seq", str(overheads["per_seq"])]
    return args


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", required=True, help="experiment JSON used to launch vLLM")
    p.add_argument("--run-dir", required=True, help="replay output directory")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args(argv)

    experiment = load_experiment(args.experiment)
    measured = pd.read_csv(os.path.join(args.run_dir, "measured.csv"))
    with open(os.path.join(args.run_dir, "server_metrics.json")) as f:
        server = json.load(f)

    os.makedirs(args.out_dir, exist_ok=True)
    trace = build_trace(measured)
    trace_path = os.path.abspath(os.path.join(args.out_dir, "trace.csv"))
    trace[["arrived_at", "num_prefill_tokens", "num_decode_tokens"]].to_csv(trace_path, index=False)
    trace[["request_index"]].to_csv(os.path.join(args.out_dir, "trace_order.csv"), index_label="sim_request_id")

    spec_cfg = experiment.get("speculative_config") or {}
    k = int(spec_cfg.get("num_speculative_tokens", 0))
    calibration = calibrate(trace, SpecCounters.from_dict(server["spec_decode"]), k)

    cache = server.get("cache_config", {})
    num_blocks = cache.get("num_gpu_blocks")
    if num_blocks in (None, "", "None"):
        num_blocks = None
        print("WARNING: vLLM did not report num_gpu_blocks; Vidur will size the cache itself", file=sys.stderr)
    if cache.get("block_size") and int(cache["block_size"]) != experiment.get("block_size", 16):
        print(f"WARNING: vLLM block_size {cache['block_size']} differs from the experiment file", file=sys.stderr)

    sim_args = vidur_args(experiment, trace_path, num_blocks, calibration)
    with open(os.path.join(args.out_dir, "calibration.json"), "w") as f:
        json.dump(calibration, f, indent=2)
    with open(os.path.join(args.out_dir, "sim_args.json"), "w") as f:
        json.dump(sim_args, f, indent=2)

    print(f"{len(trace)} requests -> {trace_path}")
    print(json.dumps(calibration, indent=2))


if __name__ == "__main__":
    main()
