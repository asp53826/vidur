"""Harness tests. No GPU: a fake vLLM server stands in for the real one.

The slow end-to-end test also runs Vidur itself on the fake run, which
needs trained predictor models; it is skipped unless VIDUR_CACHE_DIR points
at a populated cache (any previous Vidur run's cache directory).
"""

import asyncio
import json
import os
import socket
import subprocess
import sys
import threading

import numpy as np
import pandas as pd
import pytest
from aiohttp import web

HERE = os.path.dirname(os.path.abspath(__file__))
VALIDATION = os.path.dirname(HERE)
REPO = os.path.dirname(VALIDATION)
sys.path.insert(0, VALIDATION)
sys.path.insert(0, HERE)

import build_sim_inputs  # noqa: E402
import make_workload  # noqa: E402
import replay  # noqa: E402
import report  # noqa: E402
from common import (  # noqa: E402
    SpecCounters,
    cache_config_from_samples,
    mape,
    parse_prometheus,
    read_workload,
    tpot,
)
from fake_vllm_server import make_app  # noqa: E402

PROM = """
# HELP vllm:spec_decode_num_drafts Number of spec decoding drafts.
# TYPE vllm:spec_decode_num_drafts counter
vllm:spec_decode_num_drafts_total{engine="0",model_name="m"} 100.0
vllm:spec_decode_num_draft_tokens_total{engine="0",model_name="m"} 400.0
vllm:spec_decode_num_accepted_tokens_total{engine="0",model_name="m"} 180.0
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",model_name="m",position="0"} 80.0
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",model_name="m",position="1"} 60.0
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",model_name="m",position="2"} 30.0
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",model_name="m",position="3"} 10.0
vllm:cache_config_info{block_size="16",engine="0",num_gpu_blocks="27000"} 1.0
"""


def test_parse_spec_counters():
    spec = SpecCounters.from_samples(parse_prometheus(PROM))
    assert spec.num_drafts == 100 and spec.num_accepted_tokens == 180
    assert spec.mean_acceptance_length == pytest.approx(2.8)
    assert spec.draft_acceptance_rate == pytest.approx(0.45)
    assert spec.unconditional_acceptance() == pytest.approx([0.8, 0.6, 0.3, 0.1])


def test_counter_delta():
    after = SpecCounters.from_samples(parse_prometheus(PROM))
    before = SpecCounters(
        num_drafts=40,
        num_draft_tokens=160,
        num_accepted_tokens=80,
        accepted_per_pos={0: 30},
    )
    d = after.minus(before)
    assert (d.num_drafts, d.num_draft_tokens, d.num_accepted_tokens) == (60, 240, 100)
    assert d.accepted_per_pos[0] == 50 and d.accepted_per_pos[3] == 10


def test_cache_config():
    cfg = cache_config_from_samples(parse_prometheus(PROM))
    assert cfg["num_gpu_blocks"] == "27000" and cfg["block_size"] == "16"


def test_tpot_and_mape():
    assert tpot(1.1, 0.1, 101) == pytest.approx(0.01)
    assert np.isnan(tpot(1.0, 0.1, 1))
    assert mape([1.1, 0.9], [1.0, 1.0]) == pytest.approx(0.1)


def test_synthetic_workload(tmp_path):
    out = tmp_path / "w.jsonl"
    make_workload.main(
        [
            "synthetic",
            "--qps",
            "5",
            "--num-requests",
            "50",
            "--seed",
            "3",
            "--max-total-tokens",
            "2048",
            "--out",
            str(out),
        ]
    )
    w = read_workload(str(out))
    assert len(w) == 50
    arrivals = [r.arrival_s for r in w]
    assert arrivals == sorted(arrivals) and arrivals[0] == 0.0
    for r in w:
        assert len(r.prompt_token_ids) + r.max_tokens <= 2048
        assert min(r.prompt_token_ids) >= 100 and max(r.prompt_token_ids) < 32000
    out2 = tmp_path / "w2.jsonl"
    make_workload.main(
        [
            "synthetic",
            "--qps",
            "5",
            "--num-requests",
            "50",
            "--seed",
            "3",
            "--max-total-tokens",
            "2048",
            "--out",
            str(out2),
        ]
    )
    assert out.read_text() == out2.read_text()


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def fake_server():
    port = free_port()
    loop = asyncio.new_event_loop()
    runner = web.AppRunner(make_app(k=4, alpha=0.7, step_s=0.001, seed=1))
    started = threading.Event()

    def serve():
        asyncio.set_event_loop(loop)
        loop.run_until_complete(runner.setup())
        loop.run_until_complete(web.TCPSite(runner, "127.0.0.1", port).start())
        started.set()
        loop.run_forever()

    t = threading.Thread(target=serve, daemon=True)
    t.start()
    started.wait(5)
    yield f"http://127.0.0.1:{port}"
    loop.call_soon_threadsafe(loop.stop)


@pytest.fixture
def replayed(tmp_path, fake_server):
    workload = tmp_path / "w.jsonl"
    make_workload.main(
        [
            "synthetic",
            "--qps",
            "40",
            "--num-requests",
            "30",
            "--seed",
            "0",
            "--max-total-tokens",
            "1024",
            "--out",
            str(workload),
        ]
    )
    run_dir = tmp_path / "run"
    replay.main(
        [
            "--workload",
            str(workload),
            "--out-dir",
            str(run_dir),
            "--base-url",
            fake_server,
            "--model",
            "fake",
            "--start-delay",
            "0.2",
            "--settle",
            "0.2",
        ]
    )
    experiment = tmp_path / "exp.json"
    experiment.write_text(
        json.dumps(
            {
                "name": "fake",
                "model": "meta-llama/Llama-2-7b-hf",
                "device": "a100",
                "tensor_parallel_size": 1,
                "max_model_len": 4096,
                "max_num_batched_tokens": 2048,
                "max_num_seqs": 128,
                "block_size": 16,
                "speculative_config": {"method": "ngram", "num_speculative_tokens": 4},
            }
        )
    )
    return tmp_path, workload, run_dir, experiment


def test_replay_records_every_request(replayed):
    _, workload, run_dir, _ = replayed
    measured = pd.read_csv(run_dir / "measured.csv")
    w = {r.request_index: r for r in read_workload(str(workload))}
    assert len(measured) == 30 and (measured["error"].fillna("") == "").all()
    for _, row in measured.iterrows():
        r = w[row["request_index"]]
        assert row["prompt_tokens"] == len(r.prompt_token_ids)
        assert row["completion_tokens"] == r.max_tokens
        assert 0 < row["ttft_s"] <= row["e2e_s"]
    server = json.loads((run_dir / "server_metrics.json").read_text())
    assert server["cache_config"]["num_gpu_blocks"] == "1234"
    assert 2.0 < server["mean_acceptance_length"] < 3.5


def test_chunks_carry_multiple_tokens(replayed):
    _, _, run_dir, _ = replayed
    sizes = [
        n
        for line in open(run_dir / "chunks.jsonl")
        for _, n in json.loads(line)["chunks"]
    ]
    assert max(sizes) > 1 and sum(sizes) > 0


def test_calibration_recovers_acceptance(replayed):
    tmp_path, _, run_dir, experiment = replayed
    out = tmp_path / "sim_inputs"
    build_sim_inputs.main(
        [
            "--experiment",
            str(experiment),
            "--run-dir",
            str(run_dir),
            "--out-dir",
            str(out),
        ]
    )
    cal = json.loads((out / "calibration.json").read_text())
    # the fake engine accepts each position with 0.7 independently
    assert np.mean(cal["acceptance_rate_per_position"]) == pytest.approx(0.7, abs=0.08)
    # the fake engine always proposes; end-of-request capping only nudges this
    assert cal["proposal_probability"] > 0.9
    trace = pd.read_csv(out / "trace.csv")
    assert list(trace.columns) == [
        "arrived_at",
        "num_prefill_tokens",
        "num_decode_tokens",
    ]
    assert trace["arrived_at"].is_monotonic_increasing
    args = json.loads((out / "sim_args.json").read_text())
    assert "--speculative_decoding_config_proposal_probability" in args
    assert args[args.index("--sarathi_scheduler_config_num_blocks") + 1] == "1234"


def test_report_is_zero_error_when_sim_matches(replayed):
    tmp_path, _, run_dir, experiment = replayed
    inputs = tmp_path / "sim_inputs"
    build_sim_inputs.main(
        [
            "--experiment",
            str(experiment),
            "--run-dir",
            str(run_dir),
            "--out-dir",
            str(inputs),
        ]
    )
    measured = report.measured_frame(str(run_dir), str(inputs))
    # a "simulation" that reproduces the measurement exactly
    result = tmp_path / "sim" / "perfect" / "result"
    result.mkdir(parents=True)
    pd.DataFrame(
        {
            "Request Id": measured["sim_request_id"],
            "prefill_e2e_time": measured["ttft"],
            "request_e2e_time": measured["e2e"],
            "request_num_decode_tokens": measured["completion_tokens"],
        }
    ).to_csv(result / "request_metrics.csv", index=False)
    (tmp_path / "sim" / "perfect" / "variant.json").write_text(
        json.dumps({"name": "perfect", "extra_args": [], "result_dir": str(result)})
    )
    report.main(
        [
            "--run-dir",
            str(run_dir),
            "--sim-inputs",
            str(inputs),
            "--sim-dir",
            str(tmp_path / "sim"),
            "--out-dir",
            str(tmp_path / "report"),
        ]
    )
    r = json.loads((tmp_path / "report" / "report.json").read_text())["results"][
        "perfect"
    ]
    for metric in ("ttft", "tpot", "e2e"):
        assert r[metric]["mape"] == pytest.approx(0, abs=1e-9)
    # sent_s vs trace arrival differ only by the shift to t=0
    assert r["output_throughput_tok_s"]["rel_error"] == pytest.approx(0, abs=1e-6)


@pytest.mark.skipif(
    not os.environ.get("VIDUR_CACHE_DIR"), reason="needs a trained Vidur cache"
)
def test_end_to_end_with_vidur(replayed):
    tmp_path, _, run_dir, experiment = replayed
    inputs, sim, rep = tmp_path / "sim_inputs", tmp_path / "sim", tmp_path / "report"
    build_sim_inputs.main(
        [
            "--experiment",
            str(experiment),
            "--run-dir",
            str(run_dir),
            "--out-dir",
            str(inputs),
        ]
    )
    subprocess.run(
        [
            sys.executable,
            os.path.join(VALIDATION, "run_sim.py"),
            "--sim-inputs",
            str(inputs),
            "--out-dir",
            str(sim),
            "--cache-dir",
            os.environ["VIDUR_CACHE_DIR"],
            "--weights",
            "0.5",
        ],
        cwd=REPO,
        check=True,
    )
    report.main(
        [
            "--run-dir",
            str(run_dir),
            "--sim-inputs",
            str(inputs),
            "--sim-dir",
            str(sim),
            "--out-dir",
            str(rep),
        ]
    )
    results = json.loads((rep / "report.json").read_text())["results"]
    assert set(results) == {"decode_shared_kv", "decode_per_token", "interpolated_w0.5"}
    for r in results.values():
        assert np.isfinite(r["e2e"]["p50"]["simulated"])
        assert r["sim_mean_acceptance_length"] > 1.5
