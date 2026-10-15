"""Replay a workload against a running vLLM OpenAI-compatible server.

Each request is sent at its scheduled arrival time (open loop, so a slow
server builds a queue instead of slowing the client down). Streaming with
per-chunk usage gives the first-token time and how many tokens each chunk
carried, which with speculative decoding is roughly the tokens one verify
step emitted.

The server's /metrics is scraped before and after, so the speculative
decoding counters (drafts, drafted and accepted tokens, accepted per
position) cover exactly this run, along with the KV cache size vLLM chose.

Outputs in --out-dir:
  measured.csv         one row per request
  chunks.jsonl         per-request chunk timestamps and token counts
  server_metrics.json  spec-decode counter deltas and cache config
  replay_meta.json     client settings and send-lag statistics
"""

import argparse
import asyncio
import json
import os
import sys
import time
from typing import List

import aiohttp
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    SpecCounters,
    WorkloadRequest,
    cache_config_from_samples,
    parse_prometheus,
    read_workload,
)


async def scrape_metrics(session, base_url):
    async with session.get(f"{base_url}/metrics") as resp:
        resp.raise_for_status()
        return parse_prometheus(await resp.text())


async def send_one(
    session, base_url, model, req: WorkloadRequest, t0, temperature, timeout
):
    payload = {
        "model": model,
        "prompt": (
            req.prompt_token_ids if req.prompt_token_ids is not None else req.prompt
        ),
        "max_tokens": req.max_tokens,
        "temperature": temperature,
        "ignore_eos": req.ignore_eos,
        "stream": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
    }
    row = {
        "request_index": req.request_index,
        "scheduled_s": req.arrival_s,
        "sent_s": None,
        "ttft_s": float("nan"),
        "e2e_s": float("nan"),
        "prompt_tokens": None,
        "completion_tokens": None,
        "num_chunks": 0,
        "error": "",
    }
    chunks = []
    sent = time.perf_counter()
    row["sent_s"] = sent - t0
    last_completion = 0
    try:
        async with session.post(
            f"{base_url}/v1/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            if resp.status != 200:
                row["error"] = f"HTTP {resp.status}: {(await resp.text())[:200]}"
                return row, chunks
            buffer = b""
            async for data in resp.content.iter_any():
                buffer += data
                while b"\n\n" in buffer:
                    message, buffer = buffer.split(b"\n\n", 1)
                    message = message.decode().strip()
                    if not message.startswith("data:"):
                        continue
                    body = message[len("data:") :].strip()
                    if body == "[DONE]":
                        continue
                    event = json.loads(body)
                    now = time.perf_counter()
                    usage = event.get("usage") or {}
                    if usage:
                        row["prompt_tokens"] = usage.get("prompt_tokens")
                        row["completion_tokens"] = usage.get("completion_tokens")
                    if event.get("choices"):
                        completion = usage.get("completion_tokens", last_completion + 1)
                        new_tokens = completion - last_completion
                        last_completion = completion
                        if new_tokens <= 0:
                            continue
                        if not chunks:
                            row["ttft_s"] = now - sent
                        chunks.append((now - sent, new_tokens))
        if chunks:
            row["e2e_s"] = chunks[-1][0]
        else:
            row["error"] = "no tokens received"
    except Exception as e:  # keep going; one failure shouldn't sink the run
        row["error"] = f"{type(e).__name__}: {e}"
    row["num_chunks"] = len(chunks)
    return row, chunks


async def replay(args, workload: List[WorkloadRequest]):
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        before = await scrape_metrics(session, args.base_url)

        t0 = time.perf_counter() + args.start_delay
        tasks = []
        for req in workload:
            delay = t0 + req.arrival_s - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(
                asyncio.create_task(
                    send_one(
                        session,
                        args.base_url,
                        args.model,
                        req,
                        t0,
                        args.temperature,
                        args.timeout,
                    )
                )
            )
        results = await asyncio.gather(*tasks)

        # counters are updated when the engine reports stats; give it a beat
        await asyncio.sleep(args.settle)
        after = await scrape_metrics(session, args.base_url)

    return results, before, after


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--workload", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--model", required=True)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--timeout", type=float, default=1800.0)
    p.add_argument("--start-delay", type=float, default=1.0)
    p.add_argument("--settle", type=float, default=6.0)
    args = p.parse_args(argv)

    workload = read_workload(args.workload)
    results, before, after = asyncio.run(replay(args, workload))
    os.makedirs(args.out_dir, exist_ok=True)

    rows = [r for r, _ in results]
    df = pd.DataFrame(rows).sort_values("request_index")
    df.to_csv(os.path.join(args.out_dir, "measured.csv"), index=False)

    with open(os.path.join(args.out_dir, "chunks.jsonl"), "w") as f:
        for row, chunks in results:
            f.write(
                json.dumps({"request_index": row["request_index"], "chunks": chunks})
                + "\n"
            )

    spec = SpecCounters.from_samples(after).minus(SpecCounters.from_samples(before))
    server = {
        "spec_decode": spec.to_dict(),
        "mean_acceptance_length": spec.mean_acceptance_length,
        "draft_acceptance_rate": spec.draft_acceptance_rate,
        "mean_draft_length": spec.mean_draft_length,
        "cache_config": cache_config_from_samples(after),
    }
    with open(os.path.join(args.out_dir, "server_metrics.json"), "w") as f:
        json.dump(server, f, indent=2)

    lag = (df["sent_s"] - df["scheduled_s"]).to_numpy()
    meta = {
        "workload": os.path.abspath(args.workload),
        "model": args.model,
        "temperature": args.temperature,
        "num_requests": len(df),
        "num_failed": int((df["error"].fillna("") != "").sum()),
        "send_lag_p50_ms": float(np.percentile(lag, 50) * 1e3),
        "send_lag_max_ms": float(lag.max() * 1e3),
    }
    with open(os.path.join(args.out_dir, "replay_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(json.dumps(meta, indent=2))
    if meta["num_failed"]:
        print(
            f"WARNING: {meta['num_failed']} requests failed; see measured.csv",
            file=sys.stderr,
        )
    if meta["send_lag_max_ms"] > 50:
        print(
            "WARNING: client fell behind the arrival schedule by >50 ms",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
