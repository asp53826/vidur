"""A stand-in for a vLLM server, for testing the harness without a GPU.

Implements just enough of the OpenAI completions API (streaming, per-chunk
usage, ignore_eos) and /metrics (spec-decode counters, cache_config_info)
for replay.py to exercise every code path. Each "step" emits a burst of
tokens drawn from a fixed acceptance rate, like a speculative engine would.
"""

import argparse
import asyncio
import json
import random

from aiohttp import web


class FakeEngine:
    def __init__(self, k, alpha, step_s, prefill_s_per_token, seed=0):
        self.k = k
        self.alpha = alpha
        self.step_s = step_s
        self.prefill_s_per_token = prefill_s_per_token
        self.rng = random.Random(seed)
        self.num_drafts = 0
        self.num_draft_tokens = 0
        self.num_accepted = 0
        self.per_pos = [0] * k

    def step_tokens(self, remaining):
        if self.k == 0:
            return 1
        accepted = 0
        while accepted < self.k and self.rng.random() < self.alpha:
            accepted += 1
        self.num_drafts += 1
        self.num_draft_tokens += self.k
        self.num_accepted += accepted
        for i in range(accepted):
            self.per_pos[i] += 1
        return min(accepted + 1, remaining)


async def completions(request):
    engine: FakeEngine = request.app["engine"]
    body = await request.json()
    prompt = body["prompt"]
    prompt_tokens = len(prompt) if isinstance(prompt, list) else len(prompt.split())
    max_tokens = body["max_tokens"]

    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await resp.prepare(request)

    await asyncio.sleep(prompt_tokens * engine.prefill_s_per_token)
    produced = 0
    first = True
    while produced < max_tokens:
        n = 1 if first else engine.step_tokens(max_tokens - produced)
        first = False
        produced += n
        event = {
            "choices": [{"index": 0, "text": " tok" * n}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": produced},
        }
        await resp.write(f"data: {json.dumps(event)}\n\n".encode())
        if produced < max_tokens:
            await asyncio.sleep(engine.step_s)
    final = {
        "choices": [],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": produced},
    }
    await resp.write(f"data: {json.dumps(final)}\n\n".encode())
    await resp.write(b"data: [DONE]\n\n")
    return resp


async def metrics(request):
    e: FakeEngine = request.app["engine"]
    lines = [
        "# HELP vllm:cache_config_info Information of the LLMEngine CacheConfig",
        "# TYPE vllm:cache_config_info gauge",
        'vllm:cache_config_info{block_size="16",engine="0",num_gpu_blocks="1234"} 1.0',
    ]
    if e.k:
        lines += [
            f'vllm:spec_decode_num_drafts_total{{engine="0",model_name="fake"}} {e.num_drafts}.0',
            f'vllm:spec_decode_num_draft_tokens_total{{engine="0",model_name="fake"}} {e.num_draft_tokens}.0',
            f'vllm:spec_decode_num_accepted_tokens_total{{engine="0",model_name="fake"}} {e.num_accepted}.0',
        ]
        lines += [
            f'vllm:spec_decode_num_accepted_tokens_per_pos_total{{engine="0",model_name="fake",position="{i}"}} {c}.0'
            for i, c in enumerate(e.per_pos)
        ]
    return web.Response(text="\n".join(lines) + "\n")


def make_app(k=4, alpha=0.7, step_s=0.002, prefill_s_per_token=0.00001, seed=0):
    app = web.Application()
    app["engine"] = FakeEngine(k, alpha, step_s, prefill_s_per_token, seed)
    app.router.add_post("/v1/completions", completions)
    app.router.add_get("/metrics", metrics)
    app.router.add_get("/health", lambda r: web.Response(text="ok"))
    return app


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--alpha", type=float, default=0.7)
    a = p.parse_args()
    web.run_app(make_app(a.k, a.alpha), port=a.port)
