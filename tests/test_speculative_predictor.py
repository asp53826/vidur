"""Verify-step and proposer cost model, on a synthetic prediction table.

The real predictor trains random forests on profiling data; these tests swap
in a closed-form table so the speculative logic can be checked exactly.
"""

from types import SimpleNamespace

import pytest

from vidur.config import SpeculativeDecodingConfig
from vidur.entities import Batch, Request
from vidur.execution_time_predictor.random_forrest_execution_time_predictor import (
    RandomForrestExecutionTimePredictor,
)

MAX_BS, GRAN, MAX_KV = 32, 64, 4096
OVERHEAD = 0.1


def attn(bs, kv):
    return 0.02 + 1e-5 * bs * kv


def make_predictor(spec, draft=None):
    p = object.__new__(RandomForrestExecutionTimePredictor)
    p._config = SimpleNamespace(
        prediction_max_batch_size=MAX_BS,
        kv_cache_prediction_granularity=GRAN,
        prediction_max_tokens_per_request=MAX_KV,
    )
    p._spec_config = spec
    p._attention_decode_batching_overhead_fraction = OVERHEAD
    p._predictions = {
        "attn_decode": {
            (bs, kv): attn(bs, kv)
            for bs in range(1, MAX_BS + 1)
            for kv in range(0, MAX_KV + 1, GRAN)
        }
    }
    p._draft_predictor = draft
    return p


def decoding_request(processed, prefill=100, decode=1000):
    r = Request(arrived_at=0, num_prefill_tokens=prefill, num_decode_tokens=decode)
    r._is_prefill_complete = True
    r._num_processed_tokens = processed
    return r


def prefilling_request(processed, prefill=300):
    r = Request(arrived_at=0, num_prefill_tokens=prefill, num_decode_tokens=50)
    r._num_processed_tokens = processed
    return r


def spec(**kwargs):
    d = dict(num_speculative_tokens=4)
    d.update(kwargs)
    return SpeculativeDecodingConfig(**d)


def verify_batch(n_seqs=8, q=5, processed=256):
    requests = [decoding_request(processed) for _ in range(n_seqs)]
    return Batch(0, requests, [q] * n_seqs)


def with_overhead(bs, kv):
    return attn(bs, kv) * (1 + OVERHEAD * int(bs > 1))


def test_disabled_spec_matches_upstream_formula():
    p = make_predictor(SpeculativeDecodingConfig())
    batch = Batch(0, [decoding_request(256) for _ in range(8)], [1] * 8)
    assert p._get_attention_decode_execution_time(batch) == with_overhead(8, 256)


def test_lower_bound_ignores_extra_queries():
    p = make_predictor(spec(verify_attention_model="decode_shared_kv"))
    assert p._get_attention_decode_execution_time(verify_batch()) == with_overhead(8, 256)


def test_upper_bound_charges_every_query():
    p = make_predictor(spec(verify_attention_model="decode_per_token"))
    assert p._get_attention_decode_execution_time(verify_batch()) == pytest.approx(
        with_overhead(40, 256) if 40 <= MAX_BS else with_overhead(MAX_BS, 256) * 40 / MAX_BS
    )


@pytest.mark.parametrize("w", [0.0, 0.25, 1.0])
def test_interpolated_sits_between_bounds(w):
    lo = make_predictor(spec(verify_attention_model="decode_shared_kv"))
    hi = make_predictor(spec(verify_attention_model="decode_per_token"))
    mid = make_predictor(
        spec(verify_attention_model="interpolated", verify_attention_weight=w)
    )
    b = verify_batch()
    t_lo = lo._get_attention_decode_execution_time(b)
    t_hi = hi._get_attention_decode_execution_time(b)
    assert mid._get_attention_decode_execution_time(b) == pytest.approx(
        t_lo + w * (t_hi - t_lo)
    )


def test_extrapolates_past_profiled_batch_range():
    p = make_predictor(SpeculativeDecodingConfig())
    assert p._predict_decode_attention(2 * MAX_BS, 256) == pytest.approx(
        attn(MAX_BS, 256) * 2 * (1 + OVERHEAD)
    )


def test_ngram_proposer_cost():
    p = make_predictor(
        spec(proposer_fixed_overhead_ms=0.3, proposer_overhead_ms_per_seq=0.05)
    )
    batch = Batch(
        0,
        [decoding_request(200) for _ in range(6)] + [prefilling_request(0)],
        [5] * 6 + [128],
    )
    # the prefill chunk doesn't finish its prompt, so it doesn't draft
    assert p._get_speculative_proposer_time(batch) == pytest.approx(0.3 + 0.05 * 6)


def test_finishing_prefill_drafts():
    p = make_predictor(spec(proposer_overhead_ms_per_seq=1.0))
    batch = Batch(0, [prefilling_request(200, prefill=300)], [100])
    assert p._get_speculative_proposer_time(batch) == 1.0


def test_no_drafting_sequences_cost_nothing():
    p = make_predictor(spec(proposer_fixed_overhead_ms=5.0))
    batch = Batch(0, [prefilling_request(0)], [64])
    assert p._get_speculative_proposer_time(batch) == 0.0


class RecordingDraft:
    """Charges 1 ms per token and records each pass it is asked to price."""

    def __init__(self):
        self.passes = []

    def get_execution_time(self, batch, stage):
        self.passes.append(
            (list(batch.num_tokens), [r.num_processed_tokens for r in batch.requests])
        )
        return SimpleNamespace(model_time_ms=float(batch.total_num_tokens))


def test_draft_model_runs_k_passes():
    draft = RecordingDraft()
    p = make_predictor(
        spec(num_speculative_tokens=4, proposer="draft_model", draft_model_name="x"),
        draft=draft,
    )
    batch = Batch(
        0,
        [decoding_request(200), decoding_request(300), prefilling_request(0)],
        [5, 5, 64],
    )
    time = p._get_speculative_proposer_time(batch)
    assert len(draft.passes) == 4
    # pass 1 mirrors the target batch, prefill chunk included
    assert draft.passes[0][0] == [5, 5, 64]
    # passes 2..k: one token per drafting sequence. Pass 2 sees the context
    # plus the n tokens pass 1 consumed; each later pass adds one.
    assert [tokens for tokens, _ in draft.passes[1:]] == [[1, 1]] * 3
    assert [kv for _, kv in draft.passes[1:]] == [[205, 305], [206, 306], [207, 307]]
    assert time == pytest.approx(74 + 2 * 3)


def test_disabled_spec_has_no_proposer_time():
    p = make_predictor(SpeculativeDecodingConfig())
    assert p._get_speculative_proposer_time(verify_batch(q=1)) == 0.0
