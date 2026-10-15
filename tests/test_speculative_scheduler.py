import random
from math import ceil

import pytest

from vidur.config import SpeculativeDecodingConfig
from tests.scheduler_harness import make_requests, make_scheduler, run_to_completion


def spec(k=4, alpha=0.7, **kwargs):
    return SpeculativeDecodingConfig(
        num_speculative_tokens=k, acceptance_rate=alpha, **kwargs
    )


def batch_trace(batches):
    return [(tuple(r.id - batches[0].requests[0].id for r in b.requests), tuple(b.num_tokens)) for b in batches]


def random_sizes(n, seed=0, max_prompt=900, max_decode=300):
    rng = random.Random(seed)
    return [(rng.randint(1, max_prompt), rng.randint(1, max_decode)) for _ in range(n)]


def test_disabled_spec_keeps_one_token_decodes():
    scheduler = make_scheduler()
    batches = run_to_completion(scheduler, make_requests(random_sizes(30)))
    for b in batches:
        assert b.num_tokens == b.num_advance_tokens
        for is_decode, n in zip(b._is_decode, b.num_tokens):
            if is_decode:
                assert n == 1
        assert b.num_speculative_tokens_drafted == 0


def test_every_request_completes_with_exact_token_count():
    requests = make_requests(random_sizes(40, seed=1))
    run_to_completion(make_scheduler(spec()), requests)
    for r in requests:
        assert r.completed
        assert r.num_processed_tokens == r.total_tokens


def test_full_acceptance_emits_k_plus_one_per_step():
    k = 4
    requests = make_requests([(100, 41)])
    batches = run_to_completion(make_scheduler(spec(k=k, alpha=1.0)), requests)
    decode_steps = [b for b in batches if b._is_decode[0]]
    emitted = [b.num_advance_tokens[0] for b in decode_steps]
    # prefill emits the first token, 40 remain: 8 steps of 5
    assert emitted == [k + 1] * 8
    assert all(b.num_tokens[0] == k + 1 for b in decode_steps)


def test_zero_acceptance_emits_one_but_still_computes_k_plus_one():
    k = 3
    requests = make_requests([(64, 11)])
    batches = run_to_completion(make_scheduler(spec(k=k, alpha=0.0)), requests)
    decode_steps = [b for b in batches if b._is_decode[0]]
    assert [b.num_advance_tokens[0] for b in decode_steps] == [1] * 10
    # proposals shrink only when the request is about to end
    assert [b.num_tokens[0] for b in decode_steps] == [4] * 7 + [3, 2, 1]


def test_proposals_never_run_past_request_end():
    requests = make_requests(random_sizes(30, seed=2, max_decode=12))
    def check(scheduler, batch):
        for r, n, decoding in zip(batch.requests, batch.num_tokens, batch._is_decode):
            if decoding:
                assert n <= r.total_tokens - r.num_processed_tokens
    run_to_completion(make_scheduler(spec(k=8, alpha=1.0)), requests, on_batch=check)


def test_token_budget_trims_proposals():
    # 16 decoding sequences x (k+1)=5 tokens would need 80; budget is 48
    requests = make_requests([(8, 200)] * 16)
    batches = run_to_completion(
        make_scheduler(spec(k=4, alpha=0.5), chunk_size=48), requests
    )
    for b in batches:
        assert b.total_num_tokens <= 48
    assert any(n < 5 for b in batches for n, d in zip(b.num_tokens, b._is_decode) if d)


def test_kv_slots_cover_every_verify_step():
    block_size = 16
    def check(scheduler, batch):
        assert scheduler.num_allocated_blocks <= scheduler._config.num_blocks
        for r, n, decoding in zip(batch.requests, batch.num_tokens, batch._is_decode):
            if decoding:
                needed = ceil((r.num_processed_tokens + n - 1) / block_size)
                assert scheduler._allocation_map[r.id] >= needed
    requests = make_requests(random_sizes(40, seed=3))
    run_to_completion(make_scheduler(spec(k=5)), requests, on_batch=check)


def test_memory_pressure_preempts_and_still_completes():
    # small cache forces restarts; lookahead slots make pressure worse
    requests = make_requests(random_sizes(40, seed=4, max_prompt=400, max_decode=400))
    scheduler = make_scheduler(spec(k=6, alpha=0.8), num_blocks=120)
    def check(sched, batch):
        assert sched.num_allocated_blocks <= 120
    run_to_completion(scheduler, requests, on_batch=check)
    assert all(r.completed for r in requests)
    assert any(r.num_restarts > 0 for r in requests)
    assert scheduler.num_allocated_blocks == 0


def test_spec_finishes_in_fewer_iterations():
    sizes = random_sizes(20, seed=5)
    base = run_to_completion(make_scheduler(), make_requests(sizes))
    fast = run_to_completion(make_scheduler(spec(k=4, alpha=0.8)), make_requests(sizes))
    assert len(fast) < len(base) * 0.6


def test_batch_counters_match_advance():
    requests = make_requests(random_sizes(20, seed=6))
    batches = run_to_completion(make_scheduler(spec(k=4, alpha=0.6)), requests)
    drafted = sum(b.num_speculative_tokens_drafted for b in batches)
    accepted = sum(b.num_speculative_tokens_accepted for b in batches)
    assert 0 < accepted < drafted
    decode_tokens = sum(
        a for b in batches for a, d in zip(b.num_advance_tokens, b._is_decode) if d
    )
    # every decode token after the first comes from a verify step
    assert decode_tokens == sum(r.num_decode_tokens - 1 for r in requests)


def test_unsupported_scheduler_refuses_spec():
    from vidur.config import (
        FixedRequestLengthGeneratorConfig,
        ReplicaConfig,
        SyntheticRequestGeneratorConfig,
        VllmSchedulerConfig,
    )
    from vidur.entities import Replica
    from vidur.scheduler.replica_scheduler.vllm_replica_scheduler import (
        VLLMReplicaScheduler,
    )

    replica_config = ReplicaConfig(speculative_decoding_config=spec())
    gen = SyntheticRequestGeneratorConfig(
        length_generator_config=FixedRequestLengthGeneratorConfig()
    )
    with pytest.raises(NotImplementedError):
        VLLMReplicaScheduler(
            replica_config=replica_config,
            replica_scheduler_config=VllmSchedulerConfig(),
            request_generator_config=gen,
            replica=Replica(replica_config, gen),
            num_stages=1,
            execution_time_predictor=None,
        )


def test_pipeline_parallel_rejected():
    from vidur.config import ReplicaConfig
    with pytest.raises(ValueError):
        ReplicaConfig(num_pipeline_stages=2, speculative_decoding_config=spec())


def test_draft_model_shrinks_kv_capacity():
    plain = make_scheduler()
    with_draft = make_scheduler(
        spec(proposer="draft_model", draft_model_name="meta-llama/Llama-2-7b-hf")
    )
    assert with_draft._config.num_blocks < plain._config.num_blocks
