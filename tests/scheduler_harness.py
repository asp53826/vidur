"""Drive a replica scheduler without the event loop or a trained predictor.

Each iteration: schedule a batch, finish it, hand it back. Time advances by
one unit per iteration, which is all the scheduler logic needs.
"""

from vidur.config import (
    FixedRequestLengthGeneratorConfig,
    ReplicaConfig,
    SarathiSchedulerConfig,
    SpeculativeDecodingConfig,
    SyntheticRequestGeneratorConfig,
)
from vidur.entities import Replica, Request
from vidur.scheduler.replica_scheduler.sarathi_replica_scheduler import (
    SarathiReplicaScheduler,
)


def make_scheduler(
    spec=None, num_blocks=None, chunk_size=512, batch_size_cap=64, max_tokens=4096
):
    replica_config = ReplicaConfig(
        model_name="meta-llama/Llama-2-7b-hf",
        speculative_decoding_config=spec or SpeculativeDecodingConfig(),
    )
    generator_config = SyntheticRequestGeneratorConfig(
        length_generator_config=FixedRequestLengthGeneratorConfig(max_tokens=max_tokens)
    )
    scheduler_config = SarathiSchedulerConfig(
        chunk_size=chunk_size, batch_size_cap=batch_size_cap, num_blocks=num_blocks
    )
    replica = Replica(replica_config, generator_config)
    return SarathiReplicaScheduler(
        replica_config=replica_config,
        replica_scheduler_config=scheduler_config,
        request_generator_config=generator_config,
        replica=replica,
        num_stages=1,
        execution_time_predictor=None,
    )


def run_to_completion(scheduler, requests, max_iterations=100_000, on_batch=None):
    for request in requests:
        scheduler.add_request(request)

    batches = []
    time = 0.0
    for _ in range(max_iterations):
        if scheduler.is_empty():
            return batches
        scheduled = scheduler.on_schedule()
        assert scheduled, "scheduler stalled with work pending"
        for batch in scheduled:
            time += 1.0
            batch.on_schedule(time)
            if on_batch:
                on_batch(scheduler, batch)
            batch.on_batch_end(time)
            scheduler.on_batch_end(batch)
            batches.append(batch)
    raise AssertionError("did not finish")


def make_requests(sizes):
    return [
        Request(arrived_at=0.0, num_prefill_tokens=p, num_decode_tokens=d)
        for p, d in sizes
    ]
