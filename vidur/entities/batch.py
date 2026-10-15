from typing import List, Optional

from vidur.entities.base_entity import BaseEntity
from vidur.entities.request import Request
from vidur.logger import init_logger

logger = init_logger(__name__)


# a decorator which checks if the request has been scheduled
def check_scheduled(func):
    def wrapper(self, *args, **kwargs):
        if not self._scheduled:
            raise ValueError("Batch has not been scheduled yet")
        return func(self, *args, **kwargs)

    return wrapper


def check_completed(func):
    def wrapper(self, *args, **kwargs):
        if not self._completed:
            raise ValueError("Batch has not been scheduled yet")
        return func(self, *args, **kwargs)

    return wrapper


class Batch(BaseEntity):
    def __init__(
        self,
        replica_id: int,
        requests: List[Request],
        num_tokens: List[int],
        num_advance_tokens: Optional[List[int]] = None,
    ) -> None:
        """
        ``num_tokens`` is what each request feeds through the model this
        iteration and drives the cost model. ``num_advance_tokens`` is how far
        each request's progress moves when the iteration ends. They are equal
        unless speculative decoding is on, where a verify step computes k + 1
        tokens but keeps only the accepted prefix plus one.
        """
        self._id = Batch.generate_id()
        self._replica_id = replica_id

        self._requests = requests
        self._num_tokens = num_tokens
        if num_advance_tokens is None:
            num_advance_tokens = num_tokens
        assert len(num_advance_tokens) == len(num_tokens)
        assert all(a <= t for a, t in zip(num_advance_tokens, num_tokens))
        self._num_advance_tokens = num_advance_tokens
        # captured now: a request whose prefill finishes in this batch flips
        # is_prefill_complete in on_batch_end
        self._is_decode = [r.is_prefill_complete for r in requests]
        self._total_num_tokens = sum(num_tokens)
        self._num_prefill_tokens = sum(
            [
                (t if not r.is_prefill_complete else 0)
                for r, t in zip(self.requests, self._num_tokens)
            ]
        )

        self._total_num_tokens_rounded = (self._total_num_tokens + 7) // 8 * 8

        self._scheduled_at = None
        self._completed_at = None
        self._scheduled = False
        self._completed = False

    @property
    def replica_id(self) -> int:
        return self._replica_id

    @property
    def creation_time(self) -> float:
        return self._creation_time

    @property
    def num_tokens(self) -> List[int]:
        return self._num_tokens

    @property
    def num_advance_tokens(self) -> List[int]:
        return self._num_advance_tokens

    @property
    def num_speculative_tokens_drafted(self) -> int:
        """Proposals verified this iteration (0 without speculative decoding)."""
        return sum(
            t - 1 for is_decode, t in zip(self._is_decode, self._num_tokens) if is_decode
        )

    @property
    def num_speculative_tokens_accepted(self) -> int:
        """Proposals accepted this iteration (0 without speculative decoding)."""
        return sum(
            a - 1
            for is_decode, a in zip(self._is_decode, self._num_advance_tokens)
            if is_decode
        )

    @property
    def total_num_tokens(self) -> int:
        return self._total_num_tokens

    @property
    def num_prefill_tokens(self) -> int:
        return self._num_prefill_tokens

    @property
    def num_decode_tokens(self) -> int:
        return self.total_num_tokens - self.num_prefill_tokens

    @property
    @check_scheduled
    def scheduled_at(self) -> float:
        return self._scheduled_at

    @property
    @check_completed
    def completed_at(self) -> float:
        return self._completed_at

    @property
    def completed(self) -> bool:
        return self._completed

    @property
    def scheduled(self) -> bool:
        return self._scheduled

    @property
    def size(self) -> int:
        return len(self._requests)

    @property
    def requests(self) -> List[Request]:
        return self._requests

    @property
    def request_ids(self) -> List[int]:
        return [request.id for request in self._requests]

    @property
    def all_requests_completed(self) -> bool:
        return all([request.completed for request in self._requests])

    def on_schedule(
        self,
        time: float,
    ) -> None:
        self._scheduled_at = time
        self._scheduled = True

        for request in self._requests:
            request.on_batch_schedule(time)

    def on_batch_end(self, time: float):
        self._completed = True
        self._completed_at = time

        for request, num_tokens in zip(self._requests, self._num_advance_tokens):
            request.on_batch_end(time, num_tokens)

    @property
    def preempted_requests(self) -> List[Request]:
        return [request for request in self._requests if request.preempted]

    @property
    def completed_requests(self) -> List[Request]:
        return [request for request in self._requests if request.completed]

    def to_dict(self) -> dict:
        return {
            "id": self._id,
            "size": self.size,
            "replica_id": self._replica_id,
            "scheduled_at": self._scheduled_at,
            "completed_at": self._completed_at,
            "scheduled": self._scheduled,
            "request_ids": self.request_ids,
            "num_tokens": self._num_tokens,
            "num_advance_tokens": self._num_advance_tokens,
            "num_prefill_tokens": self.num_prefill_tokens,
            "num_decode_tokens": self.num_decode_tokens,
        }
