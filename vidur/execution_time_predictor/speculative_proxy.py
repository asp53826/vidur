"""Minimal stand-ins for Batch and Request when pricing draft-model passes.

The execution time predictors only read a handful of attributes, so a draft
forward pass can be priced without creating real entities (which would take
ids from the global counters and perturb the simulation).
"""

from typing import List


class ProxyRequest:
    __slots__ = ("_is_prefill_complete", "num_processed_tokens")

    def __init__(self, is_prefill_complete: bool, num_processed_tokens: int) -> None:
        self._is_prefill_complete = is_prefill_complete
        self.num_processed_tokens = num_processed_tokens

    @property
    def is_prefill_complete(self) -> bool:
        return self._is_prefill_complete

    @classmethod
    def from_request(cls, request) -> "ProxyRequest":
        return cls(request._is_prefill_complete, request.num_processed_tokens)


class ProxyBatch:
    def __init__(self, requests: List[ProxyRequest], num_tokens: List[int]) -> None:
        assert len(requests) == len(num_tokens)
        self.requests = requests
        self.num_tokens = num_tokens
        self.total_num_tokens = sum(num_tokens)
        self._total_num_tokens_rounded = (self.total_num_tokens + 7) // 8 * 8

    @property
    def size(self) -> int:
        return len(self.requests)
