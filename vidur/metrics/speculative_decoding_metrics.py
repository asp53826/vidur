"""Per-request and aggregate speculative decoding statistics.

Written next to Vidur's other outputs so a run can be compared with the
acceptance metrics vLLM logs (mean acceptance length, acceptance rate).
"""

import json
import os
from collections import defaultdict
from typing import Dict

import pandas as pd

from vidur.entities import Batch


class SpeculativeDecodingMetrics:
    def __init__(self, num_speculative_tokens: int) -> None:
        self._k = num_speculative_tokens
        self._steps: Dict[int, int] = defaultdict(int)
        self._drafted: Dict[int, int] = defaultdict(int)
        self._accepted: Dict[int, int] = defaultdict(int)
        self._emitted: Dict[int, int] = defaultdict(int)
        self._num_batches = 0
        self._proposer_time_ms = 0.0

    def on_batch_end(self, batch: Batch, is_decode) -> None:
        self._num_batches += 1
        for request, decoding, n, advance in zip(
            batch.requests, is_decode, batch.num_tokens, batch.num_advance_tokens
        ):
            if not decoding:
                continue
            self._steps[request.id] += 1
            self._drafted[request.id] += n - 1
            self._accepted[request.id] += advance - 1
            self._emitted[request.id] += advance

    def on_proposer_time(self, time_ms: float) -> None:
        self._proposer_time_ms += time_ms

    def summary(self) -> dict:
        steps = sum(self._steps.values())
        drafted = sum(self._drafted.values())
        accepted = sum(self._accepted.values())
        emitted = sum(self._emitted.values())
        return {
            "num_speculative_tokens": self._k,
            "num_verify_steps": steps,
            "num_draft_tokens": drafted,
            "num_accepted_tokens": accepted,
            # same definitions vLLM uses in its SpecDecoding log line
            "draft_acceptance_rate": accepted / drafted if drafted else None,
            "mean_acceptance_length": emitted / steps if steps else None,
            "total_proposer_time_ms": self._proposer_time_ms,
        }

    def store(self, output_dir: str) -> None:
        rows = [
            {
                "request_id": request_id,
                "num_verify_steps": self._steps[request_id],
                "num_draft_tokens": self._drafted[request_id],
                "num_accepted_tokens": self._accepted[request_id],
                "mean_acceptance_length": self._emitted[request_id]
                / self._steps[request_id],
            }
            for request_id in sorted(self._steps)
        ]
        pd.DataFrame(rows).to_csv(
            os.path.join(output_dir, "speculative_decoding_metrics.csv"), index=False
        )
        with open(
            os.path.join(output_dir, "speculative_decoding_summary.json"), "w"
        ) as f:
            json.dump(self.summary(), f, indent=2)
