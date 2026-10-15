"""Shared pieces of the validation harness: workload I/O, Prometheus parsing,
and the latency definitions both sides (vLLM and Vidur) are scored with.

Kept free of GPU and vLLM imports so it runs and tests anywhere.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import numpy as np

# ---------------------------------------------------------------- workloads


@dataclass
class WorkloadRequest:
    request_index: int
    arrival_s: float
    max_tokens: int
    prompt_token_ids: Optional[List[int]] = None
    prompt: Optional[str] = None
    ignore_eos: bool = True

    def to_json(self) -> str:
        d = {
            "request_index": self.request_index,
            "arrival_s": self.arrival_s,
            "max_tokens": self.max_tokens,
            "ignore_eos": self.ignore_eos,
        }
        if self.prompt_token_ids is not None:
            d["prompt_token_ids"] = self.prompt_token_ids
        else:
            d["prompt"] = self.prompt
        return json.dumps(d)

    @classmethod
    def from_json(cls, line: str) -> "WorkloadRequest":
        d = json.loads(line)
        return cls(
            request_index=d["request_index"],
            arrival_s=d["arrival_s"],
            max_tokens=d["max_tokens"],
            prompt_token_ids=d.get("prompt_token_ids"),
            prompt=d.get("prompt"),
            ignore_eos=d.get("ignore_eos", True),
        )


def write_workload(path: str, requests: Iterable[WorkloadRequest]) -> None:
    with open(path, "w") as f:
        for r in requests:
            f.write(r.to_json() + "\n")


def read_workload(path: str) -> List[WorkloadRequest]:
    with open(path) as f:
        return [WorkloadRequest.from_json(line) for line in f if line.strip()]


# --------------------------------------------------------------- prometheus

_SAMPLE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(\{(?P<labels>[^}]*)\})?\s+(?P<value>\S+)"
)
_LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


@dataclass
class Sample:
    name: str
    labels: Dict[str, str]
    value: float


def parse_prometheus(text: str) -> List[Sample]:
    samples = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _SAMPLE.match(line)
        if not m:
            continue
        labels = dict(_LABEL.findall(m.group("labels") or ""))
        try:
            value = float(m.group("value"))
        except ValueError:
            continue
        samples.append(Sample(m.group("name"), labels, value))
    return samples


def _sum(samples: List[Sample], name: str) -> float:
    # counters are exported with a _total suffix; accept either spelling
    names = {name, name + "_total"}
    return sum(s.value for s in samples if s.name in names)


@dataclass
class SpecCounters:
    num_drafts: float = 0.0
    num_draft_tokens: float = 0.0
    num_accepted_tokens: float = 0.0
    accepted_per_pos: Dict[int, float] = field(default_factory=dict)

    @classmethod
    def from_samples(cls, samples: List[Sample]) -> "SpecCounters":
        per_pos: Dict[int, float] = {}
        for s in samples:
            if s.name in (
                "vllm:spec_decode_num_accepted_tokens_per_pos",
                "vllm:spec_decode_num_accepted_tokens_per_pos_total",
            ):
                pos = int(s.labels["position"])
                per_pos[pos] = per_pos.get(pos, 0.0) + s.value
        return cls(
            num_drafts=_sum(samples, "vllm:spec_decode_num_drafts"),
            num_draft_tokens=_sum(samples, "vllm:spec_decode_num_draft_tokens"),
            num_accepted_tokens=_sum(samples, "vllm:spec_decode_num_accepted_tokens"),
            accepted_per_pos=per_pos,
        )

    def minus(self, before: "SpecCounters") -> "SpecCounters":
        positions = set(self.accepted_per_pos) | set(before.accepted_per_pos)
        return SpecCounters(
            num_drafts=self.num_drafts - before.num_drafts,
            num_draft_tokens=self.num_draft_tokens - before.num_draft_tokens,
            num_accepted_tokens=self.num_accepted_tokens - before.num_accepted_tokens,
            accepted_per_pos={
                p: self.accepted_per_pos.get(p, 0.0) - before.accepted_per_pos.get(p, 0.0)
                for p in sorted(positions)
            },
        )

    def to_dict(self) -> dict:
        return {
            "num_drafts": self.num_drafts,
            "num_draft_tokens": self.num_draft_tokens,
            "num_accepted_tokens": self.num_accepted_tokens,
            "accepted_per_pos": {str(k): v for k, v in self.accepted_per_pos.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SpecCounters":
        return cls(
            num_drafts=d["num_drafts"],
            num_draft_tokens=d["num_draft_tokens"],
            num_accepted_tokens=d["num_accepted_tokens"],
            accepted_per_pos={int(k): v for k, v in d["accepted_per_pos"].items()},
        )

    # vLLM's own definitions (vllm/v1/spec_decode/metrics.py)
    @property
    def mean_acceptance_length(self) -> Optional[float]:
        if not self.num_drafts:
            return None
        return 1 + self.num_accepted_tokens / self.num_drafts

    @property
    def draft_acceptance_rate(self) -> Optional[float]:
        if not self.num_draft_tokens:
            return None
        return self.num_accepted_tokens / self.num_draft_tokens

    @property
    def mean_draft_length(self) -> Optional[float]:
        if not self.num_drafts:
            return None
        return self.num_draft_tokens / self.num_drafts

    def unconditional_acceptance(self) -> List[float]:
        """Fraction of drafts accepted through each position, as vLLM logs it."""
        if not self.num_drafts:
            return []
        return [
            self.accepted_per_pos[p] / self.num_drafts
            for p in sorted(self.accepted_per_pos)
        ]


def cache_config_from_samples(samples: List[Sample]) -> Dict[str, str]:
    """Labels of vllm:cache_config_info (num_gpu_blocks, block_size, ...)."""
    for s in samples:
        if s.name == "vllm:cache_config_info":
            return dict(s.labels)
    return {}


# ------------------------------------------------------------------ latency


def tpot(e2e_s: float, ttft_s: float, output_tokens: int) -> float:
    """Time per output token after the first, as vLLM's benchmarks define it."""
    if output_tokens <= 1:
        return float("nan")
    return (e2e_s - ttft_s) / (output_tokens - 1)


def percentile(values, q) -> float:
    values = np.asarray([v for v in values if np.isfinite(v)])
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, q))


def relative_error(predicted: float, measured: float) -> float:
    if not np.isfinite(predicted) or not np.isfinite(measured) or measured == 0:
        return float("nan")
    return (predicted - measured) / measured


def mape(predicted, measured) -> float:
    predicted = np.asarray(predicted, dtype=float)
    measured = np.asarray(measured, dtype=float)
    ok = np.isfinite(predicted) & np.isfinite(measured) & (measured != 0)
    if not ok.any():
        return float("nan")
    return float(np.mean(np.abs(predicted[ok] - measured[ok]) / measured[ok]))
