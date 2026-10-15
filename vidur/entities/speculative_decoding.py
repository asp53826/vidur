"""Acceptance model for speculative decoding.

A verify step for one sequence scores ``k`` proposed tokens with the target
model. Proposals are accepted left to right until the first rejection, and the
target always contributes one more token (the correction at the rejected
position, or a bonus token if all ``k`` were accepted). So a step emits between
1 and ``k + 1`` tokens.

Writing ``p_i`` for the probability that position ``i`` is accepted given that
positions ``1..i-1`` were, the expected number of emitted tokens is

    E[emitted] = 1 + sum_{i=1..k} prod_{j=1..i} p_j

which for a constant rate ``alpha`` is the familiar ``(1 - alpha^(k+1)) / (1 - alpha)``
(Leviathan et al., 2023).

vLLM reports *unconditional* per-position acceptance (the fraction of drafts
whose position ``i`` was accepted). ``conditional_from_unconditional`` converts
that into the ``p_i`` this module expects.
"""

from typing import Dict, List, Sequence

import numpy as np

from vidur.config import SpeculativeDecodingConfig


def expected_emitted_tokens(conditional_probs: Sequence[float], k: int) -> float:
    """Expected tokens emitted by one verify step with ``k`` proposals."""
    expected = 1.0
    survival = 1.0
    for i in range(k):
        survival *= conditional_probs[i]
        expected += survival
    return expected


def conditional_from_unconditional(unconditional: Sequence[float]) -> List[float]:
    """Convert per-position acceptance as vLLM logs it into conditional rates.

    ``unconditional[i]`` is P(positions 1..i+1 all accepted). Positions after a
    zero are reported as 0, since nothing past a guaranteed rejection is
    ever accepted.
    """
    conditional = []
    previous = 1.0
    for value in unconditional:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"acceptance values must be in [0, 1], got {value}")
        if previous <= 0.0:
            conditional.append(0.0)
            continue
        if value > previous + 1e-9:
            raise ValueError(
                "unconditional acceptance must be non-increasing with position"
            )
        conditional.append(min(1.0, value / previous))
        previous = value
    return conditional


class AcceptanceSampler:
    """Decides how many tokens each verify step emits for each request."""

    def __init__(self, config: SpeculativeDecodingConfig) -> None:
        self._config = config
        self._k = config.num_speculative_tokens
        self._rng = np.random.default_rng(config.seed)
        self._request_rates: Dict[int, float] = {}
        self._carry: Dict[int, float] = {}

        if config.acceptance_model == "per_position":
            probs = list(config.acceptance_rate_per_position)
            probs += [probs[-1]] * max(0, self._k - len(probs))
            self._position_probs = probs[: self._k]
        else:
            self._position_probs = None

    @property
    def num_speculative_tokens(self) -> int:
        return self._k

    def has_proposal(self) -> bool:
        """Whether a decode step carries proposals (n-gram lookups can miss)."""
        p = self._config.proposal_probability
        if p >= 1.0:
            return True
        if p <= 0.0:
            return False
        return bool(self._rng.random() < p)

    def _request_rate(self, request_id: int) -> float:
        if request_id not in self._request_rates:
            mean = self._config.acceptance_rate
            if mean <= 0.0 or mean >= 1.0:
                rate = mean
            else:
                concentration = self._config.acceptance_beta_concentration
                rate = float(
                    self._rng.beta(mean * concentration, (1 - mean) * concentration)
                )
            self._request_rates[request_id] = rate
        return self._request_rates[request_id]

    def conditional_probs(self, request_id: int, k: int) -> List[float]:
        """Conditional acceptance probabilities for positions 1..k."""
        model = self._config.acceptance_model
        if model == "constant":
            return [self._config.acceptance_rate] * k
        if model == "per_position":
            return self._position_probs[:k]
        return [self._request_rate(request_id)] * k

    def expected_emitted(self, request_id: int, k: int) -> float:
        return expected_emitted_tokens(self.conditional_probs(request_id, k), k)

    def sample_emitted(self, request_id: int, k: int, max_tokens: int) -> int:
        """Tokens emitted by one verify step, never more than ``max_tokens``.

        ``k`` is the number of proposals actually verified this step (it can be
        below the configured k near the end of a request or when the token
        budget is tight).
        """
        if max_tokens < 1:
            raise ValueError("a verify step always emits at least one token")
        if k == 0:
            return 1

        probs = self.conditional_probs(request_id, k)

        if self._config.sampling_mode == "expected":
            # Emit the expected count, carrying the fractional part forward so
            # the long-run tokens per step is exact and runs stay deterministic.
            total = expected_emitted_tokens(probs, k) + self._carry.get(request_id, 0.0)
            emitted = int(total)
            self._carry[request_id] = total - emitted
            emitted = max(1, min(emitted, k + 1))
        else:
            accepted = 0
            for p in probs:
                if self._rng.random() >= p:
                    break
                accepted += 1
            emitted = accepted + 1

        return min(emitted, max_tokens)

    def forget(self, request_id: int) -> None:
        """Drop per-request state once a request completes."""
        self._request_rates.pop(request_id, None)
        self._carry.pop(request_id, None)
