import numpy as np
import pytest

from vidur.config import SpeculativeDecodingConfig
from vidur.entities.speculative_decoding import (
    AcceptanceSampler,
    conditional_from_unconditional,
    expected_emitted_tokens,
)


def config(**kwargs):
    defaults = dict(num_speculative_tokens=4, acceptance_rate=0.7)
    defaults.update(kwargs)
    return SpeculativeDecodingConfig(**defaults)


@pytest.mark.parametrize("alpha", [0.0, 0.3, 0.7, 0.95])
@pytest.mark.parametrize("k", [1, 2, 4, 8])
def test_expected_tokens_matches_closed_form(alpha, k):
    closed_form = (1 - alpha ** (k + 1)) / (1 - alpha)
    assert expected_emitted_tokens([alpha] * k, k) == pytest.approx(closed_form)


def test_expected_tokens_bounds():
    assert expected_emitted_tokens([1.0] * 5, 5) == 6
    assert expected_emitted_tokens([0.0] * 5, 5) == 1


@pytest.mark.parametrize("alpha", [0.3, 0.7, 0.9])
def test_sampled_mean_matches_expectation(alpha):
    sampler = AcceptanceSampler(config(acceptance_rate=alpha, seed=7))
    draws = [sampler.sample_emitted(0, k=4, max_tokens=100) for _ in range(40_000)]
    expected = expected_emitted_tokens([alpha] * 4, 4)
    # 40k draws: standard error is well under 0.02 tokens
    assert np.mean(draws) == pytest.approx(expected, abs=0.03)
    assert min(draws) >= 1 and max(draws) <= 5


def test_sampled_distribution_is_truncated_geometric():
    alpha, k = 0.6, 3
    sampler = AcceptanceSampler(config(num_speculative_tokens=k, acceptance_rate=alpha))
    draws = np.array(
        [sampler.sample_emitted(0, k=k, max_tokens=100) for _ in range(60_000)]
    )
    for emitted in range(1, k + 1):
        # emitted = accepted + 1; exactly a accepted then a rejection
        a = emitted - 1
        assert np.mean(draws == emitted) == pytest.approx(
            alpha**a * (1 - alpha), abs=0.01
        )
    assert np.mean(draws == k + 1) == pytest.approx(alpha**k, abs=0.01)


def test_max_tokens_caps_emission():
    sampler = AcceptanceSampler(config(acceptance_rate=1.0))
    assert sampler.sample_emitted(0, k=4, max_tokens=3) == 3
    assert sampler.sample_emitted(0, k=4, max_tokens=100) == 5


def test_zero_proposals_emit_one_token():
    sampler = AcceptanceSampler(config(acceptance_rate=1.0))
    assert sampler.sample_emitted(0, k=0, max_tokens=10) == 1


def test_expected_mode_is_exact_in_the_long_run():
    sampler = AcceptanceSampler(config(acceptance_rate=0.7, sampling_mode="expected"))
    steps = 10_000
    total = sum(sampler.sample_emitted(0, k=4, max_tokens=100) for _ in range(steps))
    assert total / steps == pytest.approx(
        expected_emitted_tokens([0.7] * 4, 4), abs=1e-3
    )


def test_expected_mode_is_deterministic():
    a = AcceptanceSampler(config(sampling_mode="expected"))
    b = AcceptanceSampler(config(sampling_mode="expected", seed=999))
    seq_a = [a.sample_emitted(1, 4, 100) for _ in range(50)]
    seq_b = [b.sample_emitted(1, 4, 100) for _ in range(50)]
    assert seq_a == seq_b


def test_per_position_pads_with_last_value():
    sampler = AcceptanceSampler(
        config(
            num_speculative_tokens=5,
            acceptance_model="per_position",
            acceptance_rate_per_position=[0.9, 0.8],
        )
    )
    assert sampler.conditional_probs(0, 5) == [0.9, 0.8, 0.8, 0.8, 0.8]
    assert sampler.conditional_probs(0, 2) == [0.9, 0.8]


def test_beta_rates_are_per_request_and_reproducible():
    cfg = config(acceptance_model="beta", acceptance_rate=0.6, seed=3)
    a, b = AcceptanceSampler(cfg), AcceptanceSampler(cfg)
    rates_a = [a.conditional_probs(i, 1)[0] for i in range(2000)]
    rates_b = [b.conditional_probs(i, 1)[0] for i in range(2000)]
    assert rates_a == rates_b
    assert np.mean(rates_a) == pytest.approx(0.6, abs=0.02)
    assert np.std(rates_a) > 0.05
    # a request keeps its rate across steps
    assert a.conditional_probs(5, 3) == [rates_a[5]] * 3


def test_beta_concentration_controls_spread():
    tight = AcceptanceSampler(
        config(acceptance_model="beta", acceptance_beta_concentration=200)
    )
    loose = AcceptanceSampler(
        config(acceptance_model="beta", acceptance_beta_concentration=2)
    )
    spread_tight = np.std([tight.conditional_probs(i, 1)[0] for i in range(2000)])
    spread_loose = np.std([loose.conditional_probs(i, 1)[0] for i in range(2000)])
    assert spread_tight < spread_loose / 4


def test_conditional_from_unconditional():
    # vLLM style: fraction of drafts accepted through position i
    unconditional = [0.8, 0.6, 0.3]
    assert conditional_from_unconditional(unconditional) == pytest.approx(
        [0.8, 0.75, 0.5]
    )
    assert conditional_from_unconditional([0.5, 0.0, 0.0]) == [0.5, 0.0, 0.0]


def test_conditional_from_unconditional_rejects_increasing():
    with pytest.raises(ValueError):
        conditional_from_unconditional([0.5, 0.6])


def test_config_validation():
    with pytest.raises(ValueError):
        SpeculativeDecodingConfig(num_speculative_tokens=4, proposer="eagle")
    with pytest.raises(ValueError):
        SpeculativeDecodingConfig(num_speculative_tokens=4, acceptance_rate=1.5)
    with pytest.raises(ValueError):
        SpeculativeDecodingConfig(
            num_speculative_tokens=4, acceptance_model="per_position"
        )
    with pytest.raises(ValueError):
        SpeculativeDecodingConfig(num_speculative_tokens=4, proposer="draft_model")
    # disabled config doesn't demand draft settings
    SpeculativeDecodingConfig(num_speculative_tokens=0, proposer="draft_model")
