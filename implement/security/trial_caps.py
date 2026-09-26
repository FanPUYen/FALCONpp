"""Binomial-tail and proof-cap utilities for Section 6."""

from __future__ import annotations

from dataclasses import dataclass

import mpmath as mp


@dataclass(frozen=True)
class BinomialTailResult:
    trials: int
    threshold: int
    probability: mp.mpf | None
    negative_log2: mp.mpf
    method: str
    exact: bool
    strict_target_met: bool | None = None


def binary_kl_bits(x: object, p: object) -> mp.mpf:
    """Binary KL divergence in bits, with continuous endpoint handling."""

    a = mp.mpf(x)
    b = mp.mpf(p)
    if not (0 <= a <= 1 and 0 <= b <= 1):
        raise ValueError("Bernoulli probabilities must lie in [0, 1]")
    if a == b:
        return mp.mpf("0")
    if b == 0:
        return mp.inf if a > 0 else mp.mpf("0")
    if b == 1:
        return mp.inf if a < 1 else mp.mpf("0")
    left = mp.mpf("0") if a == 0 else a * mp.log(a / b, 2)
    right = (
        mp.mpf("0")
        if a == 1
        else (1 - a) * mp.log((1 - a) / (1 - b), 2)
    )
    return left + right


def chernoff_lower_tail_bits(
    trials: int,
    success_probability: object,
    required_successes: int,
) -> mp.mpf:
    """Return the paper's KL lower-tail exponent.

    The event is ``Bin(trials,p) < required_successes``.  The bound applies
    only when ``(required_successes-1)/trials < p``.
    """

    if trials < 0 or required_successes < 0:
        raise ValueError("counts must be non-negative")
    if required_successes == 0:
        return mp.inf
    if trials < required_successes:
        return mp.mpf("0")
    p = mp.mpf(success_probability)
    if not 0 <= p <= 1:
        raise ValueError("success probability must lie in [0, 1]")
    x = mp.mpf(required_successes - 1) / trials
    if x >= p:
        raise ValueError("KL lower-tail condition (r-1)/N < p is not met")
    return trials * binary_kl_bits(x, p)


def _binomial_lower_tail_impl(
    trials: int,
    success_probability: object,
    required_successes: int,
    *,
    dps: int = 100,
    max_exact_trials: int = 1_000_000,
) -> BinomialTailResult:
    """Evaluate ``Pr[Bin(N,p) < r]`` exactly when computationally practical.

    Section 6 uses counts around 2^64.  Evaluating the regularized incomplete
    beta function at such parameters is neither necessary nor practical; for
    those inputs this function returns the explicit Chernoff--Hoeffding upper
    bound used by the paper and marks ``exact=False``.
    """

    if trials < 0 or required_successes < 0:
        raise ValueError("counts must be non-negative")
    p = mp.mpf(success_probability)
    if not 0 <= p <= 1:
        raise ValueError("success probability must lie in [0, 1]")
    if required_successes == 0:
        return BinomialTailResult(
            trials, required_successes, mp.mpf("0"), mp.inf,
            "empty_lower_tail", True,
        )
    if required_successes > trials:
        return BinomialTailResult(
            trials, required_successes, mp.mpf("1"), mp.mpf("0"),
            "certain_lower_tail", True,
        )
    if p == 0:
        return BinomialTailResult(
            trials, required_successes, mp.mpf("1"), mp.mpf("0"),
            "degenerate_p0", True,
        )
    if p == 1:
        return BinomialTailResult(
            trials, required_successes, mp.mpf("0"), mp.inf,
            "degenerate_p1", True,
        )

    with mp.workdps(dps):
        if trials <= max_exact_trials:
            k = required_successes - 1
            probability = mp.betainc(
                trials - k,
                k + 1,
                0,
                1 - p,
                regularized=True,
            )
            bits = mp.inf if probability == 0 else -mp.log(probability, 2)
            return BinomialTailResult(
                trials,
                required_successes,
                +probability,
                +bits,
                "regularized_incomplete_beta",
                True,
            )

        bits = chernoff_lower_tail_bits(trials, p, required_successes)
        probability = mp.power(2, -bits)
        return BinomialTailResult(
            trials,
            required_successes,
            +probability,
            +bits,
            "chernoff_hoeffding_binary_kl_upper_bound",
            False,
        )


def binomial_lower_tail(
    trials: int,
    success_probability: object,
    required_successes: int,
    *,
    dps: int = 100,
    max_exact_trials: int = 1_000_000,
) -> BinomialTailResult:
    """Precision-isolated wrapper for the binomial lower-tail evaluator."""

    if dps < 30:
        raise ValueError("binomial_lower_tail requires at least 30 decimal digits")
    with mp.workdps(dps):
        return _binomial_lower_tail_impl(
            trials,
            success_probability,
            required_successes,
            dps=dps,
            max_exact_trials=max_exact_trials,
        )


def proof_cap_check(
    *,
    displayed_strict_upper_cap: int,
    success_probability: object,
    required_successes: int,
    security_bits: int,
) -> BinomialTailResult:
    """Check the paper's strict ``argmin < cap`` claim at ``cap - 1``."""

    if displayed_strict_upper_cap <= 0:
        raise ValueError("displayed cap must be positive")
    checked = binomial_lower_tail(
        displayed_strict_upper_cap - 1,
        success_probability,
        required_successes,
    )
    return BinomialTailResult(
        trials=checked.trials,
        threshold=checked.threshold,
        probability=checked.probability,
        negative_log2=checked.negative_log2,
        method=checked.method,
        exact=checked.exact,
        strict_target_met=checked.negative_log2 > security_bits,
    )


def minimum_trials_chernoff(
    *,
    success_probability: object,
    required_successes: int,
    security_bits: object,
) -> int:
    """Find the least integer passing the paper's strict KL sufficient test.

    This is the minimum for the *Chernoff sufficient condition*, not the exact
    binomial-tail minimum.  The distinction matters and is preserved in the
    function name and documentation.
    """

    p = mp.mpf(success_probability)
    target = mp.mpf(security_bits)
    if not 0 < p <= 1:
        raise ValueError("success_probability must lie in (0, 1]")
    if required_successes < 1 or target < 0:
        raise ValueError("required_successes must be positive and target non-negative")
    if p == 1:
        return required_successes

    # First make the KL-domain condition strict.
    lower = max(
        required_successes,
        int(mp.floor(mp.mpf(required_successes - 1) / p)) + 1,
    )

    def passes(trials: int) -> bool:
        x = mp.mpf(required_successes - 1) / trials
        return x < p and chernoff_lower_tail_bits(
            trials, p, required_successes
        ) > target

    if passes(lower):
        return lower
    upper = lower
    while not passes(upper):
        upper *= 2
    while lower + 1 < upper:
        middle = (lower + upper) // 2
        if passes(middle):
            upper = middle
        else:
            lower = middle
    return upper
