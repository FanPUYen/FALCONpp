"""High-precision Jacobi theta helpers.

The paper uses the convention

    theta3(t) = sum_{m in Z} exp(-pi * t * m**2).

This differs from ``mpmath.jtheta``'s public argument convention, so the
wrapper below takes the paper's positive real parameter directly.  The
modular identity is used for small arguments and all summation is performed in
an isolated ``mp.workdps`` context.
"""

from __future__ import annotations

from dataclasses import dataclass

import mpmath as mp


def _as_mpf(value: object) -> mp.mpf:
    return value if isinstance(value, mp.mpf) else mp.mpf(value)


def logsumexp(values: list[mp.mpf]) -> mp.mpf:
    """Return ``log(sum(exp(values)))`` without overflowing."""

    if not values:
        return mp.ninf
    maximum = max(values)
    if maximum == mp.ninf:
        return maximum
    return maximum + mp.log(mp.fsum(mp.exp(value - maximum) for value in values))


def theta3(t: object, *, dps: int = 100) -> mp.mpf:
    """Evaluate the paper-normalized Jacobi theta function.

    Parameters
    ----------
    t:
        Positive real argument in ``exp(-pi*t*m^2)``.
    dps:
        Decimal working precision.
    """

    with mp.workdps(dps):
        x = _as_mpf(t)
        if not mp.isfinite(x) or x <= 0:
            raise ValueError("theta3 requires a finite positive argument")
        # The modular transform keeps q = exp(-pi*t) away from one.
        if x < 1:
            return +(theta3(1 / x, dps=dps + 8) / mp.sqrt(x))
        q = mp.exp(-mp.pi * x)
        # jtheta is reliable in this range; log1p is used in log_theta3 for
        # the very large-t regime where the answer is extremely close to one.
        return +mp.jtheta(3, 0, q)


def log_theta3(t: object, *, dps: int = 100) -> mp.mpf:
    """Evaluate ``log(theta3(t))`` while retaining tiny deviations from one."""

    with mp.workdps(dps):
        x = _as_mpf(t)
        if not mp.isfinite(x) or x <= 0:
            raise ValueError("theta3 requires a finite positive argument")
        if x < 1:
            return +(-mp.log(x) / 2 + log_theta3(1 / x, dps=dps + 8))

        # Directly sum theta3-1.  Stopping uses an upper bound for the
        # remaining positive terms based on the ratio of consecutive terms.
        tolerance = mp.power(10, -(dps + 8))
        total = mp.mpf("0")
        m = 1
        while True:
            term = mp.exp(-mp.pi * x * m * m)
            total += 2 * term
            ratio = mp.exp(-mp.pi * x * (2 * m + 1))
            tail = 2 * term * ratio / (1 - ratio)
            if tail < tolerance:
                break
            m += 1
        return +mp.log1p(total)


@dataclass(frozen=True)
class ThetaAUpperBound:
    """Finite-evaluation result from Lemma ``wrapped-moment-upper-bound``."""

    log_value: mp.mpf
    truncation: int
    roots: int
    truncation_epsilon: mp.mpf
    alias_stability: mp.mpf | None = None
    formula_is_rigorous: bool = True
    machine_outward_rounded: bool = False

    @property
    def value(self) -> mp.mpf:
        return mp.exp(self.log_value)


def truncation_epsilon(t: object, k: int, truncation: int) -> mp.mpf:
    """Return epsilon_{k,M}(u) with ``t = u^2`` from the paper."""

    if k < 2:
        raise ValueError("k must be at least 2")
    if truncation < 0:
        raise ValueError("truncation must be non-negative")
    x = _as_mpf(t)
    if x <= 0:
        raise ValueError("t must be positive")
    return 2 * k * mp.exp(
        -mp.pi * x * mp.mpf(k) / (k - 1) * (truncation + 1) ** 2
    )


def choose_truncation(t: object, k: int, *, tail_bits: int = 120) -> int:
    """Choose the least M whose paper truncation epsilon is below 2^-bits."""

    if tail_bits <= 0:
        raise ValueError("tail_bits must be positive")
    target = mp.power(2, -tail_bits)
    truncation = 0
    while truncation_epsilon(t, k, truncation) >= target:
        truncation += 1
    return truncation


def _finite_root_log(
    t: mp.mpf,
    k: int,
    truncation: int,
    roots: int,
    epsilon: mp.mpf,
) -> mp.mpf:
    """Evaluate the logarithm of the finite upper bound at fixed M and R."""

    if roots < 1:
        raise ValueError("roots must be positive")
    if epsilon >= 1:
        raise ValueError("truncation epsilon must be less than one")

    weights = [mp.exp(-mp.pi * t * m * m) for m in range(1, truncation + 1)]
    positive_logs: list[mp.mpf] = []
    signed_terms: list[mp.mpf] = []
    all_positive = True
    for j in range(roots):
        angle = 2 * mp.pi * j / roots
        base = 1 + 2 * mp.fsum(
            weights[m - 1] * mp.cos(m * angle)
            for m in range(1, truncation + 1)
        )
        if base > 0:
            positive_logs.append(k * mp.log(base))
        else:
            all_positive = False
            signed_terms.append(base**k)

    if all_positive:
        log_average = logsumexp(positive_logs) - mp.log(roots)
    else:
        # This branch is not reached by the Falcon++ parameter profiles, but
        # keeps the public primitive mathematically well-defined for odd k.
        total = mp.fsum(mp.exp(item) for item in positive_logs) + mp.fsum(signed_terms)
        if total <= 0:
            raise ArithmeticError("root-of-unity average lost positivity")
        log_average = mp.log(total / roots)
    return log_average - mp.log1p(-epsilon)


def theta_a_upper(
    t: object,
    k: int,
    *,
    dps: int = 100,
    tail_bits: int = 120,
    initial_roots: int = 32,
    alias_bits: int = 70,
    max_roots: int = 4096,
    truncation: int | None = None,
) -> ThetaAUpperBound:
    """Evaluate the paper's rigorous finite-upper-bound formula.

    The returned value is an upper bound for every positive ``roots`` value by
    Lemma 9 of the paper.  The adaptive root count controls *tightness*, not
    validity: powers of two are doubled until consecutive logarithms agree to
    ``alias_bits`` or ``max_roots`` is reached.  ``alias_stability`` records the
    final observed difference and is therefore a numerical diagnostic, not a
    proof of the remaining aliasing error.  Ordinary ``mpmath`` arithmetic is
    not directed interval arithmetic, so the returned machine number is not
    labelled an outward-rounded numerical certificate.
    """

    if k < 2:
        raise ValueError("k must be at least 2")
    if initial_roots < 1 or max_roots < initial_roots:
        raise ValueError("invalid root-count bounds")
    with mp.workdps(dps):
        x = _as_mpf(t)
        if not mp.isfinite(x) or x <= 0:
            raise ValueError("t must be finite and positive")
        m = choose_truncation(x, k, tail_bits=tail_bits) if truncation is None else truncation
        epsilon = truncation_epsilon(x, k, m)
        if epsilon >= 1:
            raise ValueError("selected truncation does not satisfy epsilon < 1")

        roots = 1 << max(0, (initial_roots - 1).bit_length())
        previous: mp.mpf | None = None
        stability: mp.mpf | None = None
        target = mp.power(2, -alias_bits)
        while True:
            current = _finite_root_log(x, k, m, roots, epsilon)
            if previous is not None:
                stability = abs(current - previous)
                if stability <= target or roots >= max_roots:
                    break
            elif roots >= max_roots:
                break
            previous = current
            roots = min(2 * roots, max_roots)

        return ThetaAUpperBound(
            log_value=+current,
            truncation=m,
            roots=roots,
            truncation_epsilon=+epsilon,
            alias_stability=None if stability is None else +stability,
        )
