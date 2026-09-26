"""Falconws Lemma 18 comparison used by Falcon++ Section 6."""

from __future__ import annotations

from dataclasses import dataclass

import mpmath as mp

from .profiles import SecurityProfile
from .theta import theta3


@dataclass(frozen=True)
class FalconwsTrialEstimate:
    n: int
    epsilon: mp.mpf
    u_star: mp.mpf | None
    integral_exponent: mp.mpf
    published_li_exponent: mp.mpf
    log_trials: mp.mpf
    trials: mp.mpf
    finite_sum_diagnostic_trials: mp.mpf
    method: str
    epsilon_source: str
    heuristic: bool = True
    approximate_paper_integral: bool = True
    finite_sum_is_paper_lemma18: bool = False
    comparator_scope: str = "conservative_adapted_max_gso"


def falconws_same_point_epsilon(
    q: object,
    gamma: object,
    sigma_sig: object,
    *,
    dps: int = 100,
) -> tuple[mp.mpf, mp.mpf]:
    """Return ``u_star`` and ``theta3(u_star^2)-1`` from Section 6."""

    with mp.workdps(dps):
        modulus = mp.mpf(q)
        width = mp.mpf(gamma)
        sigma = mp.mpf(sigma_sig)
        if modulus <= 0 or width <= 0 or sigma <= 0:
            raise ValueError("q, gamma, and sigma_sig must be positive")
        u_star = mp.sqrt(2 * mp.pi) * sigma / (width * mp.sqrt(modulus))
        epsilon = theta3(u_star**2, dps=dps) - 1
        return +u_star, +epsilon


def _finite_sum_exponent(n: int, epsilon: mp.mpf) -> mp.mpf:
    return mp.fsum(
        (epsilon / 2) ** (mp.mpf(n) / (n - i)) for i in range(n // 2)
    )


def falconws_lemma18(
    n: int,
    epsilon: object,
    *,
    u_star: object | None = None,
    epsilon_source: str = "caller_supplied",
    dps: int = 100,
) -> FalconwsTrialEstimate:
    """Evaluate the published Lemma 18 integral approximation stably.

    The paper replaces a finite sum by an integral, so this result is a
    theoretical estimate rather than a certified trial bound.  The exact
    finite sum is returned only as a separately labelled diagnostic.
    """

    if n < 2 or n % 2:
        raise ValueError("n must be a positive even integer")
    with mp.workdps(dps):
        eps = mp.mpf(epsilon)
        if not (0 < eps <= mp.mpf("0.047")):
            raise ValueError("Falconws Lemma 18 assumes 0 < epsilon <= 0.047")
        a = eps / 2
        integral = n * mp.quad(lambda t: a**t / (t * t), [1, 2])
        # Algebraically equivalent form printed in Falconws.  mpmath.li stays
        # on the real branch for the arguments here (0 < a^2 < a < 1).
        li_form = (
            -n * eps**2 / 8
            + n * eps / 2
            + n * mp.log(a) * (mp.li(a**2) - mp.li(a))
        )
        log_trials = mp.mpf("8.4") * integral
        finite_log_trials = mp.mpf("8.4") * _finite_sum_exponent(n, eps)
        return FalconwsTrialEstimate(
            n=n,
            epsilon=+eps,
            u_star=None if u_star is None else +mp.mpf(u_star),
            integral_exponent=+integral,
            published_li_exponent=+li_form,
            log_trials=+log_trials,
            trials=+mp.exp(log_trials),
            finite_sum_diagnostic_trials=+mp.exp(finite_log_trials),
            method="falconws_lemma18_integral_form",
            epsilon_source=epsilon_source,
        )


def falconws_profile_trials(
    profile: SecurityProfile,
    *,
    dps: int = 100,
) -> FalconwsTrialEstimate:
    u_star, epsilon = falconws_same_point_epsilon(
        profile.q,
        profile.gamma,
        profile.sigma_sig,
        dps=dps,
    )
    return falconws_lemma18(
        profile.n,
        epsilon,
        u_star=u_star,
        epsilon_source="same_point_theta_tail",
        dps=dps,
    )


def falconws_original_trials(n: int, *, dps: int = 100) -> FalconwsTrialEstimate:
    """Evaluate Falconws' original ``epsilon=1/(2n)`` comparison point."""

    return falconws_lemma18(
        n,
        mp.mpf(1) / (2 * n),
        epsilon_source="paper_1_over_2n",
        dps=dps,
    )

