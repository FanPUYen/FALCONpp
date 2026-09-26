"""Root-lattice theta evaluation for Falcon++ moment bounds.

The public high-precision routine delegates to the finite upper bound proved
in ``main.tex``.  A NumPy batch implementation is provided solely to screen
orders 2..1024 quickly; selected orders must be recomputed with the mpmath
routine before they are reported as high-precision values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import mpmath as mp

from .theta import ThetaAUpperBound, theta_a_upper


@dataclass(frozen=True)
class RootThetaBatch:
    orders: tuple[int, ...]
    log_values: tuple[tuple[float, ...], ...]
    truncation: int
    roots: int
    method: str = "finite_root_upper_float_screening"
    certified_high_precision: bool = False


@dataclass(frozen=True)
class RootThetaAggregate:
    log_product: mp.mpf
    factor_count: int
    truncation: int
    roots: int
    maximum_truncation_epsilon: mp.mpf
    method: str = "mpmath_finite_root_theta_upper_aggregate"
    formula_is_rigorous: bool = True
    machine_outward_rounded: bool = False


def log_root_theta_product_upper(
    t_values: Iterable[object],
    k: int,
    *,
    multiplicity: int = 1,
    dps: int = 100,
    truncation: int = 6,
    roots: int = 256,
) -> RootThetaAggregate:
    """Evaluate a product of root-theta upper bounds efficiently.

    Cosines are shared by every profile coordinate and the ``j`` versus
    ``R-j`` symmetry halves the work.  Every individual factor is still the
    finite upper bound in the paper's lemma.
    """

    if k < 2 or multiplicity < 1 or truncation < 0 or roots < 1:
        raise ValueError("invalid root-theta aggregate arguments")
    with mp.workdps(dps):
        ts = tuple(mp.mpf(value) for value in t_values)
        if any(not mp.isfinite(value) or value <= 0 for value in ts):
            raise ValueError("all theta arguments must be finite and positive")
        half = roots // 2
        js = range(half + 1) if roots % 2 == 0 else range(half + 1)
        cosine_table = tuple(
            tuple(mp.cos(2 * mp.pi * m * j / roots) for m in range(1, truncation + 1))
            for j in js
        )
        total_log = mp.mpf("0")
        max_epsilon = mp.mpf("0")
        for t in ts:
            weights = tuple(
                mp.exp(-mp.pi * t * m * m) for m in range(1, truncation + 1)
            )
            logs: list[mp.mpf] = []
            for j, cosine_row in enumerate(cosine_table):
                base = 1 + 2 * mp.fsum(
                    weight * cosine
                    for weight, cosine in zip(weights, cosine_row, strict=True)
                )
                if base <= 0:
                    raise ArithmeticError("finite theta base is non-positive")
                log_term = k * mp.log(base)
                is_endpoint = j == 0 or (roots % 2 == 0 and j == half)
                logs.append(log_term if is_endpoint else mp.log(2) + log_term)
            maximum = max(logs)
            log_average = maximum + mp.log(
                mp.fsum(mp.exp(value - maximum) for value in logs)
            ) - mp.log(roots)
            epsilon = 2 * k * mp.exp(
                -mp.pi * t * mp.mpf(k) / (k - 1) * (truncation + 1) ** 2
            )
            if epsilon >= 1:
                raise ArithmeticError("truncation epsilon is not below one")
            max_epsilon = max(max_epsilon, epsilon)
            total_log += multiplicity * (log_average - mp.log1p(-epsilon))
        return RootThetaAggregate(
            log_product=+total_log,
            factor_count=multiplicity * len(ts),
            truncation=truncation,
            roots=roots,
            maximum_truncation_epsilon=+max_epsilon,
        )


def root_theta_a_upper(
    t: object,
    k: int,
    *,
    dps: int = 100,
    tail_bits: int = 120,
    roots: int = 256,
    truncation: int | None = None,
) -> ThetaAUpperBound:
    """High-precision evaluation of the finite upper-bound formula."""

    return theta_a_upper(
        t,
        k,
        dps=dps,
        tail_bits=tail_bits,
        initial_roots=roots,
        alias_bits=max(40, tail_bits // 2),
        max_roots=roots,
        truncation=truncation,
    )


def screen_root_theta_logs(
    t_values: Iterable[object],
    orders: Iterable[int],
    *,
    truncation: int = 3,
    roots: int = 128,
) -> RootThetaBatch:
    """Vectorized finite-upper-bound screening in IEEE binary64.

    This is a speed optimization for finding candidate moment orders.  It uses
    exactly the finite expression from the paper, including the truncation
    denominator, but binary64 arithmetic is not an outward-rounded
    certificate.  Recompute selected candidates with :func:`root_theta_a_upper`.
    """

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise RuntimeError("NumPy is required for order screening") from exc

    ts = np.asarray([float(value) for value in t_values], dtype=np.float64)
    ks = np.asarray([int(value) for value in orders], dtype=np.int64)
    if ts.ndim != 1 or np.any(~np.isfinite(ts)) or np.any(ts <= 0):
        raise ValueError("all theta arguments must be finite and positive")
    if ks.ndim != 1 or np.any(ks < 2):
        raise ValueError("all moment orders must be at least two")
    if truncation < 0 or roots < 1:
        raise ValueError("invalid finite-evaluation controls")

    angles = 2 * np.pi * np.arange(roots, dtype=np.float64) / roots
    if truncation:
        ms = np.arange(1, truncation + 1, dtype=np.float64)
        weights = np.exp(-np.pi * ts[:, None] * ms[None, :] ** 2)
        cosines = np.cos(ms[:, None] * angles[None, :])
        bases = 1.0 + 2.0 * weights @ cosines
    else:
        bases = np.ones((len(ts), roots), dtype=np.float64)
    if np.any(bases <= 0):
        raise ArithmeticError("screening grid contains a non-positive base")
    log_bases = np.log(bases)

    rows: list[tuple[float, ...]] = []
    for k in ks:
        scaled = int(k) * log_bases
        maxima = np.max(scaled, axis=1)
        log_average = maxima + np.log(
            np.sum(np.exp(scaled - maxima[:, None]), axis=1)
        ) - math.log(roots)
        epsilon = 2.0 * int(k) * np.exp(
            -np.pi
            * ts
            * (float(k) / (int(k) - 1))
            * (truncation + 1) ** 2
        )
        if np.any(epsilon >= 1):
            raise ArithmeticError("truncation epsilon is not below one")
        log_upper = log_average - np.log1p(-epsilon)
        rows.append(tuple(float(value) for value in log_upper))
    return RootThetaBatch(
        orders=tuple(int(value) for value in ks),
        log_values=tuple(rows),
        truncation=truncation,
        roots=roots,
    )
