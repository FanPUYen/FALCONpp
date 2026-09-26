"""High-precision clipped KGPV correction for Falcon++.

The paper uses the Gaussian notation

    rho_{s,c}(Z) = sum_{z in Z} exp(-pi * (z - c)^2 / s^2),

whereas :mod:`falconpp.gaussian` and the sampling traces use the usual
standard-deviation convention ``exp(-(z-c)^2 / (2*sigma^2))``.  Consequently
``s = sqrt(2*pi) * sigma``.  Keeping that conversion in this module avoids a
particularly easy (and large) normalization error in the correction factor.

For a Klein trace with conditional centres ``c_i`` and local widths
``sigma_i``, this module evaluates

    Delta_B(x) = product_i rho_{s_i,c_i}(Z) / rho_{s_i,0}(Z)

in the log domain.  The sampled integer itself does not enter this expression;
it remains part of the trace because it is useful for audits and A/B checks.

This is research code.  It deliberately returns ``mpmath.mpf`` values instead
of silently rounding security-relevant quantities to binary64.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from decimal import (
    Decimal,
    MAX_EMAX,
    MIN_EMIN,
    ROUND_CEILING,
    ROUND_FLOOR,
    ROUND_HALF_EVEN,
    localcontext,
)
from fractions import Fraction
from functools import lru_cache
import math
import secrets
from time import perf_counter
from typing import Any

import mpmath as mp

from .gaussian import (
    bernoulli_from_log_bounds,
    bernoulli_from_probability_bounds,
    exact_real_fraction,
)


DEFAULT_DPS = 100
DEFAULT_ACCEPTANCE_BITS = 256


# Values fixed by Table 6 of the current Falcon++ manuscript.  String values
# are intentional: constructing these constants from binary64 would lose the
# decimal values printed by the paper before high-precision work even starts.
GLOBAL_CORRECTION_MULTIPLIERS: dict[tuple[int, int, str], str] = {
    (512, 953, "1.17"): "4.533254",
    (512, 953, "1.25"): "1.774456",
    (1024, 1949, "1.17"): "9.067426",
    (1024, 1949, "1.25"): "3.714950",
}

_NAME_ALIASES: dict[str, tuple[int, int, str]] = {
    "i-117": (512, 953, "1.17"),
    "i_117": (512, 953, "1.17"),
    "512-953-117": (512, 953, "1.17"),
    "n512-q953-gamma117": (512, 953, "1.17"),
    "falconpp-512-953-gamma117": (512, 953, "1.17"),
    "i-125": (512, 953, "1.25"),
    "i_125": (512, 953, "1.25"),
    "512-953-125": (512, 953, "1.25"),
    "n512-q953-gamma125": (512, 953, "1.25"),
    "falconpp-512-953-gamma125": (512, 953, "1.25"),
    "v-117": (1024, 1949, "1.17"),
    "v_117": (1024, 1949, "1.17"),
    "1024-1949-117": (1024, 1949, "1.17"),
    "n1024-q1949-gamma117": (1024, 1949, "1.17"),
    "falconpp-1024-1949-gamma117": (1024, 1949, "1.17"),
    "v-125": (1024, 1949, "1.25"),
    "v_125": (1024, 1949, "1.25"),
    "1024-1949-125": (1024, 1949, "1.25"),
    "n1024-q1949-gamma125": (1024, 1949, "1.25"),
    "falconpp-1024-1949-gamma125": (1024, 1949, "1.25"),
}

@dataclass(frozen=True, slots=True)
class CorrectionTraceEntry:
    """One univariate decision made by a Klein/ffSampling run.

    ``sigma`` is a standard deviation, not the paper's rho parameter.  A
    producer may cache the two logarithmic normalizers.  In that case
    ``log_mass_centered`` means ``log rho_{s,center}(Z)`` and
    ``log_mass_zero`` means ``log rho_{s,0}(Z)``.
    """

    index: int
    center: Any
    sigma: Any
    value: int | None = None
    gs_norm: Any | None = None
    log_mass_centered: Any | None = None
    log_mass_zero: Any | None = None
    path: Any | None = None


@dataclass(frozen=True, slots=True)
class CorrectionDecision:
    """Auditable result of one clipped-correction decision."""

    accepted: bool
    log_delta: mp.mpf | None
    log_probability: mp.mpf | None
    probability: mp.mpf | None
    multiplier: mp.mpf
    correction_backend: str = "primal"
    diagnostic_level: str = "full"
    metrics: dict[str, int | float] = field(default_factory=dict)


class CorrectionError(ValueError):
    """Raised when a trace or correction parameter is invalid."""


_MISSING = object()


# The acceptance decision below is exact in the real-arithmetic model.  All
# externally supplied finite scalars are first frozen as rationals, and every
# transcendental operation is enclosed by rational endpoints.  Decimal's
# exp() and ln() are correctly rounded; stepping once outwards from the
# rounded result turns that guarantee into a directed enclosure.
_LOG10_2 = math.log10(2.0)
_MAX_CERTIFIED_RADIUS = 1_000_000


def _digits_for_bits(bits: int, *, guard_digits: int = 32) -> int:
    return max(50, math.ceil(max(1, bits) * _LOG10_2) + guard_digits)


def _exact_fraction(value: Any, *, name: str) -> Fraction:
    """Use the exact scalar semantics shared with the Gaussian sampler."""

    try:
        return exact_real_fraction(value, name=name)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CorrectionError(f"{name} must be a finite real number") from exc


def _fraction_to_decimal_bound(
    value: Fraction, *, digits: int, rounding: str
) -> Decimal:
    with localcontext() as context:
        context.prec = digits
        context.rounding = rounding
        context.Emax = MAX_EMAX
        context.Emin = MIN_EMIN
        return +(Decimal(value.numerator) / Decimal(value.denominator))


def _exp_decimal_bounds(
    lower: Fraction,
    upper: Fraction,
    *,
    bits: int,
) -> tuple[Decimal, Decimal]:
    """Return an outward Decimal enclosure of ``exp`` on an interval."""

    if lower > upper:
        raise ArithmeticError("inverted exponential argument interval")
    if lower == upper == 0:
        one = Decimal(1)
        return one, one
    digits = _digits_for_bits(bits)
    argument_lower = _fraction_to_decimal_bound(
        lower, digits=digits, rounding=ROUND_FLOOR
    )
    argument_upper = _fraction_to_decimal_bound(
        upper, digits=digits, rounding=ROUND_CEILING
    )
    with localcontext() as context:
        context.prec = digits
        context.rounding = ROUND_HALF_EVEN
        context.Emax = MAX_EMAX
        context.Emin = MIN_EMIN
        rounded_lower = argument_lower.exp()
        rounded_upper = argument_upper.exp()
        value_lower = (
            Decimal(0)
            if rounded_lower == 0
            else rounded_lower.next_minus(context=context)
        )
        # A strictly positive exponential rounded all the way to Decimal zero
        # lies outside the representable range of a practical Fraction too.
        # Fail promptly instead of turning 0.next_plus at MIN_EMIN into a
        # denominator with roughly 10^18 zeroes.
        if rounded_upper == 0:
            raise CorrectionError(
                "Gaussian exponent lies outside the certified Decimal range"
            )
        value_upper = rounded_upper.next_plus(context=context)
    # All uses in this module have non-positive exponents.  Intersecting with
    # the analytic range [0, 1] both tightens the result and protects the tail
    # denominator against a harmless outward ulp above one.
    return max(Decimal(0), value_lower), min(Decimal(1), value_upper)


def _exp_fraction_bounds(
    lower: Fraction,
    upper: Fraction,
    *,
    bits: int,
) -> tuple[Fraction, Fraction]:
    """Return a rigorous rational enclosure of ``exp(x)`` for ``x`` in an interval."""

    value_lower, value_upper = _exp_decimal_bounds(
        lower, upper, bits=bits
    )
    return Fraction(value_lower), Fraction(value_upper)


def _ln_fraction_bounds(value: Fraction, *, bits: int) -> tuple[Fraction, Fraction]:
    """Return a rigorous rational enclosure of ``ln(value)``."""

    if value <= 0:
        raise ArithmeticError("a logarithm interval requires a positive value")
    if value == 1:
        zero = Fraction(0)
        return zero, zero
    digits = _digits_for_bits(bits)
    argument_lower = _fraction_to_decimal_bound(
        value, digits=digits, rounding=ROUND_FLOOR
    )
    argument_upper = _fraction_to_decimal_bound(
        value, digits=digits, rounding=ROUND_CEILING
    )
    with localcontext() as context:
        context.prec = digits
        context.rounding = ROUND_HALF_EVEN
        context.Emax = MAX_EMAX
        context.Emin = MIN_EMIN
        rounded_lower = argument_lower.ln()
        rounded_upper = argument_upper.ln()
        # If directed conversion rounded a rational extremely close to one
        # exactly to Decimal(1), ln(1)=0 is already the exact endpoint.  Calling
        # next_minus at MIN_EMIN would instead manufacture a Decimal with an
        # exponent near -10^18, whose conversion to Fraction is impractical.
        value_lower = (
            Decimal(0)
            if argument_lower == 1
            else rounded_lower.next_minus(context=context)
        )
        value_upper = (
            Decimal(0)
            if argument_upper == 1
            else rounded_upper.next_plus(context=context)
        )
    return Fraction(value_lower), Fraction(value_upper)


def _fractional_center_exact(center: Fraction) -> Fraction:
    """Reduce an exact centre to [-1/2, 1/2)."""

    shifted = center + Fraction(1, 2)
    nearest = shifted.numerator // shifted.denominator
    return center - nearest


@lru_cache(maxsize=128)
def _arctan_inverse_bounds(inverse: int, bits: int) -> tuple[Fraction, Fraction]:
    """Enclose atan(1/inverse) with its alternating positive-term series."""

    target = Fraction(1, 1 << (bits + 24))
    total = Fraction(0)
    index = 0
    while True:
        term = Fraction(1, (2 * index + 1) * inverse ** (2 * index + 1))
        total = total + term if index % 2 == 0 else total - term
        next_index = index + 1
        next_term = Fraction(
            1, (2 * next_index + 1) * inverse ** (2 * next_index + 1)
        )
        if next_term <= target:
            if next_index % 2 == 0:
                return total, total + next_term
            return total - next_term, total
        index = next_index


@lru_cache(maxsize=128)
def _pi_bounds(bits: int) -> tuple[Fraction, Fraction]:
    """Certified Machin-formula enclosure of pi."""

    atan5_lower, atan5_upper = _arctan_inverse_bounds(5, bits + 8)
    atan239_lower, atan239_upper = _arctan_inverse_bounds(239, bits + 8)
    return (
        16 * atan5_lower - 4 * atan239_upper,
        16 * atan5_upper - 4 * atan239_lower,
    )


def _tail_upper_bound(
    radius: int,
    coefficient_lower: Fraction,
    *,
    bits: int,
) -> Fraction:
    """Bound both omitted tails outside [-radius, radius].

    For a reduced centre ``|c| <= 1/2``, each first omitted term is at most
    ``t = exp(-a*(R+1/2)^2)``.  Consecutive terms decrease by at least the
    factor ``r = exp(-a*(2R+2))``.  Hence the two tails total at most
    ``2*t/(1-r)``.
    """

    half_distance = Fraction(2 * radius + 1, 2)
    _, first_upper = _exp_fraction_bounds(
        -coefficient_lower * half_distance * half_distance,
        -coefficient_lower * half_distance * half_distance,
        bits=bits,
    )
    _, ratio_upper = _exp_fraction_bounds(
        -coefficient_lower * (2 * radius + 2),
        -coefficient_lower * (2 * radius + 2),
        bits=bits,
    )
    if ratio_upper >= 1:
        return Fraction(10**1000)
    return 2 * first_upper / (1 - ratio_upper)


@lru_cache(maxsize=16_384)
def _gaussian_mass_bounds(
    center: Fraction,
    coefficient_lower: Fraction,
    coefficient_upper: Fraction,
    bits: int,
) -> tuple[Fraction, Fraction]:
    """Enclose ``sum_k exp(-a*(k-c)^2)`` by a positive primal series."""

    if coefficient_lower <= 0 or coefficient_lower > coefficient_upper:
        raise CorrectionError("Gaussian exponent coefficient must be positive")
    precision = max(32, int(bits))
    reduced = _fractional_center_exact(center)

    central_lower, _ = _exp_fraction_bounds(
        -coefficient_upper * reduced * reduced,
        -coefficient_lower * reduced * reduced,
        bits=precision + 24,
    )
    target_tail = central_lower / (1 << (precision + 16))

    radius = 1
    previous = 0
    while (
        _tail_upper_bound(radius, coefficient_lower, bits=precision + 24)
        > target_tail
    ):
        previous = radius
        radius *= 2
        if radius > _MAX_CERTIFIED_RADIUS:
            raise CorrectionError(
                "certified Gaussian-mass radius exceeds the implementation limit"
            )

    # Tighten the power-of-two search result; Decimal exponentials dominate
    # runtime, so avoiding unnecessary primal-series terms is worthwhile.
    lower_radius = previous + 1
    upper_radius = radius
    while lower_radius < upper_radius:
        middle = (lower_radius + upper_radius) // 2
        if (
            _tail_upper_bound(middle, coefficient_lower, bits=precision + 24)
            <= target_tail
        ):
            upper_radius = middle
        else:
            lower_radius = middle + 1
    radius = lower_radius

    term_bits = precision + radius.bit_length() + 32

    # Generate every retained positive term by recurrence.  If
    # w_k=exp(-a(k-c)^2), then
    #   w_1/w_0=exp(-a(1-2c)),
    #   (w_{k+2}/w_{k+1})/(w_{k+1}/w_k)=exp(-2a).
    # The negative side has 1+2c instead.  Thus only four certified exp calls
    # are needed per mass instead of 2R+1, while directed Decimal arithmetic
    # keeps the whole positive recurrence outward-rounded.
    base_lower, base_upper = _exp_decimal_bounds(
        -coefficient_upper * reduced * reduced,
        -coefficient_lower * reduced * reduced,
        bits=term_bits,
    )
    positive_ratio_lower, positive_ratio_upper = _exp_decimal_bounds(
        -coefficient_upper * (1 - 2 * reduced),
        -coefficient_lower * (1 - 2 * reduced),
        bits=term_bits,
    )
    negative_ratio_lower, negative_ratio_upper = _exp_decimal_bounds(
        -coefficient_upper * (1 + 2 * reduced),
        -coefficient_lower * (1 + 2 * reduced),
        bits=term_bits,
    )
    step_lower, step_upper = _exp_decimal_bounds(
        -2 * coefficient_upper,
        -2 * coefficient_lower,
        bits=term_bits,
    )
    digits = _digits_for_bits(term_bits)
    with localcontext() as context:
        context.prec = digits
        context.Emax = MAX_EMAX
        context.Emin = MIN_EMIN
        partial_lower_d = base_lower
        partial_upper_d = base_upper
        positive_lower = base_lower
        positive_upper = base_upper
        negative_lower = base_lower
        negative_upper = base_upper
        for _distance in range(1, radius + 1):
            context.rounding = ROUND_FLOOR
            positive_lower = +(positive_lower * positive_ratio_lower)
            negative_lower = +(negative_lower * negative_ratio_lower)
            partial_lower_d = +(partial_lower_d + positive_lower)
            partial_lower_d = +(partial_lower_d + negative_lower)

            context.rounding = ROUND_CEILING
            positive_upper = +(positive_upper * positive_ratio_upper)
            negative_upper = +(negative_upper * negative_ratio_upper)
            partial_upper_d = +(partial_upper_d + positive_upper)
            partial_upper_d = +(partial_upper_d + negative_upper)

            context.rounding = ROUND_FLOOR
            positive_ratio_lower = +(positive_ratio_lower * step_lower)
            negative_ratio_lower = +(negative_ratio_lower * step_lower)
            context.rounding = ROUND_CEILING
            positive_ratio_upper = +(positive_ratio_upper * step_upper)
            negative_ratio_upper = +(negative_ratio_upper * step_upper)

    partial_lower = Fraction(partial_lower_d)
    partial_upper = Fraction(partial_upper_d)
    tail = _tail_upper_bound(radius, coefficient_lower, bits=term_bits)
    return partial_lower, partial_upper + tail


def log_mass_stddev_bounds(
    sigma: Any,
    center: Any = 0,
    *,
    bits: int = DEFAULT_ACCEPTANCE_BITS,
) -> tuple[Fraction, Fraction]:
    """Certified refinable bounds for a standard-deviation Gaussian mass.

    Finite inputs are interpreted as exact rationals.  The returned rational
    endpoints contain
    ``log(sum_z exp(-(z-center)^2/(2*sigma^2)))`` and converge as ``bits``
    grows.  Only a positive primal series is used; the omitted infinite tails
    are covered by a geometric bound.
    """

    if not isinstance(bits, int) or isinstance(bits, bool) or bits <= 0:
        raise CorrectionError("bits must be a positive integer")
    center_q = _exact_fraction(center, name="center")
    sigma_q = _exact_fraction(sigma, name="sigma")
    if sigma_q <= 0:
        raise CorrectionError("sigma must be positive")
    coefficient = Fraction(1, 2) / (sigma_q * sigma_q)
    mass_lower, mass_upper = _gaussian_mass_bounds(
        center_q, coefficient, coefficient, bits
    )
    log_lower, _ = _ln_fraction_bounds(mass_lower, bits=bits + 24)
    _, log_upper = _ln_fraction_bounds(mass_upper, bits=bits + 24)
    return log_lower, log_upper


def _log_mass_rho_bounds(
    rho_s: Any,
    center: Any,
    *,
    bits: int,
) -> tuple[Fraction, Fraction]:
    """Certified bounds in the paper's rho-width convention."""

    width = _exact_fraction(rho_s, name="rho_s")
    center_q = _exact_fraction(center, name="center")
    if width <= 0:
        raise CorrectionError("rho_s must be positive")
    pi_lower, pi_upper = _pi_bounds(bits + 32)
    width_squared = width * width
    mass_lower, mass_upper = _gaussian_mass_bounds(
        center_q,
        pi_lower / width_squared,
        pi_upper / width_squared,
        bits,
    )
    log_lower, _ = _ln_fraction_bounds(mass_lower, bits=bits + 24)
    _, log_upper = _ln_fraction_bounds(mass_upper, bits=bits + 24)
    return log_lower, log_upper


def _as_mpf(value: Any, *, name: str) -> mp.mpf:
    """Convert decimal-looking inputs without first rounding through float."""

    try:
        if isinstance(value, float):
            result = mp.mpf(repr(value))
        else:
            result = mp.mpf(value)
    except (TypeError, ValueError) as exc:
        raise CorrectionError(f"{name} must be a real number, got {value!r}") from exc
    if not mp.isfinite(result):
        raise CorrectionError(f"{name} must be finite, got {value!r}")
    return result


def _fractional_center(center: mp.mpf) -> mp.mpf:
    """Reduce a centre modulo Z to the stable interval [-1/2, 1/2)."""

    return center - mp.floor(center + mp.mpf("0.5"))


def log_rho_z(
    rho_s: Any,
    center: Any = 0,
    *,
    dps: int = DEFAULT_DPS,
) -> mp.mpf:
    """Return ``log(rho_{s,center}(Z))`` at high precision.

    A direct positive-term sum is used for ``s < 1``.  For ``s >= 1`` the
    Poisson-dual series is substantially faster:

    ``rho_{s,c}(Z) = s * (1 + 2 sum_{k>=1} exp(-pi*s^2*k^2) cos(2*pi*k*c))``.

    The number of terms is selected from ``dps`` with additional guard digits.
    This is an adaptive-precision numerical evaluation, not a formal interval
    certificate; the security-model package is responsible for certified
    theta bounds used in the paper tables.
    """

    if not isinstance(dps, int) or dps < 20:
        raise CorrectionError("dps must be an integer of at least 20")

    # Guard digits matter most for the alternating dual series near c=1/2.
    with mp.workdps(dps + 30):
        s = _as_mpf(rho_s, name="rho_s")
        c = _as_mpf(center, name="center")
        if s <= 0:
            raise CorrectionError("rho_s must be positive")
        c = _fractional_center(c)

        target_log = mp.mpf(dps + 18) * mp.log(10)
        if s < 1:
            radius = int(mp.ceil(mp.mpf("0.5") + s * mp.sqrt(target_log / mp.pi))) + 1
            mass = mp.fsum(
                mp.exp(-mp.pi * (mp.mpf(k) - c) ** 2 / (s * s))
                for k in range(-radius, radius + 1)
            )
        else:
            radius = int(mp.ceil(mp.sqrt(target_log / mp.pi) / s)) + 1
            dual_tail = mp.fsum(
                mp.exp(-mp.pi * s * s * k * k) * mp.cos(2 * mp.pi * k * c)
                for k in range(1, radius + 1)
            )
            mass = s * (1 + 2 * dual_tail)

        if mass <= 0 or not mp.isfinite(mass):
            raise ArithmeticError("discrete-Gaussian mass evaluation was not positive")
        return +mp.log(mass)


def rho_z(rho_s: Any, center: Any = 0, *, dps: int = DEFAULT_DPS) -> mp.mpf:
    """Return ``rho_{s,center}(Z)``; prefer :func:`log_rho_z` in products."""

    with mp.workdps(dps + 10):
        return +mp.exp(log_rho_z(rho_s, center, dps=dps))


def log_mass_stddev(
    sigma: Any,
    center: Any = 0,
    *,
    dps: int = DEFAULT_DPS,
) -> mp.mpf:
    """Log normalizer for ``exp(-(z-center)^2/(2*sigma^2))`` on Z."""

    with mp.workdps(dps + 10):
        stddev = _as_mpf(sigma, name="sigma")
        if stddev <= 0:
            raise CorrectionError("sigma must be positive")
        return +log_rho_z(mp.sqrt(2 * mp.pi) * stddev, center, dps=dps)


def log_mass_ratio(
    center: Any,
    sigma: Any,
    *,
    dps: int = DEFAULT_DPS,
) -> mp.mpf:
    """Return ``log(rho_{s,center}(Z) / rho_{s,0}(Z))``.

    The result is non-positive.  Tiny positive values caused solely by finite
    precision are clamped to zero; a material violation raises an error.
    """

    with mp.workdps(dps + 20):
        numerator = log_mass_stddev(sigma, center, dps=dps + 10)
        denominator = log_mass_stddev(sigma, 0, dps=dps + 10)
        ratio = numerator - denominator
        tolerance = mp.power(10, -max(12, dps - 8))
        if ratio > tolerance:
            raise ArithmeticError("shifted Gaussian mass exceeded the centered mass")
        return mp.mpf("0") if ratio > 0 else +ratio


def _entry_value(entry: Any, *names: str, default: Any = _MISSING) -> Any:
    if isinstance(entry, Mapping):
        for name in names:
            if name in entry:
                return entry[name]
    else:
        for name in names:
            if hasattr(entry, name):
                return getattr(entry, name)
    if default is not _MISSING:
        return default
    joined = ", ".join(names)
    raise CorrectionError(f"trace entry is missing required field ({joined})")


def _entry_log_ratio(
    entry: Any,
    *,
    global_rho_s: Any | None,
    dps: int,
) -> mp.mpf:
    cached_centered = _entry_value(
        entry,
        "log_mass_centered",
        "log_rho_centered",
        "log_rho_shifted",
        default=None,
    )
    cached_zero = _entry_value(
        entry,
        "log_mass_zero",
        "log_rho_zero",
        default=None,
    )
    if (cached_centered is None) != (cached_zero is None):
        raise CorrectionError(
            "trace entry must provide both log_mass_centered and log_mass_zero"
        )
    if cached_centered is not None:
        return _as_mpf(cached_centered, name="log_mass_centered") - _as_mpf(
            cached_zero, name="log_mass_zero"
        )

    center = _entry_value(entry, "center", "conditional_center", "mu")
    sigma = _entry_value(
        entry,
        "sigma",
        "local_sigma",
        "sigma_local",
        "stddev",
        default=None,
    )
    if sigma is not None:
        return log_mass_ratio(center, sigma, dps=dps)

    local_rho_s = _entry_value(
        entry,
        "rho_s",
        "local_rho_s",
        "s_local",
        default=None,
    )
    if local_rho_s is None:
        gs_norm = _entry_value(entry, "gs_norm", "gso_norm", default=None)
        if global_rho_s is None or gs_norm is None:
            raise CorrectionError(
                "trace entry needs sigma, local rho_s, or (global_rho_s, gs_norm)"
            )
        local_rho_s = _as_mpf(global_rho_s, name="global_rho_s") / _as_mpf(
            gs_norm, name="gs_norm"
        )

    return log_rho_z(local_rho_s, center, dps=dps) - log_rho_z(
        local_rho_s, 0, dps=dps
    )


def log_delta_from_trace(
    trace: Iterable[Any],
    *,
    global_rho_s: Any | None = None,
    dps: int = DEFAULT_DPS,
    strict: bool = True,
) -> mp.mpf:
    """Compute ``log Delta_B(x)`` from an ffSampling/Klein trace.

    Entries may be :class:`CorrectionTraceEntry`, mappings, or objects with
    equivalent attributes.  The preferred fields are ``center`` and ``sigma``.
    Precomputed logarithmic masses are accepted to avoid duplicate theta work.

    ``global_rho_s`` is only needed by legacy traces that contain ``gs_norm``
    but no local width.  It uses the paper convention, so the local parameter
    is ``global_rho_s / gs_norm``.
    """

    if not isinstance(dps, int) or dps < 20:
        raise CorrectionError("dps must be an integer of at least 20")
    try:
        iterator = iter(trace)
    except TypeError as exc:
        raise CorrectionError("trace must be iterable") from exc

    with mp.workdps(dps + 25):
        terms = [_entry_log_ratio(entry, global_rho_s=global_rho_s, dps=dps) for entry in iterator]
        result = mp.fsum(terms) if terms else mp.mpf("0")
        # Cached values supplied as binary64 can be a few ulps positive even
        # though every exact ratio is <= 1.  Permit only a harmless tolerance.
        tolerance = mp.mpf("1e-12")
        if result > 0:
            if strict and result > tolerance:
                raise CorrectionError(
                    f"invalid correction trace: log Delta is positive ({result})"
                )
            result = mp.mpf("0")
        return +result


def delta_from_trace(
    trace: Iterable[Any],
    *,
    global_rho_s: Any | None = None,
    dps: int = DEFAULT_DPS,
    strict: bool = True,
) -> mp.mpf:
    """Compute ``Delta_B(x)`` from a trace."""

    with mp.workdps(dps + 10):
        return +mp.exp(
            log_delta_from_trace(
                trace, global_rho_s=global_rho_s, dps=dps, strict=strict
            )
        )


_CertifiedTraceTerm = tuple[str, Fraction, Fraction]


def _prepare_certified_trace(
    trace: Iterable[Any],
    *,
    global_rho_s: Any | None,
) -> tuple[_CertifiedTraceTerm, ...]:
    """Freeze every formal trace scalar with the sampler's exact semantics."""

    try:
        iterator = iter(trace)
    except TypeError as exc:
        raise CorrectionError("trace must be iterable") from exc
    global_width = (
        None
        if global_rho_s is None
        else _exact_fraction(global_rho_s, name="global_rho_s")
    )
    if global_width is not None and global_width <= 0:
        raise CorrectionError("global_rho_s must be positive")

    prepared: list[_CertifiedTraceTerm] = []
    for entry in iterator:
        cached_centered = _entry_value(
            entry,
            "log_mass_centered",
            "log_rho_centered",
            "log_rho_shifted",
            default=None,
        )
        cached_zero = _entry_value(
            entry,
            "log_mass_zero",
            "log_rho_zero",
            default=None,
        )
        if (cached_centered is None) != (cached_zero is None):
            raise CorrectionError(
                "trace entry must provide both log_mass_centered and log_mass_zero"
            )
        if cached_centered is not None:
            # A rounded logarithm cannot certify the transcendental mass that
            # it purports to cache.  The high-precision diagnostic API still
            # accepts these legacy fields, but the exact Bernoulli path does
            # not silently elevate them into certificates.
            raise CorrectionError(
                "certified correction requires center/width values, not cached logs"
            )

        center = _exact_fraction(
            _entry_value(entry, "center", "conditional_center", "mu"),
            name="center",
        )
        sigma = _entry_value(
            entry,
            "sigma",
            "local_sigma",
            "sigma_local",
            "stddev",
            default=None,
        )
        if sigma is not None:
            width = _exact_fraction(sigma, name="sigma")
            if width <= 0:
                raise CorrectionError("sigma must be positive")
            prepared.append(("sigma", center, width))
            continue

        local_rho_s = _entry_value(
            entry,
            "rho_s",
            "local_rho_s",
            "s_local",
            default=None,
        )
        if local_rho_s is None:
            gs_norm = _entry_value(entry, "gs_norm", "gso_norm", default=None)
            if global_width is None or gs_norm is None:
                raise CorrectionError(
                    "trace entry needs sigma, local rho_s, or "
                    "(global_rho_s, gs_norm)"
                )
            norm = _exact_fraction(gs_norm, name="gs_norm")
            if norm <= 0:
                raise CorrectionError("gs_norm must be positive")
            width = global_width / norm
        else:
            width = _exact_fraction(local_rho_s, name="rho_s")
        if width <= 0:
            raise CorrectionError("rho_s must be positive")
        prepared.append(("rho", center, width))
    return tuple(prepared)


def _log_delta_bounds_prepared(
    prepared: tuple[_CertifiedTraceTerm, ...],
    *,
    bits: int,
) -> tuple[Fraction, Fraction]:
    lower = Fraction(0)
    upper = Fraction(0)
    for convention, center, width in prepared:
        if convention == "sigma":
            numerator_lower, numerator_upper = log_mass_stddev_bounds(
                width, center, bits=bits
            )
            denominator_lower, denominator_upper = log_mass_stddev_bounds(
                width, 0, bits=bits
            )
        else:
            numerator_lower, numerator_upper = _log_mass_rho_bounds(
                width, center, bits=bits
            )
            denominator_lower, denominator_upper = _log_mass_rho_bounds(
                width, 0, bits=bits
            )
        ratio_lower = numerator_lower - denominator_upper
        ratio_upper = numerator_upper - denominator_lower
        # Jacobi theta attains its maximum at an integral centre.  This exact
        # analytic fact may safely tighten an outward numerical interval.
        ratio_upper = min(Fraction(0), ratio_upper)
        if ratio_lower > ratio_upper:
            raise ArithmeticError("inconsistent certified mass-ratio bounds")
        lower += ratio_lower
        upper += ratio_upper
    return lower, upper


def log_delta_bounds_from_trace(
    trace: Iterable[Any],
    *,
    global_rho_s: Any | None = None,
    bits: int = DEFAULT_ACCEPTANCE_BITS,
) -> tuple[Fraction, Fraction]:
    """Return certified refinable rational bounds on ``log Delta_B(x)``."""

    if not isinstance(bits, int) or isinstance(bits, bool) or bits <= 0:
        raise CorrectionError("bits must be a positive integer")
    prepared = _prepare_certified_trace(trace, global_rho_s=global_rho_s)
    return _log_delta_bounds_prepared(prepared, bits=bits)


def _parameter_field(parameters: Any, *names: str, default: Any = _MISSING) -> Any:
    return _entry_value(parameters, *names, default=default)


def _canonical_key(n: Any, q: Any, gamma: Any) -> tuple[int, int, str]:
    try:
        n_value = int(n)
        q_value = int(q)
    except (TypeError, ValueError) as exc:
        raise CorrectionError("n and q must be integers") from exc
    with mp.workdps(30):
        gamma_value = _as_mpf(gamma, name="gamma")
        for candidate in ("1.17", "1.25"):
            if mp.almosteq(gamma_value, mp.mpf(candidate), abs_eps=mp.mpf("1e-20")):
                return n_value, q_value, candidate
    raise CorrectionError(f"unsupported Falcon++ gamma value: {gamma!r}")


def _global_correction_multiplier_fraction(parameters: Any) -> Fraction:
    """Resolve widehat-Gamma without any intermediate floating conversion."""

    direct = _parameter_field(
        parameters,
        "widehat_gamma_decimal",
        "gamma_hat_decimal",
        "correction_multiplier_decimal",
        "widehat_gamma",
        "gamma_hat",
        "correction_multiplier",
        default=None,
    )
    if direct is not None:
        multiplier = _exact_fraction(direct, name="correction multiplier")
    elif isinstance(parameters, str):
        alias = parameters.strip().lower()
        key = _NAME_ALIASES.get(alias)
        if key is not None:
            multiplier = Fraction(Decimal(GLOBAL_CORRECTION_MULTIPLIERS[key]))
        else:
            # A decimal string is an unambiguous direct multiplier.  Other
            # unknown strings retain the historical identifier error.
            try:
                multiplier = _exact_fraction(
                    Decimal(parameters), name="correction multiplier"
                )
            except Exception as exc:
                raise CorrectionError(
                    f"unknown Falcon++ parameter identifier: {parameters!r}"
                ) from exc
    elif isinstance(parameters, (tuple, list)) and len(parameters) == 3:
        key = _canonical_key(*parameters)
        try:
            multiplier = Fraction(Decimal(GLOBAL_CORRECTION_MULTIPLIERS[key]))
        except KeyError as exc:
            raise CorrectionError(
                f"unsupported Falcon++ parameter tuple: {key!r}"
            ) from exc
    else:
        n = _parameter_field(parameters, "n", default=None)
        q = _parameter_field(parameters, "q", default=None)
        gamma = _parameter_field(parameters, "gamma", default=None)
        if n is not None and q is not None and gamma is not None:
            key = _canonical_key(n, q, gamma)
            try:
                multiplier = Fraction(
                    Decimal(GLOBAL_CORRECTION_MULTIPLIERS[key])
                )
            except KeyError as exc:
                raise CorrectionError(
                    f"unsupported Falcon++ parameter tuple: {key!r}"
                ) from exc
        else:
            # Plain numerics are always direct multipliers.  In particular,
            # integer 1 means Gamma=1; parameter identifiers must be textual
            # or carried by a parameter object.
            multiplier = _exact_fraction(parameters, name="correction multiplier")

    if multiplier <= 0:
        raise CorrectionError("correction multiplier must be positive")
    return multiplier


def global_correction_multiplier(
    parameters: Any,
    *,
    dps: int = DEFAULT_DPS,
) -> mp.mpf:
    """Resolve the fixed manuscript ``widehat Gamma`` for a parameter set.

    Accepted inputs include a ``FalconPPParameters`` object, a mapping, an
    ``(n, q, gamma)`` tuple, a known textual identifier, or a direct positive
    numeric multiplier.  Plain integers are multipliers, not parameter IDs.
    Explicit object fields take precedence over lookup;
    exact ``*_decimal`` variants are preferred when present, followed by
    ``widehat_gamma``, ``gamma_hat``, and ``correction_multiplier``.  The
    KeyGen field named merely ``gamma`` is never mistaken for the correction
    multiplier.
    """

    if not isinstance(dps, int) or isinstance(dps, bool) or dps < 20:
        raise CorrectionError("dps must be an integer of at least 20")

    # Resolve to an exact rational first.  In particular, do not turn the
    # manuscript decimal into an mpf and then pretend that rounded mpf is the
    # exact multiplier used by the Bernoulli decision.
    with mp.workdps(dps + 10):
        multiplier = _global_correction_multiplier_fraction(parameters)
        return +(mp.mpf(multiplier.numerator) / multiplier.denominator)


def _log_acceptance_bounds_prepared(
    prepared: tuple[_CertifiedTraceTerm, ...],
    multiplier: Fraction,
    *,
    bits: int,
) -> tuple[Fraction, Fraction]:
    delta_lower, delta_upper = _log_delta_bounds_prepared(prepared, bits=bits)
    gamma_lower, gamma_upper = _ln_fraction_bounds(
        multiplier, bits=bits + 24
    )
    raw_lower = gamma_lower + delta_lower
    raw_upper = gamma_upper + delta_upper
    # min(0, x) is monotone, including when the enclosure straddles the
    # clipping boundary.  If the true raw value is non-negative, successive
    # refinements eventually collapse this interval to [0, 0] except at the
    # measure-zero exact boundary, where the Bernoulli outcome is still one.
    return min(Fraction(0), raw_lower), min(Fraction(0), raw_upper)


def log_acceptance_bounds_from_trace(
    trace: Iterable[Any],
    parameters: Any,
    *,
    global_rho_s: Any | None = None,
    bits: int = DEFAULT_ACCEPTANCE_BITS,
) -> tuple[Fraction, Fraction]:
    """Certified bounds on ``log(min(1, widehat_Gamma*Delta_B(x)))``."""

    if not isinstance(bits, int) or isinstance(bits, bool) or bits <= 0:
        raise CorrectionError("bits must be a positive integer")
    prepared = _prepare_certified_trace(trace, global_rho_s=global_rho_s)
    multiplier = _global_correction_multiplier_fraction(parameters)
    return _log_acceptance_bounds_prepared(
        prepared, multiplier, bits=bits
    )


def clipped_log_acceptance(
    log_delta: Any,
    parameters: Any,
    *,
    dps: int = DEFAULT_DPS,
) -> mp.mpf:
    """Return ``log(min(1, widehat_Gamma * Delta))`` without underflow."""

    with mp.workdps(dps + 10):
        log_d = _as_mpf(log_delta, name="log_delta")
        if log_d > mp.mpf("1e-12"):
            raise CorrectionError("log_delta cannot be positive")
        if log_d > 0:
            log_d = mp.mpf("0")
        multiplier = global_correction_multiplier(parameters, dps=dps)
        return +min(mp.mpf("0"), mp.log(multiplier) + log_d)


def clipped_acceptance_probability(
    log_delta: Any,
    parameters: Any,
    *,
    dps: int = DEFAULT_DPS,
) -> mp.mpf:
    """Return ``min(1, widehat_Gamma * Delta)``."""

    with mp.workdps(dps + 10):
        return +mp.exp(clipped_log_acceptance(log_delta, parameters, dps=dps))


def _random_bytes(source: Any, length: int) -> bytes:
    if source is None:
        output = secrets.token_bytes(length)
    elif callable(source):
        output = source(length)
    else:
        output = None
        for method_name in ("random_bytes", "token_bytes", "read", "squeeze"):
            method = getattr(source, method_name, None)
            if method is not None:
                output = method(length)
                break
        if output is None:
            raise CorrectionError(
                "random source must be callable or expose random_bytes/read/squeeze"
            )
    if not isinstance(output, (bytes, bytearray, memoryview)):
        raise CorrectionError("random source did not return bytes")
    result = bytes(output)
    if len(result) != length:
        raise CorrectionError(
            f"random source returned {len(result)} bytes; expected {length}"
        )
    return result


def accept_log_probability(
    log_probability: Any,
    *,
    random_source: Callable[[int], bytes] | Any | None = None,
    bits: int = DEFAULT_ACCEPTANCE_BITS,
) -> bool:
    """Sample exactly from a supplied finite log probability.

    The supplied scalar is frozen as an exact rational and compared with one
    lazily revealed uniform real through :func:`bernoulli_from_log_bounds`.
    ``bits`` is only the first random/refinement chunk; it does not quantize
    the probability.  The full correction path uses certified theta-mass
    bounds directly instead of first passing through this scalar helper.
    """

    if not isinstance(bits, int) or bits < 32:
        raise CorrectionError("bits must be an integer of at least 32")
    try:
        log_p = exact_real_fraction(log_probability, name="log_probability")
    except (TypeError, ValueError, OverflowError) as exc:
        # Negative infinity denotes probability zero and is the sole allowed
        # non-finite value, retained for compatibility with the old helper.
        try:
            diagnostic = mp.mpf(log_probability)
        except (TypeError, ValueError) as conversion_exc:
            raise CorrectionError(
                f"log_probability must be a real number, got {log_probability!r}"
            ) from conversion_exc
        if diagnostic == mp.ninf:
            return False
        raise CorrectionError(
            "log_probability must be finite or negative infinity"
        ) from exc
    if log_p > 0:
        raise CorrectionError("log_probability cannot be positive")
    if log_p == 0:
        return True

    coin_source: Any = random_source
    if (
        random_source is not None
        and not callable(random_source)
        and not any(
            hasattr(random_source, method)
            for method in ("read", "random_bytes", "randbytes")
        )
    ):
        # Preserve the historical token_bytes/squeeze protocol through the
        # exact Bernoulli primitive, whose native protocol is slightly smaller.
        class _ReaderAdapter:
            def read(self, length: int) -> bytes:
                return _random_bytes(random_source, length)

        coin_source = _ReaderAdapter()
    try:
        return bernoulli_from_log_bounds(
            lambda _requested_bits: (log_p, log_p),
            coin_source,
            initial_bits=bits,
            refinement_bits=max(16, bits // 2),
        )
    except (TypeError, ValueError) as exc:
        raise CorrectionError(str(exc)) from exc


def evaluate_correction(
    trace: Iterable[Any],
    parameters: Any,
    *,
    random_source: Callable[[int], bytes] | Any | None = None,
    bits: int = DEFAULT_ACCEPTANCE_BITS,
    global_rho_s: Any | None = None,
    dps: int = DEFAULT_DPS,
    strict: bool = True,
    correction_backend: str = "primal",
    diagnostic_level: str = "full",
    correction_context: Any = None,
) -> CorrectionDecision:
    """Evaluate diagnostics and make an exact adaptive correction decision.

    The returned ``mpmath`` fields are high-precision diagnostics only.  The
    Boolean decision is sampled from certified, refinable rational bounds
    independently of ``dps`` diagnostic rounding.  ``primal`` preserves the
    original log-domain implementation; ``thetadiv`` uses direct probability
    bounds from the theta product.  ``counts`` omits all three diagnostic
    values rather than reporting invented zeros.  ``bits`` controls only the
    initial lazy comparison chunk.  These guarantees concern the finite
    trace scalars, not unquantified upstream FFT/LDL approximation error.
    """

    if not isinstance(bits, int) or isinstance(bits, bool) or bits < 32:
        raise CorrectionError("bits must be an integer of at least 32")
    if correction_backend not in {"primal", "thetadiv"}:
        raise CorrectionError("correction_backend must be 'primal' or 'thetadiv'")
    if diagnostic_level not in {"full", "counts"}:
        raise CorrectionError("diagnostic_level must be 'full' or 'counts'")
    if correction_backend == "primal" and correction_context is not None:
        raise CorrectionError("correction_context is only used by thetadiv")
    started = perf_counter()
    try:
        trace_entries = tuple(trace)
    except TypeError as exc:
        raise CorrectionError("trace must be iterable") from exc

    diagnostic_started = perf_counter()
    log_delta = log_probability = probability = None
    if diagnostic_level == "full":
        log_delta = log_delta_from_trace(
            trace_entries, global_rho_s=global_rho_s, dps=dps, strict=strict
        )
    with mp.workdps(dps + 10):
        multiplier = global_correction_multiplier(parameters, dps=dps)
        if diagnostic_level == "full":
            log_probability = clipped_log_acceptance(
                log_delta, parameters, dps=dps
            )
            probability = +mp.exp(log_probability)
    metrics: dict[str, int | float] = {
        "diagnostic_seconds": perf_counter() - diagnostic_started,
        "certified_bounds_seconds": 0.0,
        "bounds_evaluations": 0,
        "bounds_cache_hits": 0,
    }

    exact_multiplier = _global_correction_multiplier_fraction(parameters)
    if correction_backend == "primal":
        prepared = _prepare_certified_trace(
            trace_entries, global_rho_s=global_rho_s
        )
        def compute_bounds(requested_bits: int) -> tuple[Fraction, Fraction]:
            return _log_acceptance_bounds_prepared(
                prepared, exact_multiplier, bits=requested_bits
            )
        coin_function = bernoulli_from_log_bounds
    else:
        from .theta_division import ThetaDivContext

        context = ThetaDivContext() if correction_context is None else correction_context
        if not isinstance(context, ThetaDivContext):
            raise CorrectionError("correction_context must be a ThetaDivContext")
        def compute_bounds(requested_bits: int) -> tuple[Fraction, Fraction]:
            return context.trace_probability_bounds(
                trace_entries, exact_multiplier, bits=requested_bits,
                global_rho_s=global_rho_s,
            )
        coin_function = bernoulli_from_probability_bounds

    # This call-local cache includes centres only indirectly in its closure;
    # it is discarded on return, never persisted in the per-key width cache.
    bounds_cache: dict[int, tuple[Fraction, Fraction]] = {}
    def certified_bounds(requested_bits: int) -> tuple[Fraction, Fraction]:
        requested_bits = max(32, requested_bits)
        if requested_bits in bounds_cache:
            metrics["bounds_cache_hits"] += 1
            return bounds_cache[requested_bits]
        before = perf_counter()
        result = compute_bounds(requested_bits)
        metrics["certified_bounds_seconds"] += perf_counter() - before
        metrics["bounds_evaluations"] += 1
        bounds_cache[requested_bits] = result
        return result

    first_lower, first_upper = certified_bounds(bits + 32)
    certain_one = 0 if correction_backend == "primal" else 1
    if first_lower == first_upper == certain_one:
        accepted = True
    elif correction_backend == "thetadiv" and first_lower == first_upper == 0:
        accepted = False
    else:
        coin_source: Any = random_source
        if (
            random_source is not None
            and not callable(random_source)
            and not any(
                hasattr(random_source, method)
                for method in ("read", "random_bytes", "randbytes")
            )
        ):
            class _ReaderAdapter:
                def read(self, length: int) -> bytes:
                    return _random_bytes(random_source, length)

            coin_source = _ReaderAdapter()
        try:
            accepted = coin_function(
                certified_bounds,
                coin_source,
                initial_bits=bits,
                refinement_bits=max(16, bits // 2),
            )
        except (TypeError, ValueError) as exc:
            raise CorrectionError(str(exc)) from exc
    metrics["total_seconds"] = perf_counter() - started
    return CorrectionDecision(
        accepted=accepted,
        log_delta=log_delta,
        log_probability=log_probability,
        probability=probability,
        multiplier=multiplier,
        correction_backend=correction_backend,
        diagnostic_level=diagnostic_level,
        metrics=metrics,
    )


__all__ = [
    "CorrectionDecision",
    "CorrectionError",
    "CorrectionTraceEntry",
    "DEFAULT_ACCEPTANCE_BITS",
    "DEFAULT_DPS",
    "GLOBAL_CORRECTION_MULTIPLIERS",
    "accept_log_probability",
    "clipped_acceptance_probability",
    "clipped_log_acceptance",
    "delta_from_trace",
    "evaluate_correction",
    "global_correction_multiplier",
    "log_acceptance_bounds_from_trace",
    "log_delta_from_trace",
    "log_delta_bounds_from_trace",
    "log_mass_ratio",
    "log_mass_stddev",
    "log_mass_stddev_bounds",
    "log_rho_z",
    "rho_z",
]
