r"""One-dimensional discrete Gaussians for the Falcon++ reference code.

Throughout this module ``sigma`` is the *ordinary standard deviation* and

.. math::

   Pr[X=k] \propto \exp(-(k-c)^2/(2\sigma^2)), \qquad k\in\mathbb Z.

Thus the paper's ``rho_s`` width is ``s = sqrt(2*pi) * sigma``.  Making this
conversion explicit prevents one of the most common implementation errors in
GPV/Klein samplers.

The authoritative sampler has no finite table and no probability rounding.
It samples an exact, infinite two-sided geometric proposal and makes the
transcendental rejection decision by lazily refining rigorous intervals while
revealing bits of one uniform random real.  The comparison terminates almost
surely.  This is a transparent research implementation, not a constant-time
production sampler.

Finite fixed-point CDFs remain available for tests and compression modelling,
but are explicitly approximate.  Likewise, ``mode="fast"`` uses a quantized
proposal and binary64 rejection exponential and is only a diagnostic
accelerator; it must never support a security claim.

Randomness is dependency-injected.  A source may expose ``read(n)``,
``random_bytes(n)``, or ``randbytes(n)``, or it may itself be callable as
``source(n) -> bytes``.  This accepts :class:`falconpp.randomness.Shake256PRNG`
as well as small test doubles.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
from dataclasses import dataclass
from decimal import (
    Decimal,
    InvalidOperation,
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
from typing import Any, Callable, Iterable


_LOG10_2 = math.log10(2.0)
_PI_DECIMAL = Decimal(
    "3.141592653589793238462643383279502884197169399375105820974944592307816406286"
)


def _decimal_digits(bits: int, guard_digits: int = 20) -> int:
    return max(28, math.ceil(bits * _LOG10_2) + guard_digits)


def _validate_precision(precision_bits: int, tail_bits: int) -> None:
    if not isinstance(precision_bits, int) or isinstance(precision_bits, bool):
        raise TypeError("precision_bits must be an integer")
    if not isinstance(tail_bits, int) or isinstance(tail_bits, bool):
        raise TypeError("tail_bits must be an integer")
    if not 32 <= precision_bits <= 4096:
        raise ValueError("precision_bits must lie in [32, 4096]")
    if not 32 <= tail_bits <= 4096:
        raise ValueError("tail_bits must lie in [32, 4096]")


def _as_decimal(value: Any, name: str) -> Decimal:
    """Convert a real scalar without silently discarding an imaginary part."""

    if isinstance(value, Decimal):
        result = value
    elif isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, not bool")
    elif isinstance(value, int):
        result = Decimal(value)
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        # The calling FFT has already selected a binary64 centre.  Retaining
        # that exact value is more honest than pretending str(value) supplied
        # additional mathematical precision.
        result = Decimal.from_float(value)
    elif isinstance(value, complex):
        scale = max(1.0, abs(value.real))
        if abs(value.imag) > 2.0e-12 * scale:
            raise ValueError(f"{name} has a non-negligible imaginary part")
        result = Decimal.from_float(float(value.real))
    elif hasattr(value, "real") and hasattr(value, "imag"):
        # In particular, accept mpmath.mpc leaf values without first reducing
        # their real part to binary64.
        real = value.real
        imag = value.imag
        try:
            scale = max(Decimal(1), abs(Decimal(str(real))))
            imag_d = abs(Decimal(str(imag)))
        except (InvalidOperation, ValueError) as exc:
            raise TypeError(f"{name} must be a finite real number") from exc
        if imag_d > Decimal("2e-12") * scale:
            raise ValueError(f"{name} has a non-negligible imaginary part")
        result = Decimal(str(real))
    else:
        # mpmath numbers and textual values have reliable decimal strings.
        try:
            result = Decimal(str(value))
        except (InvalidOperation, ValueError) as exc:
            raise TypeError(f"{name} must be a finite real number") from exc
    if not result.is_finite():
        raise ValueError(f"{name} must be finite")
    return result


def _validate_center_sigma(center: Any, sigma: Any) -> tuple[Decimal, Decimal]:
    center_d = _as_decimal(center, "center")
    sigma_d = _as_decimal(sigma, "sigma")
    if sigma_d <= 0:
        raise ValueError("sigma must be strictly positive")
    return center_d, sigma_d


def _read_random(source: Any, count: int) -> bytes:
    if count < 0:
        raise ValueError("count must be non-negative")
    if source is None:
        data = secrets.token_bytes(count)
    elif hasattr(source, "read"):
        data = source.read(count)
    elif hasattr(source, "random_bytes"):
        data = source.random_bytes(count)
    elif hasattr(source, "randbytes"):
        data = source.randbytes(count)
    elif callable(source):
        data = source(count)
    else:
        raise TypeError(
            "rng must provide read/random_bytes/randbytes or be callable"
        )
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("the randomness source must return bytes-like data")
    result = bytes(data)
    if len(result) != count:
        raise ValueError(
            f"the randomness source returned {len(result)} bytes, expected {count}"
        )
    return result


def _random_bits(source: Any, bits: int) -> int:
    byte_count = (bits + 7) // 8
    value = int.from_bytes(_read_random(source, byte_count), "little")
    return value & ((1 << bits) - 1)


def _randbelow(source: Any, upper: int) -> int:
    """Return a uniform integer in ``range(upper)`` without modulo bias."""

    if not isinstance(upper, int) or isinstance(upper, bool) or upper <= 0:
        raise ValueError("upper must be a positive integer")
    if upper == 1:
        return 0
    bits = (upper - 1).bit_length()
    while True:
        candidate = _random_bits(source, bits)
        if candidate < upper:
            return candidate


def exact_real_fraction(value: Any, *, name: str = "value") -> Fraction:
    """Interpret a supplied finite real scalar as an exact rational.

    ``float`` and ``mpmath.mpf`` inputs use their exact stored dyadic value;
    in particular, conversion never depends on the ambient ``mp.dps``.
    Decimal, Fraction, and integer inputs retain their exact value.  A complex
    scalar is accepted only when its imaginary component is exactly zero.
    This is the canonical scalar interpretation shared by formal sampling and
    correction decisions.
    """

    if isinstance(value, Fraction):
        return value
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError(f"{name} must be finite")
        return Fraction(value)
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, not bool")
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return Fraction.from_float(value)
    if isinstance(value, complex):
        if value.imag != 0:
            raise ValueError(f"{name} must be exactly real")
        if not math.isfinite(value.real):
            raise ValueError(f"{name} must be finite")
        return Fraction.from_float(value.real)
    mpc_tuple = getattr(value, "_mpc_", None)
    if mpc_tuple is not None:
        real_tuple, imaginary_tuple = mpc_tuple
        if imaginary_tuple[3] < 0:
            raise ValueError(f"{name} must be finite")
        if imaginary_tuple[1] != 0:
            raise ValueError(f"{name} must be exactly real")
        mpf_tuple = real_tuple
    else:
        mpf_tuple = getattr(value, "_mpf_", None)
    if mpf_tuple is not None:
        # Preserve every bit of an mpmath.mpf even if its creating workdps
        # context has already ended.  The internal tuple is
        # (negative, mantissa, binary exponent, bitcount).
        negative, mantissa, exponent, bitcount = mpf_tuple
        if bitcount < 0:
            raise ValueError(f"{name} must be finite")
        numerator = -mantissa if negative else mantissa
        if exponent >= 0:
            return Fraction(numerator << exponent)
        return Fraction(numerator, 1 << (-exponent))
    as_integer_ratio = getattr(value, "as_integer_ratio", None)
    if callable(as_integer_ratio):
        try:
            numerator, denominator = as_integer_ratio()
            return Fraction(int(numerator), int(denominator))
        except (OverflowError, ValueError) as exc:
            raise ValueError(f"{name} must be finite") from exc
    try:
        decimal_value = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise TypeError(f"{name} must be a finite real number") from exc
    if not decimal_value.is_finite():
        raise ValueError(f"{name} must be finite")
    return Fraction(decimal_value)


def _fraction_to_decimal_bound(
    value: Fraction, *, digits: int, rounding: str
) -> Decimal:
    """Convert a rational to a directed Decimal bound."""

    with localcontext() as context:
        context.prec = digits
        context.rounding = rounding
        context.Emax = MAX_EMAX
        context.Emin = MIN_EMIN
        return +(Decimal(value.numerator) / Decimal(value.denominator))


def _ln_fraction_interval(value: Fraction, *, bits: int) -> tuple[Fraction, Fraction]:
    """Rigorous rational enclosure of ``ln(value)`` for positive ``value``.

    Python's Decimal ``ln`` is correctly rounded with ROUND_HALF_EVEN.  We
    first enclose the rational input with directed division, then step one
    representable Decimal below/above the rounded logarithms.  Monotonicity of
    ``ln`` therefore yields a genuine enclosure rather than a precision
    heuristic.
    """

    if value <= 0:
        raise ValueError("a logarithm interval requires a positive value")
    if value == 1:
        zero = Fraction(0)
        return zero, zero
    digits = _decimal_digits(max(32, bits), guard_digits=24)
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
        # A directed endpoint can round to exactly 1 even when the rational
        # lies infinitesimally above/below it.  log(1) is exactly zero; taking
        # next_minus/next_plus at the global Decimal Emin would manufacture a
        # number with an denominator around 10**10**18 when converted to a
        # Fraction.  Keeping the exact zero endpoint is both tighter and the
        # mathematically correct directed bound.
        lower = (
            Decimal(0)
            if argument_lower == 1
            else argument_lower.ln().next_minus(context=context)
        )
        upper = (
            Decimal(0)
            if argument_upper == 1
            else argument_upper.ln().next_plus(context=context)
        )
    return Fraction(lower), Fraction(upper)


@lru_cache(maxsize=256)
def _cached_ln_interval(
    value: Fraction, bits: int
) -> tuple[Fraction, Fraction]:
    """Cache logarithms of the few fixed proposal ratios in use."""

    return _ln_fraction_interval(value, bits=bits)


LogBounds = Callable[[int], tuple[Any, Any]]


def bernoulli_from_probability_bounds(
    probability_bounds: Callable[[int], tuple[Any, Any]],
    rng: Any = None,
    *,
    initial_bits: int = 64,
    refinement_bits: int = 32,
) -> bool:
    """Compare one lazy uniform real with certified bounds on ``p`` directly.

    The provider must enclose the same probability at every precision and
    converge to it.  No logarithms or finite-bit probability rounding are
    used.  This helper is separate from, and does not modify, the existing
    exact one-dimensional Gaussian sampler.
    """

    if not callable(probability_bounds):
        raise TypeError("probability_bounds must be callable")
    for value, name in ((initial_bits, "initial_bits"),
                        (refinement_bits, "refinement_bits")):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    prefix = 0
    revealed = 0
    chunk = initial_bits
    certified_lower = Fraction(0)
    certified_upper = Fraction(1)
    while True:
        prefix = (prefix << chunk) | _random_bits(rng, chunk)
        revealed += chunk
        raw_lower, raw_upper = probability_bounds(revealed + 32)
        lower = exact_real_fraction(raw_lower, name="lower probability bound")
        upper = exact_real_fraction(raw_upper, name="upper probability bound")
        if lower > upper:
            raise ValueError("probability_bounds returned an inverted interval")
        if lower > 1 or upper < 0:
            raise ValueError("the Bernoulli probability must lie in [0, 1]")
        lower = max(certified_lower, lower)
        upper = min(certified_upper, upper)
        if lower > upper:
            raise ValueError("probability_bounds returned inconsistent intervals")
        certified_lower, certified_upper = lower, upper
        denominator = 1 << revealed
        # U lies in [prefix/denominator, (prefix+1)/denominator).
        if Fraction(prefix + 1, denominator) <= lower:
            return True
        if Fraction(prefix, denominator) >= upper:
            return False
        chunk = refinement_bits


def bernoulli_from_log_bounds(
    log_bounds: LogBounds,
    rng: Any = None,
    *,
    initial_bits: int = 64,
    refinement_bits: int = 32,
) -> bool:
    """Sample a Bernoulli from certified, refinable bounds on ``log(p)``.

    ``log_bounds(bits)`` must return lower and upper bounds that contain the
    true ``log(p)`` and converge to it as ``bits`` increases.  Bounds may be
    :class:`Fraction`, :class:`Decimal`, integers, or finite real values whose
    decimal rendering is to be treated as exact.  The true probability must
    lie in ``[0, 1]``.

    One uniform real ``U`` is revealed in successive binary chunks.  At every
    stage both ``U`` and ``log(p)`` are intervals; a decision is made only when
    those intervals are disjoint.  Since a uniform real equals the boundary
    with probability zero, the loop terminates almost surely.  No fixed-bit
    probability rounding is performed.
    """

    if not callable(log_bounds):
        raise TypeError("log_bounds must be callable")
    for value, name in (
        (initial_bits, "initial_bits"),
        (refinement_bits, "refinement_bits"),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    prefix = 0
    revealed = 0
    chunk = initial_bits
    certified_lower: Fraction | None = None
    certified_upper: Fraction | None = None
    while True:
        prefix = (prefix << chunk) | _random_bits(rng, chunk)
        revealed += chunk
        denominator = 1 << revealed

        raw_lower, raw_upper = log_bounds(revealed + 32)
        probability_lower = exact_real_fraction(
            raw_lower, name="lower log bound"
        )
        probability_upper = exact_real_fraction(
            raw_upper, name="upper log bound"
        )
        if probability_lower > probability_upper:
            raise ValueError("log_bounds returned an inverted interval")
        if probability_lower > 0:
            raise ValueError("the Bernoulli probability exceeds one")
        # A certified interval may harmlessly straddle zero while refining a
        # probability equal or close to one.  Intersect it with log(p) <= 0.
        probability_upper = min(Fraction(0), probability_upper)
        # Intersect all valid enclosures seen so far.  A certified provider
        # need not return syntactically nested intervals, only intervals that
        # all contain the same true value.
        if certified_lower is not None:
            probability_lower = max(certified_lower, probability_lower)
            probability_upper = min(certified_upper, probability_upper)  # type: ignore[arg-type]
            if probability_lower > probability_upper:
                raise ValueError("log_bounds returned inconsistent intervals")
        certified_lower = probability_lower
        certified_upper = probability_upper
        if probability_lower == probability_upper == 0:
            return True

        uniform_upper = Fraction(prefix + 1, denominator)
        _, log_uniform_upper = _ln_fraction_interval(
            uniform_upper, bits=revealed + 32
        )
        # b <= p follows from upper(log b) <= lower(log p).
        if log_uniform_upper <= probability_lower:
            return True

        if prefix:
            uniform_lower = Fraction(prefix, denominator)
            log_uniform_lower, _ = _ln_fraction_interval(
                uniform_lower, bits=revealed + 32
            )
            # a >= p follows from lower(log a) >= upper(log p).
            if log_uniform_lower >= probability_upper:
                return False
        chunk = refinement_bits


def bernoulli_exp_exact(
    log_probability: Any,
    rng: Any = None,
    *,
    initial_bits: int = 64,
    refinement_bits: int = 32,
) -> bool:
    """Sample a Bernoulli with probability ``exp(log_probability)``.

    The finite input is interpreted as an exact real number.  Unlike a
    fixed-width comparison, ``initial_bits`` only controls the first lazy
    refinement chunk and has no effect on the returned distribution.
    """

    exact_log = exact_real_fraction(log_probability, name="log_probability")
    if exact_log > 0:
        raise ValueError("log_probability must be non-positive")
    if exact_log == 0:
        return True
    return bernoulli_from_log_bounds(
        lambda _bits: (exact_log, exact_log),
        rng,
        initial_bits=initial_bits,
        refinement_bits=refinement_bits,
    )


def _tail_radius(sigma: Decimal, tail_bits: int) -> int:
    """Choose a conservative relative-weight cutoff radius.

    At this radius an individual Gaussian weight is below roughly
    ``2**(-(tail_bits+16))`` times the peak.  Extra lattice points cover the
    half-integer displacement of the chosen integral base point.
    """

    # This ceiling determines which lattice points are present, so taking it
    # after a binary64 conversion can remove a point when the exact product is
    # just above an integer.  Retain every supplied decimal digit and bias the
    # last working-precision ulp upward before applying the ceiling.
    source_digits = len(sigma.as_tuple().digits)
    with localcontext() as context:
        context.prec = max(_decimal_digits(tail_bits), source_digits + 20)
        cutoff = sigma * (
            Decimal(2) * Decimal(2).ln() * Decimal(tail_bits + 16)
        ).sqrt()
        cutoff = cutoff.next_plus(context=context)
        radius = int(cutoff.to_integral_value(rounding=ROUND_CEILING)) + 3
    if radius > 1_000_000:
        raise ValueError("the requested Gaussian support is impractically large")
    return max(2, radius)


def _nearest_integer(center: Decimal) -> int:
    return int(center.to_integral_value(rounding=ROUND_HALF_EVEN))


def _nearest_fraction(center: Fraction) -> int:
    """Nearest integer with exact ties-to-even for a rational input."""

    lower = center.numerator // center.denominator
    remainder = center - lower
    doubled = 2 * remainder
    if doubled < 1:
        return lower
    if doubled > 1:
        return lower + 1
    return lower if lower % 2 == 0 else lower + 1


@dataclass(frozen=True, slots=True)
class DiscreteGaussianTable:
    """An explicitly approximate fixed-point finite Gaussian CDF."""

    support: tuple[int, ...]
    frequencies: tuple[int, ...]
    cumulative: tuple[int, ...]
    denominator: int
    center: Decimal
    sigma: Decimal
    precision_bits: int
    tail_bits: int

    def __post_init__(self) -> None:
        if not self.support or len(self.support) != len(self.frequencies):
            raise ValueError("support and frequencies must be non-empty and aligned")
        if len(self.cumulative) != len(self.support):
            raise ValueError("cumulative frequencies have the wrong length")
        if any(freq < 0 for freq in self.frequencies):
            raise ValueError("frequencies cannot be negative")
        if self.cumulative[-1] != self.denominator:
            raise ValueError("frequencies do not sum to the denominator")

    def sample(self, rng: Any = None) -> int:
        """Draw exactly from this fixed-point table."""

        value = _random_bits(rng, self.precision_bits)
        index = bisect_right(self.cumulative, value)
        return self.support[index]

    def probability(self, value: int) -> Decimal:
        """Return the table's quantized probability at ``value``."""

        try:
            index = self.support.index(value)
        except ValueError:
            return Decimal(0)
        return Decimal(self.frequencies[index]) / Decimal(self.denominator)


def discrete_gaussian_table(
    center: Any,
    sigma: Any,
    *,
    precision_bits: int = 128,
    tail_bits: int = 192,
) -> DiscreteGaussianTable:
    """Build an approximate finite table for ``D_{Z,sigma,center}``.

    Integer frequencies are assigned with deterministic largest-remainder
    rounding, so they sum exactly to ``2**precision_bits``.  This helper is
    useful for tests and codec modelling, but is not the formal sampler.
    """

    _validate_precision(precision_bits, tail_bits)
    center_d, sigma_d = _validate_center_sigma(center, sigma)
    radius = _tail_radius(sigma_d, tail_bits)
    base = _nearest_integer(center_d)
    support = tuple(range(base - radius, base + radius + 1))
    denominator = 1 << precision_bits

    with localcontext() as context:
        context.prec = _decimal_digits(max(precision_bits, tail_bits))
        two_sigma_squared = Decimal(2) * sigma_d * sigma_d
        weights = tuple(
            (-((Decimal(value) - center_d) ** 2) / two_sigma_squared).exp()
            for value in support
        )
        total = sum(weights, Decimal(0))
        scaled = tuple(weight * denominator / total for weight in weights)
        frequencies = [
            int(value.to_integral_value(rounding=ROUND_FLOOR)) for value in scaled
        ]
        missing = denominator - sum(frequencies)
        if missing < 0 or missing > len(frequencies):
            raise ArithmeticError("fixed-point Gaussian normalization failed")
        fractions = [value - integer for value, integer in zip(scaled, frequencies)]
        # Stable secondary key makes ties reproducible across Python versions.
        order = sorted(
            range(len(support)),
            key=lambda index: (fractions[index], -abs(support[index] - base), -index),
            reverse=True,
        )
        for index in order[:missing]:
            frequencies[index] += 1

    cumulative: list[int] = []
    running = 0
    for frequency in frequencies:
        running += frequency
        cumulative.append(running)
    return DiscreteGaussianTable(
        support=support,
        frequencies=tuple(frequencies),
        cumulative=tuple(cumulative),
        denominator=denominator,
        center=center_d,
        sigma=sigma_d,
        precision_bits=precision_bits,
        tail_bits=tail_bits,
    )


def gaussian_mass(
    center: Any,
    sigma: Any,
    *,
    precision_bits: int = 160,
    tail_bits: int = 256,
) -> Decimal:
    """Approximate ``sum_z exp(-(z-center)^2/(2*sigma^2))``.

    The computation uses Decimal arithmetic and an explicitly chosen finite
    support.  It is suitable for correction diagnostics; the security module
    owns the separate theta-function bounds used by the paper.
    """

    _validate_precision(precision_bits, tail_bits)
    center_d, sigma_d = _validate_center_sigma(center, sigma)
    radius = _tail_radius(sigma_d, tail_bits)
    base = _nearest_integer(center_d)
    with localcontext() as context:
        context.prec = _decimal_digits(max(precision_bits, tail_bits))
        two_sigma_squared = Decimal(2) * sigma_d * sigma_d
        result = sum(
            (
                (-((Decimal(value) - center_d) ** 2) / two_sigma_squared).exp()
                for value in range(base - radius, base + radius + 1)
            ),
            Decimal(0),
        )
        return +result


def log_gaussian_mass(
    center: Any,
    sigma: Any,
    *,
    precision_bits: int = 160,
    tail_bits: int = 256,
) -> Decimal:
    """Natural logarithm of :func:`gaussian_mass`."""

    with localcontext() as context:
        context.prec = _decimal_digits(max(precision_bits, tail_bits))
        return +gaussian_mass(
            center,
            sigma,
            precision_bits=precision_bits,
            tail_bits=tail_bits,
        ).ln()


# Common notation used in the correction proof.
log_rho_z = log_gaussian_mass


def sigma_to_paper_width(sigma: Any, *, precision_bits: int = 160) -> Decimal:
    """Convert standard deviation ``sigma`` to ``s=sqrt(2*pi)*sigma``."""

    sigma_d = _as_decimal(sigma, "sigma")
    if sigma_d <= 0:
        raise ValueError("sigma must be strictly positive")
    with localcontext() as context:
        context.prec = _decimal_digits(precision_bits)
        return +(Decimal(2) * _PI_DECIMAL).sqrt() * sigma_d


def paper_width_to_sigma(width: Any, *, precision_bits: int = 160) -> Decimal:
    """Convert paper width ``s`` to ordinary standard deviation."""

    width_d = _as_decimal(width, "width")
    if width_d <= 0:
        raise ValueError("width must be strictly positive")
    with localcontext() as context:
        context.prec = _decimal_digits(precision_bits)
        return +width_d / (Decimal(2) * _PI_DECIMAL).sqrt()


def _sample_two_sided_geometric(
    numerator: int, denominator: int, rng: Any
) -> int:
    """Sample an exact two-sided geometric with ratio ``numerator/denominator``.

    Its mass is ``A*r**abs(z)``, where ``A=(1-r)/(1+r)``.  All branch and
    geometric decisions are rational and use unbiased ``randbelow`` calls.
    """

    if not 0 < numerator < denominator:
        raise ValueError("the geometric ratio must lie strictly between 0 and 1")
    gap = denominator - numerator
    branch = _randbelow(rng, denominator + numerator)
    if branch < gap:
        return 0
    negative = branch < denominator
    magnitude = 1
    while _randbelow(rng, denominator) >= gap:
        magnitude += 1
    return -magnitude if negative else magnitude


def _proposal_ratio(sigma: Fraction) -> tuple[int, int, Fraction]:
    """Choose an exact rational geometric ratio and an upper bound on ``-ln r``.

    For the Falcon++ leaf range the result is the simple ratio 1/2.  Wider
    toy inputs select ``r=1-1/D`` with dyadic ``D`` on the scale of sigma, so
    the acceptance rate does not collapse like ``exp(-sigma**2/2)``.
    ``-ln(1-x) <= x/(1-x)`` supplies the exact rational upper bound.
    """

    denominator = 2
    while sigma > denominator:
        denominator <<= 1
    numerator = denominator - 1
    log_ratio_upper = Fraction(1, numerator)
    return numerator, denominator, log_ratio_upper


def _exact_gaussian_log_bounds(
    rational_part: Fraction,
    log_ratio_multiplier: int,
    log_ratio: Fraction,
) -> LogBounds:
    """Build bounds for ``rational_part + multiplier*ln(log_ratio)``."""

    if log_ratio_multiplier < 0:
        raise ValueError("log_ratio_multiplier cannot be negative")
    if log_ratio <= 1:
        raise ValueError("log_ratio must exceed one")

    def bounds(bits: int) -> tuple[Fraction, Fraction]:
        if log_ratio_multiplier == 0:
            return rational_part, rational_part
        lower, upper = _cached_ln_interval(log_ratio, bits)
        return (
            rational_part + log_ratio_multiplier * lower,
            rational_part + log_ratio_multiplier * upper,
        )

    return bounds


def _nonpositive_exp_probability_bounds(
    lower: Fraction, upper: Fraction, *, bits: int
) -> tuple[Fraction, Fraction]:
    """Exponentiate a nonpositive enclosure without huge tiny denominators.

    Since exp(-b) <= 2**(-b) for positive integer b, sufficiently negative
    arguments can use a coarse dyadic enclosure.  It converges as the lazy
    comparison requests more bits, without imposing a support cutoff.
    """

    if bits <= 0:
        raise ValueError("bits must be positive")
    if lower > upper or lower > 0:
        raise ArithmeticError("invalid nonpositive log-probability interval")
    upper = min(Fraction(0), upper)
    if upper <= -bits:
        return Fraction(0), Fraction(1, 1 << bits)
    if upper == 0:
        value_upper = Fraction(1)
    else:
        # Import lazily: correction imports this module's shared scalar and
        # Bernoulli helpers.  Its directed Decimal exp implementation is reused
        # only once both modules are initialized.
        from .correction import _exp_fraction_bounds

        _, value_upper = _exp_fraction_bounds(upper, upper, bits=bits)
    if lower <= -bits:
        value_lower = Fraction(0)
    elif lower == 0:
        value_lower = Fraction(1)
    else:
        from .correction import _exp_fraction_bounds

        value_lower, _ = _exp_fraction_bounds(lower, lower, bits=bits)
    return value_lower, value_upper


def _scaled_exp_probability_bounds(
    rational_numerator: int, rational_denominator: int,
    magnitude: int, bits: int,
) -> tuple[int, int]:
    """Enclose ``2**magnitude * exp(numerator/denominator)`` on a dyadic grid.

    The caller supplies the nonpositive rational part of the *unchanged*
    half-geometric rejection probability.  Returned integers divided by
    ``2**bits`` enclose that probability; they are not a rounded probability
    used for a finite-bit coin.  The lazy comparator refines this grid without
    a precision limit whenever it cannot yet make a certified decision.

    Work precision includes ``magnitude``: multiplication by ``2**magnitude``
    must not amplify an unaccounted exponential error.  For an extreme tail,
    exp(-x) <= 2**(-x) gives a converging coarse enclosure *before* constructing
    either an enormous power of two or a tiny Decimal/Fraction.  In particular
    the upper endpoint remains positive, so no tail is cut off.
    """

    if rational_denominator <= 0 or rational_numerator > 0:
        raise ValueError("expected a nonpositive rational exponent")
    if magnitude < 0 or bits <= 0:
        raise ValueError("magnitude must be nonnegative and bits positive")
    work_bits = bits + magnitude
    if -rational_numerator >= work_bits * rational_denominator:
        return 0, 1
    scale = 1 << bits
    if rational_numerator == 0:
        if magnitude:
            raise ArithmeticError("the Bernoulli probability exceeds one")
        return scale, scale

    # Decimal.exp is correctly rounded (ROUND_HALF_EVEN).  Directed rational
    # division followed by one outward representable step therefore encloses
    # the exact exponential.  This makes precisely two exp calls, no log(U),
    # no Fraction normalization, and no dependence on the ambient context.
    with localcontext() as context:
        context.prec = _decimal_digits(work_bits, guard_digits=12)
        context.Emax = MAX_EMAX
        context.Emin = MIN_EMIN
        numerator = Decimal(rational_numerator)
        denominator = Decimal(rational_denominator)
        context.rounding = ROUND_FLOOR
        argument_lower = numerator / denominator
        context.rounding = ROUND_CEILING
        argument_upper = numerator / denominator
        context.rounding = ROUND_HALF_EVEN
        rounded_lower = argument_lower.exp()
        rounded_upper = argument_upper.exp()
        if rounded_upper == 0:
            # The coarse-tail branch should cover this for feasible inputs.
            # Fail rather than silently replacing a positive mass by zero.
            raise ArithmeticError("exponent is outside the certified Decimal range")
        lower_decimal = (Decimal(0) if rounded_lower == 0 else
                         rounded_lower.next_minus(context=context))
        upper_decimal = rounded_upper.next_plus(context=context)
    lower_numerator, lower_denominator = lower_decimal.as_integer_ratio()
    upper_numerator, upper_denominator = upper_decimal.as_integer_ratio()
    lower = (lower_numerator << work_bits) // lower_denominator
    upper = ((upper_numerator << work_bits) + upper_denominator - 1) // upper_denominator
    if lower > scale:
        raise ArithmeticError("the Bernoulli probability exceeds one")
    return max(0, lower), min(scale, upper)


def _bernoulli_scaled_probability_bounds(
    probability_bounds: Callable[[int], tuple[int, int]],
    rng: Any = None, *, initial_bits: int = 64, refinement_bits: int = 32,
) -> bool:
    """Exact lazy comparison with integer, dyadically scaled probability bounds.

    This is the integer-arithmetic counterpart of
    :func:`bernoulli_from_probability_bounds`.  A single random prefix is
    extended (never restarted); all previous enclosures are intersected after
    exact rescaling.  Work-grid bits are not final probability precision.
    """

    for value in (initial_bits, refinement_bits):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("random-bit blocks must be positive integers")
    prefix = 0
    revealed = 0
    chunk = initial_bits
    previous_bits = 0
    certified_lower = 0
    certified_upper = 1
    while True:
        prefix = (prefix << chunk) | _random_bits(rng, chunk)
        revealed += chunk
        bound_bits = revealed + 32
        lower, upper = probability_bounds(bound_bits)
        if (not isinstance(lower, int) or not isinstance(upper, int)
                or isinstance(lower, bool) or isinstance(upper, bool)):
            raise TypeError("scaled probability bounds must be integers")
        if lower > upper or lower > (1 << bound_bits) or upper < 0:
            raise ValueError("invalid scaled probability interval")
        shift = bound_bits - previous_bits
        lower = max(lower, certified_lower << shift)
        upper = min(upper, certified_upper << shift)
        if lower > upper:
            raise ValueError("scaled probability bounds returned inconsistent intervals")
        certified_lower, certified_upper = lower, upper
        previous_bits = bound_bits
        if ((prefix + 1) << 32) <= lower:
            return True
        if (prefix << 32) >= upper:
            return False
        chunk = refinement_bits


@dataclass(frozen=True)
class _PythonWidthSetup:
    """Exact immutable, center-independent constants for one leaf width."""

    sigma: Fraction
    proposal_numerator: int
    proposal_denominator: int
    numerator_squared: int
    denominator_squared: int
    numerator_fourth: int
    denominator_fourth: int


def _python_center_parameters(
    center: Fraction, setup: _PythonWidthSetup,
) -> tuple[int, int, int, int, int, int]:
    """Construct the original half-geometric rational exponent using integers.

    With r=a/b and sigma=u/v, the old exponent is exactly

      R = -(z-r)^2/(2*sigma^2) - abs(r) - sigma^2/2
        = -((z*b-a)^2*v^4 + 2*abs(a)*b*u^2*v^2 + b^2*u^4)
          / (2*b^2*u^2*v^2).

    Keeping the common denominator unreduced avoids repeated Fraction GCDs.
    Only the center-local result is returned; no center is retained in a cache.
    """

    base = _nearest_fraction(center)
    denominator = center.denominator
    residual_numerator = center.numerator - base * denominator
    square_denominator = denominator * denominator
    squared_product = setup.numerator_squared * setup.denominator_squared
    constant = (2 * abs(residual_numerator) * denominator * squared_product
                + square_denominator * setup.numerator_fourth)
    exponent_denominator = 2 * square_denominator * squared_product
    return (base, residual_numerator, denominator, setup.denominator_fourth,
            constant, exponent_denominator)


class DiscreteGaussianSampler:
    """Exact lazy rejection sampler for ``D_{Z,sigma,center}``.

    Parameters
    ----------
    precision_bits:
        First random-bit/refinement block for exact mode.  It is only a
        performance hint and introduces no probability quantization.  In the
        experimental finite-table methods it remains the CDF resolution.
    tail_bits:
        Relative tail target used only by experimental finite-table methods.
    mode:
        ``"exact"`` (default) and its backwards-compatible alias
        ``"high_precision"`` use the infinite proposal and lazy interval
        comparison.  ``"fast"`` uses a finite quantized proposal and
        binary64 exponential and is only a diagnostic accelerator.
    method:
        ``"rejection"`` (default) is exact in exact/high-precision mode.
        ``"inversion"`` explicitly selects a finite, fixed-point approximate
        CDF and exists only as an independent experimental test path.
    zero_center_backend:
        ``"reference"`` preserves the existing seeded execution.  The opt-in
        ``"cached_probability"`` path uses the same exact geometric proposal
        and acceptance law at integral centers, but compares certified
        probability bounds directly.  Its bounded, per-sampler cache stores
        width/proposal constants only, never accepted vectors or centers.
    general_center_backend:
        ``"reference"`` keeps all existing behavior.  Opt-in
        ``"exact_python_fast"`` preserves the same infinite proposal/envelope
        and exact finite-input distribution, but uses direct probability
        bounds and integer arithmetic at nonintegral centers with sigma <= 2.
        Other widths and all integral centers retain their existing paths.
        This is still variable-time research Python, not a constant-time
        production sampler.
    general_center_initial_bits, general_center_refinement_bits:
        Initial and subsequent random-bit blocks for the opt-in backend only.
        These are performance hints, never a precision cap or probability
        quantization.  Existing ``precision_bits`` semantics are unchanged.
    general_center_cache_capacity:
        Maximum number of immutable exact width setups retained by the opt-in
        backend.  ``prepare_widths(..., max_widths=2*n)`` can replace this cache
        for a new key without changing any serialized private-key format.
    """

    def __init__(
        self,
        *,
        precision_bits: int = 128,
        tail_bits: int = 192,
        mode: str = "exact",
        method: str = "rejection",
        max_attempts: int | None = None,
        zero_center_backend: str = "reference",
        general_center_backend: str = "reference",
        general_center_initial_bits: int = 64,
        general_center_refinement_bits: int = 32,
        general_center_cache_capacity: int = 2048,
    ) -> None:
        _validate_precision(precision_bits, tail_bits)
        if mode not in {"exact", "high_precision", "fast"}:
            raise ValueError("mode must be 'exact', 'high_precision', or 'fast'")
        if method not in {"rejection", "inversion"}:
            raise ValueError("method must be 'rejection' or 'inversion'")
        if zero_center_backend not in {"reference", "cached_probability"}:
            raise ValueError("zero_center_backend must be 'reference' or 'cached_probability'")
        if zero_center_backend == "cached_probability" and (
            mode not in {"exact", "high_precision"} or method != "rejection"
        ):
            raise ValueError("cached_probability requires exact rejection sampling")
        if general_center_backend not in {"reference", "exact_python_fast"}:
            raise ValueError("general_center_backend must be 'reference' or 'exact_python_fast'")
        if general_center_backend == "exact_python_fast" and (
            mode not in {"exact", "high_precision"} or method != "rejection"
        ):
            raise ValueError("exact_python_fast requires exact rejection sampling")
        for value, name in (
            (general_center_initial_bits, "general_center_initial_bits"),
            (general_center_refinement_bits, "general_center_refinement_bits"),
            (general_center_cache_capacity, "general_center_cache_capacity"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if max_attempts is not None and (
            not isinstance(max_attempts, int)
            or isinstance(max_attempts, bool)
            or max_attempts <= 0
        ):
            raise ValueError("max_attempts must be None or a positive integer")
        self.precision_bits = precision_bits
        self.tail_bits = tail_bits
        self.mode = mode
        self.method = method
        self.max_attempts = max_attempts
        self.zero_center_backend = zero_center_backend
        self.general_center_backend = general_center_backend
        self.general_center_initial_bits = general_center_initial_bits
        self.general_center_refinement_bits = general_center_refinement_bits
        self.general_center_cache_capacity = general_center_cache_capacity
        self._python_width_setups: OrderedDict[tuple[Any, ...], _PythonWidthSetup] = OrderedDict()
        self._python_width_hits = 0
        self._python_width_misses = 0
        self._python_prepared_widths = 0
        self._proposal_tables: dict[Decimal, DiscreteGaussianTable] = {}
        self._zero_center_setups: OrderedDict[Fraction, tuple[Any, ...]] = OrderedDict()
        self._zero_center_probability_cache: OrderedDict[
            tuple[Fraction, int, int], tuple[Fraction, Fraction]
        ] = OrderedDict()
        self._zero_center_cache_hits = 0
        self._zero_center_cache_misses = 0

    @staticmethod
    def _proposal_sigma(target_sigma: Decimal) -> Decimal:
        # Geometric buckets keep sigma_p between 1.25 and 1.875 times sigma.
        # The minimum bucket also handles toy tests with very narrow targets.
        required = Decimal("1.25") * target_sigma
        bucket = Decimal("0.125")
        while bucket < required:
            bucket *= Decimal("1.5")
        if bucket <= target_sigma:  # Defensive against altered rounding modes.
            bucket *= Decimal("1.5")
        return bucket

    def _proposal(self, target_sigma: Decimal) -> DiscreteGaussianTable:
        proposal_sigma = self._proposal_sigma(target_sigma)
        table = self._proposal_tables.get(proposal_sigma)
        if table is None:
            table = discrete_gaussian_table(
                0,
                proposal_sigma,
                precision_bits=self.precision_bits,
                tail_bits=self.tail_bits,
            )
            self._proposal_tables[proposal_sigma] = table
        return table

    def _bernoulli_fast(self, log_probability: Decimal, rng: Any) -> bool:
        logp = float(log_probability)
        if logp >= 0.0:
            return True
        # A 64-bit comparison is enough for the explicitly non-authoritative
        # fast mode.  Very small probabilities safely round to zero.
        probability = 0.0 if logp < -750.0 else math.exp(logp)
        threshold = min(1 << 64, int(probability * (1 << 64)))
        return _random_bits(rng, 64) < threshold

    def sample(self, center: Any, sigma: Any, rng: Any = None) -> int:
        """Draw one integer from ``D_{Z,sigma,center}``."""

        if self.method == "rejection" and self.mode in {
            "exact",
            "high_precision",
        }:
            center_q = exact_real_fraction(center, name="center")
            if (self.general_center_backend == "exact_python_fast"
                    and center_q.denominator != 1):
                setup = self._python_width_setup(sigma)
                if setup.proposal_denominator == 2:
                    return self._sample_python_exact(center_q, setup, rng)
                # Decide fallback using exact width before consuming randomness.
                return self._sample_exact(center_q, setup.sigma, rng)
            sigma_q = exact_real_fraction(sigma, name="sigma")
            if sigma_q <= 0:
                raise ValueError("sigma must be strictly positive")
            if self.zero_center_backend == "cached_probability" and center_q.denominator == 1:
                return self._sample_integral_exact(center_q.numerator, sigma_q, rng)
            return self._sample_exact(center_q, sigma_q, rng)

        center_d, sigma_d = _validate_center_sigma(center, sigma)
        if self.method == "inversion":
            return discrete_gaussian_table(
                center_d,
                sigma_d,
                precision_bits=self.precision_bits,
                tail_bits=self.tail_bits,
            ).sample(rng)

        return self._sample_fast(center_d, sigma_d, rng)

    @staticmethod
    def _python_width_key(sigma: Any) -> tuple[Any, ...]:
        # Binary64 leaves are the hot path: use the exact finite immutable
        # float itself as a typed key, avoiding Fraction conversion on hits.
        # All other inputs are canonicalized to an immutable exact Fraction;
        # no mutable mpmath object or ambient-precision hash is retained.
        if type(sigma) is float:
            if not math.isfinite(sigma) or sigma <= 0:
                raise ValueError("sigma must be finite and strictly positive")
            return float, sigma
        sigma_q = exact_real_fraction(sigma, name="sigma")
        if sigma_q <= 0:
            raise ValueError("sigma must be strictly positive")
        return Fraction, sigma_q

    @staticmethod
    def _make_python_width_setup(key: tuple[Any, ...]) -> _PythonWidthSetup:
        sigma_q = (Fraction.from_float(key[1]) if key[0] is float else key[1])
        numerator, denominator, _ = _proposal_ratio(sigma_q)
        numerator_squared = sigma_q.numerator ** 2
        denominator_squared = sigma_q.denominator ** 2
        return _PythonWidthSetup(
            sigma_q, numerator, denominator, numerator_squared,
            denominator_squared, numerator_squared ** 2, denominator_squared ** 2,
        )

    def _python_width_setup(self, sigma: Any) -> _PythonWidthSetup:
        key = self._python_width_key(sigma)
        setup = self._python_width_setups.get(key)
        if setup is not None:
            self._python_width_hits += 1
            self._python_width_setups.move_to_end(key)
            return setup
        self._python_width_misses += 1
        setup = self._make_python_width_setup(key)
        self._python_width_setups[key] = setup
        if len(self._python_width_setups) > self.general_center_cache_capacity:
            self._python_width_setups.popitem(last=False)
        return setup

    def prepare_widths(self, widths: Iterable[Any], *, max_widths: int | None = None) -> None:
        """Atomically replace center-free width setups for one signing key.

        A successful call clears the previous key's cache and counters.  If
        validation fails, the old cache is left untouched.  Duplicate exact
        typed widths share an entry.  More unique widths than the chosen
        capacity fail explicitly, rather than silently preparing a thrashing
        partial cache.  This workspace is never part of private-key encoding.
        """

        if self.general_center_backend != "exact_python_fast":
            raise ValueError("prepare_widths requires exact_python_fast")
        capacity = self.general_center_cache_capacity if max_widths is None else max_widths
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("max_widths must be a positive integer")
        prepared: OrderedDict[tuple[Any, ...], _PythonWidthSetup] = OrderedDict()
        for width in widths:
            key = self._python_width_key(width)
            if key not in prepared:
                if len(prepared) >= capacity:
                    raise ValueError("distinct widths exceed the prepared cache capacity")
                prepared[key] = self._make_python_width_setup(key)
        self._python_width_setups = prepared
        self.general_center_cache_capacity = capacity
        self._python_width_hits = self._python_width_misses = 0
        self._python_prepared_widths = len(prepared)

    def clear_prepared_widths(self) -> None:
        """Forget all general-center width setups and their nonsecret counters."""

        self._python_width_setups.clear()
        self._python_width_hits = self._python_width_misses = 0
        self._python_prepared_widths = 0

    def general_center_cache_info(self) -> dict[str, int]:
        """Return counts only: never leaf widths, centers, or sampled integers.

        ``prepared_widths`` counts distinct typed widths from the last
        successful explicit preparation; subsequent lazy insertions or LRU
        evictions do not change that setup-work count.
        """

        return {
            "hits": self._python_width_hits,
            "misses": self._python_width_misses,
            "width_entries": len(self._python_width_setups),
            "max_width_entries": self.general_center_cache_capacity,
            "prepared_widths": self._python_prepared_widths,
        }

    def _sample_python_exact(
        self, center_q: Fraction, setup: _PythonWidthSetup, rng: Any,
    ) -> int:
        base, residual, denominator, factor, constant, exponent_denominator = (
            _python_center_parameters(center_q, setup)
        )
        attempts = 0
        while self.max_attempts is None or attempts < self.max_attempts:
            attempts += 1
            # Exactly the same infinite geometric proposal and rational
            # envelope as _sample_exact.  Only the decision arithmetic differs.
            offset = _sample_two_sided_geometric(1, 2, rng)
            displacement = offset * denominator - residual
            rational_numerator = -(displacement * displacement * factor + constant)
            magnitude = abs(offset)
            if _bernoulli_scaled_probability_bounds(
                lambda bits: _scaled_exp_probability_bounds(
                    rational_numerator, exponent_denominator, magnitude, bits,
                ),
                rng,
                initial_bits=self.general_center_initial_bits,
                refinement_bits=self.general_center_refinement_bits,
            ):
                return base + offset
        raise RuntimeError(
            "exact discrete Gaussian rejection sampler exceeded the explicit "
            "diagnostic max_attempts cap"
        )

    def _zero_center_setup(self, sigma_q: Fraction) -> tuple[Any, ...]:
        setup = self._zero_center_setups.get(sigma_q)
        if setup is not None:
            self._zero_center_setups.move_to_end(sigma_q)
            return setup
        sigma_squared = sigma_q * sigma_q
        numerator, denominator, log_ratio_upper = _proposal_ratio(sigma_q)
        setup = (
            numerator, denominator, sigma_squared,
            sigma_squared * log_ratio_upper * log_ratio_upper / 2,
            Fraction(denominator, numerator),
        )
        self._zero_center_setups[sigma_q] = setup
        if len(self._zero_center_setups) > 64:
            self._zero_center_setups.popitem(last=False)
        return setup

    def _zero_center_acceptance_bounds(
        self, sigma_q: Fraction, magnitude: int, bits: int
    ) -> tuple[Fraction, Fraction]:
        """Same proposal cancellation law as _sample_exact with residual=0."""

        if sigma_q <= 0 or magnitude < 0:
            raise ValueError("invalid zero-center proposal parameters")
        key = (sigma_q, magnitude, bits)
        result = self._zero_center_probability_cache.get(key)
        if result is not None:
            self._zero_center_probability_cache.move_to_end(key)
            self._zero_center_cache_hits += 1
            return result
        self._zero_center_cache_misses += 1
        _, _, sigma_squared, bound, inverse_ratio = self._zero_center_setup(sigma_q)
        rational_part = -Fraction(magnitude * magnitude, 1) / (2 * sigma_squared) - bound
        lower, upper = _exact_gaussian_log_bounds(
            rational_part, magnitude, inverse_ratio
        )(bits)
        result = _nonpositive_exp_probability_bounds(lower, upper, bits=bits)
        self._zero_center_probability_cache[key] = result
        if len(self._zero_center_probability_cache) > 256:
            self._zero_center_probability_cache.popitem(last=False)
        return result

    def zero_center_cache_info(self) -> dict[str, int]:
        """Nonsecret cache statistics, excluding widths and proposal values."""

        return {
            "hits": self._zero_center_cache_hits,
            "misses": self._zero_center_cache_misses,
            "probability_entries": len(self._zero_center_probability_cache),
            "width_entries": len(self._zero_center_setups),
            "max_probability_entries": 256,
            "max_width_entries": 64,
        }

    def _sample_integral_exact(self, base: int, sigma_q: Fraction, rng: Any) -> int:
        numerator, denominator, _, _, _ = self._zero_center_setup(sigma_q)
        attempts = 0
        while self.max_attempts is None or attempts < self.max_attempts:
            attempts += 1
            offset = _sample_two_sided_geometric(numerator, denominator, rng)
            magnitude = abs(offset)
            if bernoulli_from_probability_bounds(
                lambda bits: self._zero_center_acceptance_bounds(sigma_q, magnitude, bits),
                rng,
                initial_bits=self.precision_bits,
                refinement_bits=max(16, self.precision_bits // 2),
            ):
                return base + offset
        raise RuntimeError(
            "exact discrete Gaussian rejection sampler exceeded the explicit "
            "diagnostic max_attempts cap"
        )

    def _sample_exact(
        self, center_q: Fraction, sigma_q: Fraction, rng: Any
    ) -> int:
        """Exact infinite-support sampler using a geometric proposal.

        Let ``center = base + r`` with ``|r| <= 1/2`` and draw an exact
        two-sided geometric

        ``q(z) = A * ratio**|z|``.

        The function

        ``-(z-r)^2/(2*sigma^2) + |z|*(-ln(ratio))``

        is bounded above by ``L*|r| + sigma^2*L^2/2`` whenever
        ``L >= -ln(ratio)``.  We choose a rational ratio and rational ``L``.
        Accepting with the corresponding probability cancels ``q`` exactly,
        leaving the desired discrete Gaussian.  The adaptive ratio also keeps
        wide-sigma toy inputs practical.  Both the proposal and Bernoulli
        comparison have unbounded support/precision and terminate almost
        surely.
        """

        base = _nearest_fraction(center_q)
        residual = center_q - base
        sigma_squared = sigma_q * sigma_q
        numerator, denominator, log_ratio_upper = _proposal_ratio(sigma_q)
        bound = (
            log_ratio_upper * abs(residual)
            + sigma_squared * log_ratio_upper * log_ratio_upper / 2
        )
        inverse_ratio = Fraction(denominator, numerator)

        attempts = 0
        while self.max_attempts is None or attempts < self.max_attempts:
            attempts += 1
            offset = _sample_two_sided_geometric(numerator, denominator, rng)
            displacement = Fraction(offset) - residual
            rational_part = -(
                displacement * displacement / (2 * sigma_squared)
            ) - bound
            if bernoulli_from_log_bounds(
                _exact_gaussian_log_bounds(
                    rational_part, abs(offset), inverse_ratio
                ),
                rng,
                initial_bits=self.precision_bits,
                refinement_bits=max(16, self.precision_bits // 2),
            ):
                return base + offset
        raise RuntimeError(
            "exact discrete Gaussian rejection sampler exceeded the explicit "
            "diagnostic max_attempts cap"
        )

    def _sample_fast(
        self, center_d: Decimal, sigma_d: Decimal, rng: Any
    ) -> int:
        """Finite/quantized diagnostic path retained for speed comparisons."""

        base = _nearest_integer(center_d)
        residual = center_d - Decimal(base)
        proposal = self._proposal(sigma_d)
        proposal_sigma = proposal.sigma

        with localcontext() as context:
            context.prec = _decimal_digits(self.precision_bits)
            sigma_squared = sigma_d * sigma_d
            proposal_squared = proposal_sigma * proposal_sigma
            # Maximum over real z of
            # z^2/(2 sigma_p^2) - (z-r)^2/(2 sigma^2).
            log_bound = residual * residual / (
                Decimal(2) * (proposal_squared - sigma_squared)
            )
            inv_two_sigma_squared = Decimal(1) / (Decimal(2) * sigma_squared)
            inv_two_proposal_squared = Decimal(1) / (
                Decimal(2) * proposal_squared
            )

            attempt = 0
            while self.max_attempts is None or attempt < self.max_attempts:
                attempt += 1
                offset = proposal.sample(rng)
                offset_d = Decimal(offset)
                log_probability = (
                    offset_d * offset_d * inv_two_proposal_squared
                    - (offset_d - residual) ** 2 * inv_two_sigma_squared
                    - log_bound
                )
                # Decimal rounding may produce a tiny positive residue even
                # though the analytic bound is non-positive.
                if log_probability > 0:
                    log_probability = Decimal(0)
                accepted = self._bernoulli_fast(log_probability, rng)
                if accepted:
                    return base + offset
        raise RuntimeError(
            "fast approximate Gaussian rejection sampler exceeded the explicit "
            "diagnostic max_attempts cap"
        )

    def sample_vector(
        self,
        count: int,
        sigma: Any,
        rng: Any = None,
        *,
        center: Any = 0,
    ) -> list[int]:
        """Draw ``count`` independent samples with a common centre/width."""

        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError("count must be a non-negative integer")
        if (count and self.zero_center_backend == "cached_probability"
                and self.method == "rejection"
                and self.mode in {"exact", "high_precision"}):
            center_q = exact_real_fraction(center, name="center")
            sigma_q = exact_real_fraction(sigma, name="sigma")
            if sigma_q <= 0:
                raise ValueError("sigma must be strictly positive")
            if center_q.denominator == 1:
                return [self._sample_integral_exact(center_q.numerator, sigma_q, rng)
                        for _ in range(count)]
        return [self.sample(center, sigma, rng) for _ in range(count)]

    def sample_zero_centered(
        self, count: int, sigma: Any, rng: Any = None
    ) -> list[int]:
        """Convenience wrapper used by KeyGen."""

        return self.sample_vector(count, sigma, rng, center=0)


def sample_discrete_gaussian(
    center: Any,
    sigma: Any,
    rng: Any = None,
    *,
    precision_bits: int = 128,
    tail_bits: int = 192,
    mode: str = "exact",
    method: str = "rejection",
) -> int:
    """Stateless convenience wrapper around :class:`DiscreteGaussianSampler`."""

    sampler = DiscreteGaussianSampler(
        precision_bits=precision_bits,
        tail_bits=tail_bits,
        mode=mode,
        method=method,
    )
    return sampler.sample(center, sigma, rng)


__all__ = [
    "DiscreteGaussianSampler",
    "DiscreteGaussianTable",
    "bernoulli_exp_exact",
    "bernoulli_from_log_bounds",
    "bernoulli_from_probability_bounds",
    "discrete_gaussian_table",
    "exact_real_fraction",
    "gaussian_mass",
    "log_gaussian_mass",
    "log_rho_z",
    "paper_width_to_sigma",
    "sample_discrete_gaussian",
    "sigma_to_paper_width",
]
