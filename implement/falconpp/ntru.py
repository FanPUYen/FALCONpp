"""Exact NTRU solving primitives for the Falcon++ research implementation.

The recursive construction follows Falcon's public NTRUSolve description, but
this is an independent, deliberately readable implementation.  In particular,
all ring operations and all equation checks use Python integers.  An integer
interval fast path certifies Babai rounding and GSO decisions when possible.
Inconclusive Babai inputs retain the original mpmath precision schedule and
exact fallback; numerical agreement at two precisions remains a sanity check,
not a rigorous bound on floating-point rounding error.  The fast certificate
does not provide an end-to-end numerical proof for the entire scheme.

The sign convention throughout this module is::

    f * G - g * F = q  in Z[x] / (x**n + 1).

Two involutions occur in Falcon and must not be confused: recursive lifting
uses the Galois conjugate ``a(-x)``, while Babai reduction uses the adjoint
``a(x**-1)``.

References used to understand the algorithms (not copied source): Falcon
specification, Algorithms 5--7; Pornin--Prest, "More Efficient Algorithms for
the NTRU Key Generation Using the Field Norm"; and the tprest/falcon.py
educational implementation at https://github.com/tprest/falcon.py. Archived
reference copies are not runtime dependencies of this independent module.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from typing import Any, Sequence

from .certified_fft import (
    CertifiedBasisWorkspace,
    certified_babai_multiplier,
    certified_gso_squared_bounds,
)
from .polynomial import (
    Polynomial,
    add,
    adjoint,
    field_norm,
    galois_conjugate,
    lift,
    negacyclic_mul,
    sub,
    validate_power_of_two,
    verify_ntru,
)


_FAST_PATH_COUNTS = {
    "staged_attempts": 0, "staged_certified": 0, "staged_fallbacks": 0,
    "gso_attempts": 0, "gso_certified": 0, "gso_fallbacks": 0,
}


def fast_path_diagnostics() -> dict[str, int]:
    """Copy process-local aggregate counts; never exposes coefficients."""

    return dict(_FAST_PATH_COUNTS)


class NTRUError(ValueError):
    """Base class for a rejected or invalid NTRU construction."""


class NTRUNoSolutionError(NTRUError):
    """The recursive resultants do not permit an integral NTRU solution."""


class NTRUReductionError(NTRUError):
    """A numerically stable Babai reduction could not be completed."""


class NTRUReductionPrecisionError(NTRUReductionError):
    """The rounded Babai quotient did not stabilize at allowed precisions."""


class NTRUReductionStalledError(NTRUReductionError):
    """A stable Babai step failed to improve the reduction measure."""


@dataclass(frozen=True)
class ReductionConfig:
    """Public controls for deterministic Babai staging and its fallback.

    ``precision_schedule`` is measured in decimal digits.  When the integer
    interval path cannot certify a quotient, the unchanged fallback computes
    successive entries until ties-to-even rounding agrees twice and is
    separated from every half integer by the stated guard.
    The input integers are shifted to a fixed top-bit window, as in Falcon's
    staged reduction, so deeply recursive intermediate values remain practical.
    """

    precision_schedule: tuple[int, ...] = (80, 120, 180, 260, 400)
    window_bits: int = 192
    stage_bits: int = 8
    guard_digits: int = 20
    max_iterations: int = 2048

    def __post_init__(self) -> None:
        if len(self.precision_schedule) < 2:
            raise ValueError("precision_schedule needs at least two entries")
        if any(p < 30 for p in self.precision_schedule):
            raise ValueError("each reduction precision must be at least 30 dps")
        if tuple(sorted(set(self.precision_schedule))) != self.precision_schedule:
            raise ValueError("precision_schedule must be strictly increasing")
        if self.window_bits < 64:
            raise ValueError("window_bits must be at least 64")
        if self.stage_bits <= 0 or self.window_bits % self.stage_bits:
            raise ValueError("stage_bits must be positive and divide window_bits")
        if self.guard_digits < 8:
            raise ValueError("guard_digits must be at least 8")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")


@dataclass(frozen=True)
class ReductionStep:
    """Diagnostic record for one exact Babai update.

    Measures are safe summaries ``(max_coefficient_bits,
    squared_norm_bits)``.  Storing the enormous exact resultant norms would
    make even ``repr(solution)`` exceed Python's integer-to-string safety limit
    for real degree-1024 keys.

    ``precision_dps == -1`` denotes certified integer-interval rounding;
    zero denotes the exact rational fallback.  Positive values are mpmath dps.
    """

    iteration: int
    precision_dps: int
    scalar_shift: int
    multiplier_nonzero: int
    multiplier_max_bits: int
    before_measure: tuple[int, int]
    after_measure: tuple[int, int]


@dataclass(frozen=True)
class NTRUSolution:
    """A solved NTRU pair plus optional reduction diagnostics."""

    capital_f: Polynomial
    capital_g: Polynomial
    reduction_steps: tuple[ReductionStep, ...] = ()

    def __iter__(self):
        # Convenient compatibility with ``F, G = solve_ntru(...)``.
        yield self.capital_f
        yield self.capital_g


def _require_mpmath() -> Any:
    try:
        import mpmath  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - dependency installation path
        raise RuntimeError(
            "NTRU reduction requires mpmath; install implement/requirements.txt"
        ) from exc
    return mpmath


def integer_xgcd(a: int, b: int) -> tuple[int, int, int]:
    """Return nonnegative ``(d, u, v)`` such that ``u*a + v*b == d``."""

    old_r, r = int(a), int(b)
    old_u, u = 1, 0
    old_v, v = 0, 1
    while r:
        quotient = old_r // r
        old_r, r = r, old_r - quotient * r
        old_u, u = u, old_u - quotient * u
        old_v, v = v, old_v - quotient * v
    if old_r < 0:
        old_r, old_u, old_v = -old_r, -old_u, -old_v
    return old_r, old_u, old_v


def _coefficient_bits(*polynomials: Sequence[int]) -> int:
    return max(
        (abs(int(coefficient)).bit_length()
         for polynomial in polynomials for coefficient in polynomial),
        default=0,
    )


def _squared_norm(*polynomials: Sequence[int]) -> int:
    return sum(
        int(coefficient) * int(coefficient)
        for polynomial in polynomials
        for coefficient in polynomial
    )


def _reduction_measure(capital_f: Sequence[int], capital_g: Sequence[int]) -> tuple[int, int]:
    return (
        _coefficient_bits(capital_f, capital_g),
        _squared_norm(capital_f, capital_g),
    )


@lru_cache(maxsize=256)
def _transform_powers(
    degree: int, precision_bits: int, inverse: bool, twist: bool
) -> tuple[Any, ...]:
    """Cache only public FFT constants, separately for every bit precision.

    Repeated roots and their sequential powers were formerly rebuilt at each
    recursive node and each Babai quotient.  Repeating that exact arithmetic
    once per degree/precision preserves its rounded values; no secret inputs,
    quotient vectors, or sampling centers enter this bounded process cache.
    The key uses the actual mpmath bit precision, not a rounded dps display.
    """

    mp = _require_mpmath()
    with mp.workprec(precision_bits):
        sign = -1 if inverse else 1
        if twist:
            root = mp.exp(sign * mp.pi * mp.j / degree)
            count = degree
        else:
            root = mp.exp(sign * 2 * mp.pi * mp.j / degree)
            count = degree // 2
        power = mp.mpc(1)
        powers: list[Any] = []
        for _ in range(count):
            powers.append(power)
            power *= root
        return tuple(powers)


@lru_cache(maxsize=16)
def _fft_input_order(degree: int) -> tuple[int, ...]:
    """Public bit-reversal permutation; no input values enter this cache."""

    validate_power_of_two(degree)
    order = (0,)
    while len(order) < degree:
        order = tuple(index * 2 for index in order) + tuple(
            index * 2 + 1 for index in order
        )
    return order


def _fft(values: Sequence[Any], mp: Any, *, inverse: bool = False) -> list[Any]:
    """Iterative radix-2 DFT preserving the former recursive butterflies.

    Bit-reversal puts the leaves in the same order as the recursive even/odd
    traversal.  Each stage performs the same multiplication, addition and
    subtraction with the same rounded roots; only independent butterflies
    are rescheduled.  In particular, multiplication by the first (unit) root
    is not skipped.  This removes recursive slices, temporary result arrays
    and repeated public-root cache lookups without changing the precision.
    """

    n = len(values)
    result = [mp.mpc(values[index]) for index in _fft_input_order(n)]
    width = 2
    while width <= n:
        half = width // 2
        powers = _transform_powers(width, mp.mp.prec, inverse, False)
        for start in range(0, n, width):
            for index, power in enumerate(powers):
                left = start + index
                right = left + half
                even = result[left]
                term = power * result[right]
                result[left] = even + term
                result[right] = even - term
        width *= 2
    return result


def _negacyclic_fft(a: Sequence[int], mp: Any) -> list[Any]:
    """Evaluate at all roots of ``x**n + 1`` in a consistent order."""

    n = len(a)
    powers = _transform_powers(n, mp.mp.prec, False, True)
    twisted: list[Any] = []
    for coefficient, power in zip(a, powers, strict=True):
        twisted.append(mp.mpf(int(coefficient)) * power)
    return _fft(twisted, mp)


def _negacyclic_ifft(values: Sequence[Any], mp: Any) -> list[Any]:
    n = len(values)
    transformed = _fft(values, mp, inverse=True)
    powers = _transform_powers(n, mp.mp.prec, True, True)
    result: list[Any] = []
    for value, power in zip(transformed, powers, strict=True):
        result.append((value / n) * power)
    return result


def _round_ties_even(value: Any, mp: Any) -> int:
    """Round an ``mpf`` to the nearest integer, resolving exact ties to even."""

    lower = int(mp.floor(value))
    fraction = value - lower
    half = mp.mpf("0.5")
    if fraction < half:
        return lower
    if fraction > half:
        return lower + 1
    return lower if lower % 2 == 0 else lower + 1


def _babai_basis_spectrum(
    f: Sequence[int], g: Sequence[int], mp: Any,
) -> tuple[tuple[Any, Any, Any], ...]:
    """Compute conjugates and denominators, preserving the former arithmetic."""

    f_fft = _negacyclic_fft(f, mp)
    g_fft = _negacyclic_fft(g, mp)
    result: list[tuple[Any, Any, Any]] = []
    for fi, gi in zip(f_fft, g_fft, strict=True):
        conjugate_f, conjugate_g = mp.conj(fi), mp.conj(gi)
        denominator = fi * conjugate_f + gi * conjugate_g
        if abs(denominator) == 0:
            raise NTRUReductionError("zero denominator in Babai quotient")
        # Keep division in the caller; multiplying by a cached reciprocal
        # would change the rounded arithmetic.
        result.append((conjugate_f, conjugate_g, denominator))
    return tuple(result)


class _BabaiBasisCache:
    """Bounded, private workspace owned by ONE invocation of babai_reduce.

    Exact integer inputs distinguish staged and full-input variants; actual
    bit precision distinguishes arithmetic contexts.  Never reuse rounded
    spectra across precisions, inputs, recursive nodes, or generated keys.
    This object must not enter a module-level cache or a serialized key.
    """

    def __init__(self, max_entries: int = 16) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self.max_entries = max_entries
        self._entries: OrderedDict[
            tuple[int, tuple[int, ...], tuple[int, ...]],
            tuple[tuple[Any, Any, Any], ...],
        ] = OrderedDict()
        self.certified_workspace = CertifiedBasisWorkspace()

    def spectrum(
        self, f: Sequence[int], g: Sequence[int], mp: Any,
    ) -> tuple[tuple[Any, Any, Any], ...]:
        key = (mp.mp.prec, tuple(int(v) for v in f), tuple(int(v) for v in g))
        if key in self._entries:
            self._entries.move_to_end(key)
            return self._entries[key]
        result = _babai_basis_spectrum(f, g, mp)
        self._entries[key] = result
        if len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
        return result

    def clear(self) -> None:
        self._entries.clear()
        self.certified_workspace.clear()


def _rounded_babai_quotient(
    f: Sequence[int],
    g: Sequence[int],
    capital_f: Sequence[int],
    capital_g: Sequence[int],
    dps: int,
    *,
    basis_cache: _BabaiBasisCache | None = None,
) -> tuple[list[int], Any, Any]:
    """Compute one rounded quotient and numerical-separation diagnostics."""

    mp = _require_mpmath()
    with mp.workdps(dps):
        basis = (
            _babai_basis_spectrum(f, g, mp)
            if basis_cache is None else basis_cache.spectrum(f, g, mp)
        )
        capital_f_fft = _negacyclic_fft(capital_f, mp)
        capital_g_fft = _negacyclic_fft(capital_g, mp)

        quotient_fft: list[Any] = []
        for (conjugate_f, conjugate_g, denominator), big_fi, big_gi in zip(
            basis, capital_f_fft, capital_g_fft, strict=True
        ):
            numerator = big_fi * conjugate_f + big_gi * conjugate_g
            quotient_fft.append(numerator / denominator)

        coefficients = _negacyclic_ifft(quotient_fft, mp)
        rounded: list[int] = []
        min_half_distance = mp.inf
        max_imaginary = mp.mpf(0)
        for coefficient in coefficients:
            real = mp.re(coefficient)
            imaginary = abs(mp.im(coefficient))
            max_imaginary = max(max_imaginary, imaginary)
            lower = mp.floor(real)
            half_distance = abs((real - lower) - mp.mpf("0.5"))
            min_half_distance = min(min_half_distance, half_distance)
            rounded.append(_round_ties_even(real, mp))
        # Copy mp values out of the context at the current precision.
        return rounded, +min_half_distance, +max_imaginary


def _exact_ring_quotient(
    numerator: Sequence[int], denominator: Sequence[int]
) -> list[Fraction]:
    """Solve ``denominator * x == numerator`` exactly over the cyclotomic Q-ring.

    A dense Fraction Gaussian elimination would be cubic and unusable at the
    formal degree 1024.  Instead, recursively construct an integer
    quasi-inverse ``u`` and a scalar resultant ``d`` such that::

        denominator * u = d  (mod x**n + 1).

    Then ``x = numerator*u/d`` coefficient-wise.  This is the same field-norm
    tower identity used by NTRUSolve, costs only exact negacyclic products, and
    gives an authoritative tie decision at every supported degree.
    """

    n = len(numerator)
    if n == 0 or len(denominator) != n:
        raise ValueError("ring-degree mismatch in exact quotient")
    validate_power_of_two(n)
    quasi_inverse, resultant = _ring_quasi_inverse(denominator)
    product = negacyclic_mul(numerator, quasi_inverse)
    return [Fraction(coefficient, resultant) for coefficient in product]


def _ring_quasi_inverse(denominator: Sequence[int]) -> tuple[Polynomial, int]:
    """Return integral ``(u,d)`` with ``denominator*u == d`` in the ring."""

    n = len(denominator)
    validate_power_of_two(n)
    integer_denominator = [int(value) for value in denominator]
    if n == 1:
        if integer_denominator[0] == 0:
            raise NTRUReductionError("Babai denominator is not invertible over Q")
        return [1], integer_denominator[0]

    norm = field_norm(integer_denominator)
    lower_inverse, resultant = _ring_quasi_inverse(norm)
    quasi_inverse = negacyclic_mul(
        galois_conjugate(integer_denominator), lift(lower_inverse)
    )
    expected = [resultant, *([0] * (n - 1))]
    if negacyclic_mul(integer_denominator, quasi_inverse) != expected:
        raise ArithmeticError("exact ring quasi-inverse failed verification")
    return quasi_inverse, resultant


def _round_fraction_ties_even(value: Fraction) -> int:
    lower = value.numerator // value.denominator
    twice_remainder = 2 * (value.numerator - lower * value.denominator)
    if twice_remainder < value.denominator:
        return lower
    if twice_remainder > value.denominator:
        return lower + 1
    return lower if lower % 2 == 0 else lower + 1


def _exact_babai_multiplier(
    f: Sequence[int],
    g: Sequence[int],
    capital_f: Sequence[int],
    capital_g: Sequence[int],
) -> list[int]:
    denominator = add(
        negacyclic_mul(f, adjoint(f)),
        negacyclic_mul(g, adjoint(g)),
    )
    numerator = add(
        negacyclic_mul(capital_f, adjoint(f)),
        negacyclic_mul(capital_g, adjoint(g)),
    )
    return [_round_fraction_ties_even(value)
            for value in _exact_ring_quotient(numerator, denominator)]


def _stable_babai_multiplier(
    f: Sequence[int],
    g: Sequence[int],
    capital_f: Sequence[int],
    capital_g: Sequence[int],
    config: ReductionConfig,
    *,
    basis_cache: _BabaiBasisCache | None = None,
) -> tuple[list[int], int, int]:
    """Return a stable staged multiplier, precision used, and scalar shift."""

    small_bits = max(1, _coefficient_bits(f, g))
    large_bits = max(1, _coefficient_bits(capital_f, capital_g))
    # Falcon stages its large-integer reduction in whole-byte chunks.  Rounding
    # sizes upward to ``stage_bits`` likewise removes several high bits per
    # pass, instead of needlessly peeling one bit at a time.
    def staged_size(bits: int) -> int:
        rounded = ((bits + config.stage_bits - 1) // config.stage_bits) * config.stage_bits
        return max(config.window_bits, rounded)

    small_shift = staged_size(small_bits) - config.window_bits
    large_shift = staged_size(large_bits) - config.window_bits
    scalar_shift = large_shift - small_shift
    if scalar_shift < 0:
        # The reduction loop should have stopped before this case.
        return [0] * len(f), config.precision_schedule[0], 0

    scaled_f = [int(value) >> small_shift for value in f]
    scaled_g = [int(value) >> small_shift for value in g]
    scaled_capital_f = [int(value) >> large_shift for value in capital_f]
    scaled_capital_g = [int(value) >> large_shift for value in capital_g]

    _FAST_PATH_COUNTS["staged_attempts"] += 1
    certified = certified_babai_multiplier(
        scaled_f, scaled_g, scaled_capital_f, scaled_capital_g,
        workspace=None if basis_cache is None else basis_cache.certified_workspace,
    )
    if certified is not None:
        _FAST_PATH_COUNTS["staged_certified"] += 1
        # -1 denotes an integer-interval certificate, not decimal precision.
        return certified, -1, scalar_shift
    _FAST_PATH_COUNTS["staged_fallbacks"] += 1

    previous: list[int] | None = None
    for dps in config.precision_schedule:
        candidate, half_margin, imaginary = _rounded_babai_quotient(
            scaled_f, scaled_g, scaled_capital_f, scaled_capital_g, dps,
            basis_cache=basis_cache,
        )
        # A numerical sanity check, not an outward-rounded error certificate.
        # Reject apparent agreement near a half integer or with lost real
        # symmetry; matching two precisions alone is not a proof of rounding.
        threshold = _require_mpmath().power(10, -min(config.guard_digits, dps // 3))
        separated = half_margin > threshold and imaginary < threshold
        if previous == candidate and separated:
            return candidate, dps, scalar_shift
        previous = candidate

    candidate = _exact_babai_multiplier(
        scaled_f, scaled_g, scaled_capital_f, scaled_capital_g
    )
    return candidate, 0, scalar_shift


def _stable_full_babai_multiplier(
    f: Sequence[int],
    g: Sequence[int],
    capital_f: Sequence[int],
    capital_g: Sequence[int],
    config: ReductionConfig,
    *,
    basis_cache: _BabaiBasisCache | None = None,
) -> tuple[list[int], int]:
    """Recompute a stalled staged quotient from the unshifted integers.

    Agreement between high-precision runs is a numerical stability check; it
    does not certify rounding or detect error from discarded staging bits.
    This slower path retains every input bit when retrying a stalled update.
    Only its exact rational fallback supplies an exact rounding decision.
    """

    quotient_bits = max(
        0,
        _coefficient_bits(capital_f, capital_g) - _coefficient_bits(f, g),
    )
    # To resolve the fractional part of a number with ``quotient_bits`` integer
    # bits, retain those bits plus a 192-bit guard.  Convert bits to decimal
    # digits conservatively without relying on binary64 logarithms.
    base_dps = max(
        config.precision_schedule[0],
        (quotient_bits + config.window_bits) * 30103 // 100000 + 20,
    )
    schedule = (base_dps, base_dps + 48, base_dps + 112)
    previous: list[int] | None = None
    mp = _require_mpmath()
    for dps in schedule:
        candidate, half_margin, imaginary = _rounded_babai_quotient(
            f, g, capital_f, capital_g, dps, basis_cache=basis_cache,
        )
        threshold = mp.power(10, -min(config.guard_digits, dps // 3))
        if previous == candidate and half_margin > threshold and imaginary < threshold:
            return candidate, dps
        previous = candidate

    return _exact_babai_multiplier(f, g, capital_f, capital_g), 0


def babai_reduce(
    f: Sequence[int],
    g: Sequence[int],
    capital_f: Sequence[int],
    capital_g: Sequence[int],
    *,
    config: ReductionConfig | None = None,
    require_equation_q: int | None = None,
) -> tuple[Polynomial, Polynomial, tuple[ReductionStep, ...]]:
    """Reduce ``(F,G)`` relative to ``(f,g)`` with exact integer updates.

    If ``require_equation_q`` is supplied, the NTRU equation is checked before
    and after every update.  Callers solving a key should always supply it.

    Fixed-basis spectra are retained only during this call.  Clearing the
    workspace on success AND exceptions releases its cached references even
    if a traceback retains the workspace.  Other computation frames may still
    hold input or intermediate values; Python provides no secure erasure.
    """

    basis_cache = _BabaiBasisCache()
    try:
        return _babai_reduce_with_cache(
            f, g, capital_f, capital_g, config=config,
            require_equation_q=require_equation_q, basis_cache=basis_cache,
        )
    finally:
        basis_cache.clear()


def _babai_reduce_with_cache(
    f: Sequence[int],
    g: Sequence[int],
    capital_f: Sequence[int],
    capital_g: Sequence[int],
    *,
    config: ReductionConfig | None,
    require_equation_q: int | None,
    basis_cache: _BabaiBasisCache,
) -> tuple[Polynomial, Polynomial, tuple[ReductionStep, ...]]:
    """Original reduction and fallback policy with a call-local workspace."""

    if not f or len(f) != len(g) or len(f) != len(capital_f) or len(f) != len(capital_g):
        raise ValueError("all four NTRU polynomials must have equal nonzero degree")
    validate_power_of_two(len(f))
    settings = config or ReductionConfig()
    reduced_f = [int(value) for value in capital_f]
    reduced_g = [int(value) for value in capital_g]
    base_f = [int(value) for value in f]
    base_g = [int(value) for value in g]

    if require_equation_q is not None and not verify_ntru(
        base_f, base_g, reduced_f, reduced_g, require_equation_q
    ):
        raise NTRUError("input to Babai reduction does not satisfy the NTRU equation")

    trace: list[ReductionStep] = []
    base_bits = max(1, _coefficient_bits(base_f, base_g))
    for iteration in range(settings.max_iterations):
        before = _reduction_measure(reduced_f, reduced_g)
        if before[0] < base_bits:
            return reduced_f, reduced_g, tuple(trace)

        multiplier, dps, scalar_shift = _stable_babai_multiplier(
            base_f, base_g, reduced_f, reduced_g, settings, basis_cache=basis_cache,
        )
        if not any(multiplier):
            # A zero quotient is authoritative only if staging retained every
            # input bit.  In particular, right-shifting both operands can turn
            # a small but very high-significance quotient into zero.  Stopping
            # there lets huge recursive resultants leak into all upper levels.
            staged_input_was_truncated = max(
                _coefficient_bits(base_f, base_g),
                _coefficient_bits(reduced_f, reduced_g),
            ) > settings.window_bits
            if not staged_input_was_truncated:
                return reduced_f, reduced_g, tuple(trace)
            multiplier, dps = _stable_full_babai_multiplier(
                base_f, base_g, reduced_f, reduced_g, settings, basis_cache=basis_cache,
            )
            scalar_shift = 0
            if not any(multiplier):
                return reduced_f, reduced_g, tuple(trace)
        if scalar_shift:
            multiplier = [value << scalar_shift for value in multiplier]

        next_f = sub(reduced_f, negacyclic_mul(multiplier, base_f))
        next_g = sub(reduced_g, negacyclic_mul(multiplier, base_g))
        after = _reduction_measure(next_f, next_g)

        if require_equation_q is not None and not verify_ntru(
            base_f, base_g, next_f, next_g, require_equation_q
        ):
            raise ArithmeticError("Babai update broke the exact NTRU invariant")
        if after >= before:
            # Staging throws away low input bits.  Numerical stability at two
            # precisions does not certify that approximation, so retry once
            # from the full integers before declaring a genuine stall.
            full_multiplier, full_dps = _stable_full_babai_multiplier(
                base_f, base_g, reduced_f, reduced_g, settings, basis_cache=basis_cache,
            )
            if not any(full_multiplier):
                return reduced_f, reduced_g, tuple(trace)
            next_f = sub(reduced_f, negacyclic_mul(full_multiplier, base_f))
            next_g = sub(reduced_g, negacyclic_mul(full_multiplier, base_g))
            after = _reduction_measure(next_f, next_g)
            multiplier = full_multiplier
            dps = full_dps
            scalar_shift = 0
            if require_equation_q is not None and not verify_ntru(
                base_f, base_g, next_f, next_g, require_equation_q
            ):
                raise ArithmeticError("full-input Babai update broke the NTRU invariant")
            if after >= before:
                raise NTRUReductionStalledError(
                    "full-input Babai step did not reduce the exact measure "
                    f"(coefficient bits {before[0]} -> {after[0]}, "
                    f"squared-norm bits {before[1].bit_length()} -> "
                    f"{after[1].bit_length()})"
                )

        trace.append(
            ReductionStep(
                iteration=iteration,
                precision_dps=dps,
                scalar_shift=scalar_shift,
                multiplier_nonzero=sum(value != 0 for value in multiplier),
                multiplier_max_bits=_coefficient_bits(multiplier),
                before_measure=(before[0], before[1].bit_length()),
                after_measure=(after[0], after[1].bit_length()),
            )
        )
        reduced_f, reduced_g = next_f, next_g

    raise NTRUReductionError(
        f"Babai reduction exceeded {settings.max_iterations} iterations"
    )


def _solve_recursive(
    f: Polynomial,
    g: Polynomial,
    q: int,
    config: ReductionConfig,
) -> NTRUSolution:
    n = len(f)
    if n == 1:
        gcd, coefficient_f, coefficient_g = integer_xgcd(f[0], g[0])
        # Falcon's construction requires coprime terminal resultants.  Merely
        # being invertible modulo q is not enough to guarantee this condition.
        if gcd != 1:
            raise NTRUNoSolutionError(
                f"terminal resultants are not coprime (gcd={gcd})"
            )
        capital_f = [-q * coefficient_g]
        capital_g = [q * coefficient_f]
        if not verify_ntru(f, g, capital_f, capital_g, q):
            raise ArithmeticError("base-case Bezout signs are inconsistent")
        return NTRUSolution(capital_f, capital_g)

    norm_f = field_norm(f)
    norm_g = field_norm(g)
    lower = _solve_recursive(norm_f, norm_g, q, config)

    capital_f = negacyclic_mul(lift(lower.capital_f), galois_conjugate(g))
    capital_g = negacyclic_mul(lift(lower.capital_g), galois_conjugate(f))
    if not verify_ntru(f, g, capital_f, capital_g, q):
        raise ArithmeticError("recursive NTRU lift failed exact verification")

    capital_f, capital_g, local_steps = babai_reduce(
        f,
        g,
        capital_f,
        capital_g,
        config=config,
        require_equation_q=q,
    )
    if not verify_ntru(f, g, capital_f, capital_g, q):
        raise ArithmeticError("reduced NTRU solution failed exact verification")
    return NTRUSolution(
        capital_f,
        capital_g,
        lower.reduction_steps + local_steps,
    )


def solve_ntru(
    f: Sequence[int],
    g: Sequence[int],
    q: int,
    *,
    reduction_config: ReductionConfig | None = None,
) -> NTRUSolution:
    """Solve ``f*G - g*F = q`` by recursive field norms and lifting.

    ``f`` and ``g`` are never modified.  Failure of the terminal resultant
    coprimality condition is reported as :class:`NTRUNoSolutionError`, allowing
    KeyGen to account for it as a distinct rejection reason.
    """

    if not isinstance(q, int) or isinstance(q, bool) or q <= 1:
        raise ValueError("q must be an integer greater than one")
    if not f or len(f) != len(g):
        raise ValueError("f and g must have equal nonzero degree")
    validate_power_of_two(len(f))
    copied_f = [int(value) for value in f]
    copied_g = [int(value) for value in g]
    solution = _solve_recursive(
        copied_f, copied_g, q, reduction_config or ReductionConfig()
    )
    if not verify_ntru(copied_f, copied_g, solution.capital_f, solution.capital_g, q):
        raise ArithmeticError("NTRUSolve returned an invalid exact solution")
    return solution


def recover_capital_g(
    f: Sequence[int],
    g: Sequence[int],
    capital_f: Sequence[int],
    q: int,
) -> Polynomial:
    """Recover ``G`` exactly from a future compact ``(f,g,F)`` secret key.

    Unlike Falcon's centered lift modulo ``q``, exact rational ring division
    does not silently assume ``|G_i| < q/2``.  This matters because Falcon++ has
    deliberately postponed coefficient-width restrictions.  The result is
    accepted only when every coefficient is integral and the exact NTRU
    equation verifies.
    """

    if not f or len(f) != len(g) or len(f) != len(capital_f):
        raise ValueError("f, g, and F must have equal nonzero degree")
    target = add(
        [int(q), *([0] * (len(f) - 1))],
        negacyclic_mul(g, capital_f),
    )
    quotient = _exact_ring_quotient(target, f)
    if any(value.denominator != 1 for value in quotient):
        raise NTRUError("(f,g,F) does not yield an integral G")
    capital_g = [value.numerator for value in quotient]
    if not verify_ntru(f, g, capital_f, capital_g, q):
        raise NTRUError("recovered G does not satisfy the exact NTRU equation")
    return capital_g


def gram_schmidt_norm_squared(
    f: Sequence[int],
    g: Sequence[int],
    q: int,
    *,
    dps: int = 100,
) -> Any:
    """Return Falcon's squared NTRU-basis Gram--Schmidt quality measure.

    The first branch is ``||f||^2 + ||g||^2``.  The second is the squared norm
    of ``q*(g*/D, f*/D)``, where ``D=f*f*+g*g*``.  It depends only on ``f,g``
    and can therefore be used before the expensive recursive solver.
    """

    if not f or len(f) != len(g):
        raise ValueError("f and g must have equal nonzero degree")
    validate_power_of_two(len(f))
    if dps < 30:
        raise ValueError("dps must be at least 30")
    mp = _require_mpmath()
    with mp.workdps(dps):
        f_fft = _negacyclic_fft(f, mp)
        g_fft = _negacyclic_fft(g, mp)
        dual_f_fft: list[Any] = []
        dual_g_fft: list[Any] = []
        for fi, gi in zip(f_fft, g_fft, strict=True):
            denominator = fi * mp.conj(fi) + gi * mp.conj(gi)
            if abs(denominator) == 0:
                return mp.inf
            dual_f_fft.append(q * mp.conj(gi) / denominator)
            dual_g_fft.append(q * mp.conj(fi) / denominator)
        dual_f = _negacyclic_ifft(dual_f_fft, mp)
        dual_g = _negacyclic_ifft(dual_g_fft, mp)
        dual_norm = mp.fsum(
            mp.re(value) ** 2 + mp.im(value) ** 2
            for value in (*dual_f, *dual_g)
        )
        primal_norm = mp.mpf(_squared_norm(f, g))
        return +max(primal_norm, dual_norm)


def passes_gram_schmidt_bound(
    f: Sequence[int],
    g: Sequence[int],
    q: int,
    gamma: Any,
    *,
    dps: int = 100,
) -> bool:
    """Certify the GS inequality when possible; otherwise use the old path.

    The fast threshold is the exact rational value of gamma's decimal text.
    The enclosing calculation uses Parseval and integer interval arithmetic.
    Invalid/general legacy inputs and inconclusive intervals retain the former
    validation and mpmath calculation below.
    """

    supported = (
        isinstance(q, int) and not isinstance(q, bool) and q > 0
        and isinstance(dps, int) and not isinstance(dps, bool) and dps >= 30
        and bool(f) and len(f) == len(g)
        and not (len(f) & (len(f) - 1))
        and all(isinstance(value, int) for p in (f, g) for value in p)
    )
    if supported:
        try:
            exact_bound = Fraction(str(gamma)) ** 2 * q
        except (ValueError, ZeroDivisionError):
            exact_bound = None
        if exact_bound is not None:
            _FAST_PATH_COUNTS["gso_attempts"] += 1
            enclosure = certified_gso_squared_bounds(f, g, q)
            if enclosure is not None:
                low, high, denominator = enclosure
                threshold = exact_bound.numerator * denominator
                if high * exact_bound.denominator <= threshold:
                    _FAST_PATH_COUNTS["gso_certified"] += 1
                    return True
                if low * exact_bound.denominator > threshold:
                    _FAST_PATH_COUNTS["gso_certified"] += 1
                    return False
            _FAST_PATH_COUNTS["gso_fallbacks"] += 1

    mp = _require_mpmath()
    with mp.workdps(dps):
        gamma_mp = mp.mpf(str(gamma))
        bound = gamma_mp * gamma_mp * q
        return gram_schmidt_norm_squared(f, g, q, dps=dps) <= bound


# Familiar aliases for readers comparing this module with Falcon's notation.
ntru_solve = solve_ntru
complete_private = recover_capital_g


__all__ = [
    "NTRUError",
    "NTRUNoSolutionError",
    "NTRUReductionError",
    "NTRUReductionPrecisionError",
    "NTRUReductionStalledError",
    "NTRUSolution",
    "ReductionConfig",
    "ReductionStep",
    "babai_reduce",
    "complete_private",
    "gram_schmidt_norm_squared",
    "fast_path_diagnostics",
    "integer_xgcd",
    "ntru_solve",
    "passes_gram_schmidt_bound",
    "recover_capital_g",
    "solve_ntru",
]
