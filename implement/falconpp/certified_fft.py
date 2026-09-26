"""Pure-Python fixed-point enclosures for fast NTRU numerical decisions.

Only Python integers are used in the certificate.  A complex ball ``(a,b,e)``
at scale ``Q=2**bits`` means the exact real and imaginary components lie in
``[a-e,a+e]/Q`` and ``[b-e,b+e]/Q``.  These are rectangular coordinate bounds,
not Euclidean radii.  Roots are enclosed with integer-square-root half-angle
identities; no assumed accuracy of libm, floating point, or mpmath is needed.

For multiplication the coordinate perturbation in integer units is at most
``((|a|+|b|)*r + (|c|+|d|)*e + 2*e*r)/Q``.  We round this bound upward and
add one unit for flooring the computed centre.  Division by ``(d +/- r)/Q``
requires ``d>r``; its perturbation is bounded by
``Q*(e*d + max(|a|,|b|)*r)/(d*(d-r))``, again rounded up plus one unit.
FFT butterflies use these operations throughout, so cancellation does not
invalidate the absolute error bounds.  Half-integer boundaries are never
guessed: a Babai coefficient is returned only when its WHOLE interval lies
strictly inside one nearest-integer rounding cell.  Ambiguous input returns
``None`` for the caller's original high-precision/exact fallback.

The finite precision/degrees here are fast-path resource budgets, not claims
that every valid input is certifiable at those budgets.  They do not cap the
precision of the caller's fallback.  Public root caches are bounded; private
basis spectra require an explicitly owned, bounded, clearable workspace.
"""

from __future__ import annotations

from collections import OrderedDict
from functools import lru_cache
from math import isqrt
from operator import index
from typing import Sequence


Ball = tuple[int, int, int]
RealBall = tuple[int, int]
Basis = tuple[tuple[Ball, Ball, RealBall], ...]

DEFAULT_FRACTION_BITS = 128
MAX_WORK_BITS = 384
MAX_DEGREE = 1024


def _ceil_div(value: int, divisor: int) -> int:
    return -((-value) // divisor)


def _ceil_sqrt(value: int) -> int:
    result = isqrt(value)
    return result + (result * result != value)


def _mul(left: Ball, right: Ball, bits: int) -> Ball:
    a, b, e = left
    c, d, r = right
    if e == 0 and a == 1 << bits and b == 0:
        return right
    if r == 0 and c == 1 << bits and d == 0:
        return left
    numerator = (abs(a) + abs(b)) * r + (abs(c) + abs(d)) * e + 2 * e * r
    error = ((numerator + (1 << bits) - 1) >> bits) + 1
    return (a * c - b * d) >> bits, (a * d + b * c) >> bits, error


def _abs_squared(value: Ball, bits: int) -> RealBall:
    a, b, e = value
    error_numerator = 2 * (abs(a) + abs(b)) * e + 2 * e * e
    error = ((error_numerator + (1 << bits) - 1) >> bits) + 1
    return (a * a + b * b) >> bits, error


def _divide_real(value: Ball, divisor: RealBall, bits: int) -> Ball | None:
    a, b, e = value
    d, r = divisor
    if d <= r:
        return None
    error = _ceil_div(
        (e * d + max(abs(a), abs(b)) * r) << bits,
        d * (d - r),
    ) + 1
    return (a << bits) // d, (b << bits) // d, error


def _integer_ball(value: int, scale_bits: int, bits: int) -> Ball:
    """Enclose EXACT value/2**scale_bits, including discarded input bits."""
    shift = bits - scale_bits
    if shift >= 0:
        return value << shift, 0, 0
    result = value >> -shift
    return result, 0, int((result << -shift) != value)


def _root(order: int, bits: int) -> Ball:
    """Enclose exp(2*pi*i/order) for power-of-two orders, using only isqrt."""
    q = 1 << bits
    if order == 1:
        return q, 0, 0
    if order == 2:
        return -q, 0, 0
    # Start at pi/2; at every step take the positive half angle.  Radicands
    # below are exact integers because Q is even, including when c is odd.
    cosine_low = cosine_high = 0
    sine_low = sine_high = q
    current_order = 4
    while current_order < order:
        new_cosine_low = isqrt(((q + cosine_low) * q) // 2)
        new_cosine_high = _ceil_sqrt(((q + cosine_high) * q) // 2)
        sine_low = isqrt(((q - cosine_high) * q) // 2)
        sine_high = _ceil_sqrt(((q - cosine_low) * q) // 2)
        cosine_low, cosine_high = new_cosine_low, new_cosine_high
        current_order *= 2
    real = (cosine_low + cosine_high) // 2
    imaginary = (sine_low + sine_high) // 2
    error = max(cosine_high - real, sine_high - imaginary)
    return real, imaginary, error


@lru_cache(maxsize=96)
def _powers(degree: int, bits: int, inverse: bool, twist: bool) -> tuple[Ball, ...]:
    """Bounded cache whose keys and values depend on PUBLIC parameters only."""
    count = degree if twist else degree // 2
    root = _root(2 * degree if twist else degree, bits)
    if inverse:
        root = root[0], -root[1], root[2]
    power = (1 << bits, 0, 0)
    result = []
    for _ in range(count):
        result.append(power)
        power = _mul(power, root, bits)
    return tuple(result)


@lru_cache(maxsize=11)
def _bit_reverse(degree: int) -> tuple[int, ...]:
    order = (0,)
    while len(order) < degree:
        order = tuple(i * 2 for i in order) + tuple(i * 2 + 1 for i in order)
    return order


def _fft(values: Sequence[Ball], bits: int, *, inverse: bool = False) -> list[Ball]:
    degree = len(values)
    result = [values[i] for i in _bit_reverse(degree)]
    width = 2
    while width <= degree:
        half = width // 2
        powers = _powers(width, bits, inverse, False)
        for start in range(0, degree, width):
            for j, power in enumerate(powers):
                left = start + j
                right = left + half
                a, b, e = result[left]
                c, d, r = _mul(result[right], power, bits)
                result[left] = a + c, b + d, e + r
                result[right] = a - c, b - d, e + r
        width *= 2
    return result


def _negacyclic_fft(values: Sequence[int], scale_bits: int, bits: int) -> list[Ball]:
    twisted = [
        _mul(_integer_ball(value, scale_bits, bits), power, bits)
        for value, power in zip(values, _powers(len(values), bits, False, True), strict=True)
    ]
    return _fft(twisted, bits)


def _negacyclic_ifft(values: Sequence[Ball], bits: int) -> list[Ball]:
    degree = len(values)
    transformed = _fft(values, bits, inverse=True)
    return [
        _mul((a // degree, b // degree, _ceil_div(e, degree) + 1), power, bits)
        for (a, b, e), power in zip(
            transformed, _powers(degree, bits, True, True), strict=True
        )
    ]


def _validate_inputs(*polynomials: Sequence[int]) -> tuple[tuple[int, ...], ...]:
    if not polynomials or not polynomials[0]:
        raise ValueError("polynomials must have equal nonzero power-of-two degree")
    degree = len(polynomials[0])
    if degree & (degree - 1) or any(len(p) != degree for p in polynomials):
        raise ValueError("polynomials must have equal nonzero power-of-two degree")
    # operator.index forbids silent loss of bits from float/Decimal inputs.
    return tuple(tuple(index(value) for value in p) for p in polynomials)


def _basis(f: tuple[int, ...], g: tuple[int, ...], scale_bits: int, bits: int) -> Basis | None:
    result = []
    for fi, gi in zip(
        _negacyclic_fft(f, scale_bits, bits), _negacyclic_fft(g, scale_bits, bits), strict=True
    ):
        fd, fe = _abs_squared(fi, bits)
        gd, ge = _abs_squared(gi, bits)
        denominator = fd + gd, fe + ge
        if denominator[0] <= denominator[1]:
            return None
        result.append(((fi[0], -fi[1], fi[2]), (gi[0], -gi[1], gi[2]), denominator))
    return tuple(result)


class CertifiedBasisWorkspace:
    """One-node private workspace.  Caller MUST clear it in a finally block.

    Exact integer f,g, common scaling, and precision all enter the cache key.
    This never belongs in a global cache or a serialized private key.  Clearing
    releases references; Python does not promise secure physical memory erasure.
    """

    def __init__(self, max_entries: int = 8) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self.max_entries = max_entries
        self._entries: OrderedDict[tuple, Basis | None] = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0

    def spectrum(
        self, f: tuple[int, ...], g: tuple[int, ...], scale_bits: int, bits: int
    ) -> Basis | None:
        key = bits, scale_bits, f, g
        if key in self._entries:
            self.cache_hits += 1
            self._entries.move_to_end(key)
            return self._entries[key]
        self.cache_misses += 1
        result = _basis(f, g, scale_bits, bits)
        self._entries[key] = result
        if len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
        return result

    def clear(self) -> None:
        self._entries.clear()


def certified_babai_multiplier(
    f: Sequence[int],
    g: Sequence[int],
    capital_f: Sequence[int],
    capital_g: Sequence[int],
    *,
    workspace: CertifiedBasisWorkspace | None = None,
    fraction_bits: int = DEFAULT_FRACTION_BITS,
) -> list[int] | None:
    """Certify exact nearest-integer Babai quotient, or decline with None.

    All four polynomial inputs are scaled by the SAME exact power of two, so
    their quotient is unchanged.  Retained fixed-point bits grow with the
    integer-size gap, not with an assumed distribution of coefficients.
    This function does not itself truncate the caller's staged inputs.
    """
    f, g, capital_f, capital_g = _validate_inputs(f, g, capital_f, capital_g)
    if fraction_bits < 32:
        raise ValueError("fraction_bits must be at least 32")
    if len(f) > MAX_DEGREE:
        return None
    scale_bits = max(1, *(abs(v).bit_length() for p in (f, g) for v in p))
    large_bits = max(1, *(abs(v).bit_length() for p in (capital_f, capital_g) for v in p))
    # Public 32-bit precision buckets let nearby coefficient sizes share
    # roots and a per-node basis without reusing any lower-precision result.
    requested_bits = fraction_bits + max(0, large_bits - scale_bits)
    bits = ((requested_bits + 31) // 32) * 32
    if bits > MAX_WORK_BITS:
        return None
    basis = (
        _basis(f, g, scale_bits, bits) if workspace is None
        else workspace.spectrum(f, g, scale_bits, bits)
    )
    if basis is None:
        return None
    quotient = []
    for (fi, gi, denominator), big_f, big_g in zip(
        basis, _negacyclic_fft(capital_f, scale_bits, bits),
        _negacyclic_fft(capital_g, scale_bits, bits), strict=True,
    ):
        a, b, e = _mul(big_f, fi, bits)
        c, d, r = _mul(big_g, gi, bits)
        value = _divide_real((a + c, b + d, e + r), denominator, bits)
        if value is None:
            return None
        quotient.append(value)
    rounded = []
    q = 1 << bits
    for a, b, e in _negacyclic_ifft(quotient, bits):
        candidate = (a + q // 2) // q
        if abs(a - candidate * q) + e >= q // 2 or abs(b) > e:
            return None
        rounded.append(candidate)
    return rounded


def certified_gso_squared_bounds(
    f: Sequence[int], g: Sequence[int], q: int, *,
    precision_bits: int = DEFAULT_FRACTION_BITS,
) -> tuple[int, int, int] | None:
    """Return rational (lower numerator, upper numerator, denominator) for GS².

    Parseval gives dual_norm=q²/n * sum(1/(|FFT(f)|²+|FFT(g)|²)).
    The normalized basis divides each input by 2**scale_bits; reciprocals must
    therefore be multiplied by 2**(-2*scale_bits) to recover the original
    dual norm.  The exact primal squared norm is included via max(primal,dual).
    No inverse FFT or floating-point threshold comparison is needed.
    """
    f, g = _validate_inputs(f, g)
    q = index(q)
    if q <= 0:
        raise ValueError("q must be positive")
    if precision_bits < 32:
        raise ValueError("precision_bits must be at least 32")
    if len(f) > MAX_DEGREE or precision_bits > MAX_WORK_BITS:
        return None
    scale_bits = max(1, *(abs(v).bit_length() for p in (f, g) for v in p))
    basis = _basis(f, g, scale_bits, precision_bits)
    if basis is None:
        return None
    numerator_scale = 1 << (2 * precision_bits)
    low = high = 0
    for _, _, (d, e) in basis:
        low += numerator_scale // (d + e)
        high += _ceil_div(numerator_scale, d - e)
    denominator = len(f) << (2 * scale_bits + precision_bits)
    primal_scaled = sum(v * v for p in (f, g) for v in p) * denominator
    return max(primal_scaled, q * q * low), max(primal_scaled, q * q * high), denominator


__all__ = [
    "CertifiedBasisWorkspace", "certified_babai_multiplier",
    "certified_gso_squared_bounds",
]
