"""Readable Fourier arithmetic for the negacyclic Falcon ring.

The ring used throughout Falcon++ is

``R = R[x] / (x**n + 1)``, with ``n`` a power of two.

The transform below evaluates a polynomial at all ``n`` odd ``2n``-th
roots of unity.  Values are kept in the recursive order used by Falcon's
``split_fft``/``merge_fft`` algorithms: roots that differ only by a sign are
adjacent.  Keeping the full conjugate spectrum is intentionally less compact
than the optimized C representation, but makes the reference implementation
and its invariants much easier to inspect.

Two numerical paths are exposed:

* :func:`fft` and :func:`ifft` use Python binary64 complex arithmetic and are
  the ordinary, fast reference path (``O(n log n)``);
* the ``*_high_precision`` functions use :mod:`mpmath`, also in
  ``O(n log n)``, for checks and the high-precision signing path.

This module is an independent implementation of the algorithms described in
the Falcon specification.  It is not constant time.
"""

from __future__ import annotations

from functools import lru_cache
import cmath
import math
from numbers import Number
from typing import Any, Sequence, TypeVar


Scalar = TypeVar("Scalar")
MAX_DEGREE = 1024


def _check_degree(n: int, *, allow_one: bool = True) -> None:
    """Validate a supported power-of-two transform degree."""

    minimum = 1 if allow_one else 2
    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError("the transform degree must be an integer")
    if n < minimum or n > MAX_DEGREE or n & (n - 1):
        qualifier = "a power of two" if allow_one else "an even power of two"
        raise ValueError(
            f"the transform degree must be {qualifier} in "
            f"[{minimum}, {MAX_DEGREE}], got {n}"
        )


def _same_length(left: Sequence[Any], right: Sequence[Any]) -> int:
    n = len(left)
    if n != len(right):
        raise ValueError(f"length mismatch: {n} != {len(right)}")
    _check_degree(n)
    return n


@lru_cache(maxsize=None)
def root_exponents(n: int) -> tuple[int, ...]:
    """Return Falcon-order exponents for the roots of ``x**n + 1``.

    An exponent ``e`` denotes ``exp(pi*i*e/n)``.  The result is useful in
    tests and in the high-precision implementation because it avoids deriving
    high-precision roots from binary64 constants.
    """

    _check_degree(n)
    if n == 1:
        return (1,)

    half = n // 2
    result: list[int] = []
    for exponent in root_exponents(half):
        # Choose the square root whose argument lies in (-pi/2, pi/2], then
        # place its negation immediately after it.  This is the root ordering
        # for which Falcon's split/merge equations hold verbatim.
        signed = exponent if exponent <= half else exponent - 2 * half
        first = signed % (2 * n)
        result.extend((first, (first + n) % (2 * n)))
    return tuple(result)


@lru_cache(maxsize=None)
def cyclotomic_roots(n: int) -> tuple[complex, ...]:
    """Return all roots of ``x**n + 1`` in Falcon recursive order."""

    return tuple(
        cmath.rect(1.0, math.pi * exponent / n)
        for exponent in root_exponents(n)
    )


def split_coefficients(values: Sequence[Scalar]) -> tuple[list[Scalar], list[Scalar]]:
    """Split ``f(x)`` as ``f0(x**2) + x*f1(x**2)``."""

    n = len(values)
    _check_degree(n, allow_one=False)
    return list(values[0::2]), list(values[1::2])


def merge_coefficients(parts: Sequence[Sequence[Scalar]]) -> list[Scalar]:
    """Inverse of :func:`split_coefficients`."""

    if len(parts) != 2:
        raise ValueError("merge_coefficients expects exactly two parts")
    even, odd = parts
    if len(even) != len(odd):
        raise ValueError("the two coefficient parts must have equal length")
    _check_degree(2 * len(even), allow_one=False)
    result: list[Scalar] = []
    for a, b in zip(even, odd, strict=True):
        result.extend((a, b))
    return result


# Familiar short names are useful in NTRU code.
split = split_coefficients
merge = merge_coefficients


def merge_fft(parts: Sequence[Sequence[complex]]) -> list[complex]:
    """Merge the transforms of the even and odd coefficient polynomials.

    This is Algorithm ``mergefft_2`` in Falcon terminology.
    """

    if len(parts) != 2:
        raise ValueError("merge_fft expects exactly two parts")
    even, odd = parts
    if len(even) != len(odd):
        raise ValueError("the two FFT parts must have equal length")
    n = 2 * len(even)
    _check_degree(n, allow_one=False)
    roots = cyclotomic_roots(n)
    result: list[complex] = [0j] * n
    for i, (a, b) in enumerate(zip(even, odd, strict=True)):
        product = roots[2 * i] * b
        result[2 * i] = a + product
        result[2 * i + 1] = a - product
    return result


def split_fft(values: Sequence[complex]) -> tuple[list[complex], list[complex]]:
    """Split a Falcon-order FFT vector into its even/odd transforms.

    This is Algorithm ``splitfft_2`` in Falcon terminology and is the exact
    inverse of :func:`merge_fft` up to floating-point rounding.
    """

    n = len(values)
    _check_degree(n, allow_one=False)
    roots = cyclotomic_roots(n)
    even: list[complex] = [0j] * (n // 2)
    odd: list[complex] = [0j] * (n // 2)
    for i in range(n // 2):
        positive = values[2 * i]
        negative = values[2 * i + 1]
        even[i] = 0.5 * (positive + negative)
        # roots have unit magnitude, hence division by w equals multiplication
        # by conjugate(w).  Writing it this way mirrors the Falcon algorithm.
        odd[i] = 0.5 * (positive - negative) * roots[2 * i].conjugate()
    return even, odd


def fft(coefficients: Sequence[Number]) -> list[complex]:
    """Evaluate a polynomial at the roots of ``x**n + 1``.

    ``n`` must be a power of two no larger than 1024.  The returned full
    spectrum has length ``n``.
    """

    n = len(coefficients)
    _check_degree(n)
    if n == 1:
        return [complex(coefficients[0])]
    even, odd = split_coefficients(coefficients)
    return merge_fft((fft(even), fft(odd)))


def _ifft_complex(values: Sequence[complex]) -> list[complex]:
    n = len(values)
    if n == 1:
        return [complex(values[0])]
    even_fft, odd_fft = split_fft(values)
    return merge_coefficients((_ifft_complex(even_fft), _ifft_complex(odd_fft)))


def _real_if_near(value: complex, tolerance: float) -> complex | float:
    scale = max(1.0, abs(value.real))
    if abs(value.imag) <= tolerance * scale:
        return float(value.real)
    return value


def ifft(
    values: Sequence[complex],
    *,
    real_if_close: bool = True,
    tolerance: float = 2.0e-12,
) -> list[complex | float]:
    """Invert :func:`fft`.

    Real ring polynomials have a conjugate-symmetric spectrum.  For those
    inputs the default converts harmless residual imaginary parts to floats.
    Set ``real_if_close=False`` to retain complex values unconditionally.
    """

    n = len(values)
    _check_degree(n)
    if tolerance < 0 or not math.isfinite(tolerance):
        raise ValueError("tolerance must be a finite non-negative number")
    result = _ifft_complex(values)
    if not real_if_close:
        return result
    return [_real_if_near(value, tolerance) for value in result]


def add(left: Sequence[Scalar], right: Sequence[Scalar]) -> list[Scalar]:
    """Coefficient-wise addition (also valid for FFT vectors)."""

    _same_length(left, right)
    return [a + b for a, b in zip(left, right, strict=True)]  # type: ignore[operator]


def neg(values: Sequence[Scalar]) -> list[Scalar]:
    """Coefficient-wise negation."""

    _check_degree(len(values))
    return [-value for value in values]  # type: ignore[operator]


def sub(left: Sequence[Scalar], right: Sequence[Scalar]) -> list[Scalar]:
    """Coefficient-wise subtraction (also valid for FFT vectors)."""

    _same_length(left, right)
    return [a - b for a, b in zip(left, right, strict=True)]  # type: ignore[operator]


def scale(values: Sequence[Scalar], factor: Any) -> list[Any]:
    """Multiply every coefficient/value by ``factor``."""

    _check_degree(len(values))
    return [factor * value for value in values]


def add_fft(left: Sequence[Scalar], right: Sequence[Scalar]) -> list[Scalar]:
    return add(left, right)


def sub_fft(left: Sequence[Scalar], right: Sequence[Scalar]) -> list[Scalar]:
    return sub(left, right)


def neg_fft(values: Sequence[Scalar]) -> list[Scalar]:
    return neg(values)


def mul_fft(left: Sequence[Scalar], right: Sequence[Scalar]) -> list[Any]:
    """Pointwise product of two FFT vectors."""

    _same_length(left, right)
    return [a * b for a, b in zip(left, right, strict=True)]


def div_fft(left: Sequence[Scalar], right: Sequence[Scalar]) -> list[Any]:
    """Pointwise quotient of two FFT vectors."""

    _same_length(left, right)
    if any(value == 0 for value in right):
        raise ZeroDivisionError("division by a zero Fourier component")
    return [a / b for a, b in zip(left, right, strict=True)]


def adj_fft(values: Sequence[Scalar]) -> list[Any]:
    """Apply the ring adjoint in Fourier representation."""

    _check_degree(len(values))
    return [value.conjugate() for value in values]  # type: ignore[attr-defined]


def adj(coefficients: Sequence[Scalar]) -> list[Any]:
    """Return ``f(x^-1)`` modulo ``x**n + 1`` for real coefficients."""

    n = len(coefficients)
    _check_degree(n)
    if n == 1:
        return [coefficients[0]]
    return [coefficients[0], *(-coefficients[n - i] for i in range(1, n))]


def mul(left: Sequence[Number], right: Sequence[Number]) -> list[complex | float]:
    """Negacyclic polynomial multiplication through the FFT."""

    _same_length(left, right)
    return ifft(mul_fft(fft(left), fft(right)))


def div(left: Sequence[Number], right: Sequence[Number]) -> list[complex | float]:
    """Polynomial division in ``R[x]/(x**n+1)`` when ``right`` is invertible."""

    _same_length(left, right)
    return ifft(div_fft(fft(left), fft(right)))


def negacyclic_convolution(left: Sequence[Scalar], right: Sequence[Scalar]) -> list[Any]:
    """Dependency-free exact ``O(n**2)`` negacyclic convolution.

    This function is deliberately separate from :func:`mul`: it is useful as
    a test oracle for integer polynomials and for small exact computations.
    """

    n = _same_length(left, right)
    result: list[Any] = [0 for _ in range(n)]
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            index = i + j
            product = a * b  # type: ignore[operator]
            if index < n:
                result[index] += product
            else:
                result[index - n] -= product
    return result


def _require_mpmath() -> Any:
    try:
        import mpmath  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised without extras
        raise ImportError(
            "high-precision FFT operations require the optional 'mpmath' "
            "dependency"
        ) from exc
    return mpmath


def _mp_roots(n: int, mp: Any) -> tuple[Any, ...]:
    return tuple(
        mp.exp(mp.j * mp.pi * exponent / n) for exponent in root_exponents(n)
    )


def _mp_merge_fft(parts: Sequence[Sequence[Any]], mp: Any) -> list[Any]:
    if len(parts) != 2:
        raise ValueError("merge_fft_high_precision expects exactly two parts")
    even, odd = parts
    if len(even) != len(odd):
        raise ValueError("the two FFT parts must have equal length")
    n = 2 * len(even)
    _check_degree(n, allow_one=False)
    roots = _mp_roots(n, mp)
    result = [mp.mpc(0) for _ in range(n)]
    for i, (a, b) in enumerate(zip(even, odd, strict=True)):
        product = roots[2 * i] * b
        result[2 * i] = a + product
        result[2 * i + 1] = a - product
    return result


def _mp_split_fft(values: Sequence[Any], mp: Any) -> tuple[list[Any], list[Any]]:
    n = len(values)
    _check_degree(n, allow_one=False)
    roots = _mp_roots(n, mp)
    half = mp.mpf("0.5")
    even = [mp.mpc(0) for _ in range(n // 2)]
    odd = [mp.mpc(0) for _ in range(n // 2)]
    for i in range(n // 2):
        positive = values[2 * i]
        negative = values[2 * i + 1]
        even[i] = half * (positive + negative)
        odd[i] = half * (positive - negative) / roots[2 * i]
    return even, odd


def fft_high_precision(
    coefficients: Sequence[Any], *, dps: int = 100
) -> list[Any]:
    """High-precision ``O(n log n)`` negacyclic FFT using :mod:`mpmath`."""

    n = len(coefficients)
    _check_degree(n)
    if dps < 20:
        raise ValueError("dps must be at least 20")
    mpmath = _require_mpmath()
    with mpmath.workdps(dps):
        values = [mpmath.mpc(value) for value in coefficients]

        def recurse(items: Sequence[Any]) -> list[Any]:
            if len(items) == 1:
                return [mpmath.mpc(items[0])]
            even, odd = split_coefficients(items)
            return _mp_merge_fft((recurse(even), recurse(odd)), mpmath)

        # Unary plus rounds/copies the value into the current context before
        # leaving workdps, so callers retain approximately ``dps`` digits.
        return [+value for value in recurse(values)]


def split_fft_high_precision(
    values: Sequence[Any], *, dps: int = 100
) -> tuple[list[Any], list[Any]]:
    """High-precision counterpart of :func:`split_fft`."""

    if dps < 20:
        raise ValueError("dps must be at least 20")
    mpmath = _require_mpmath()
    with mpmath.workdps(dps):
        converted = [mpmath.mpc(value) for value in values]
        even, odd = _mp_split_fft(converted, mpmath)
        return ([+value for value in even], [+value for value in odd])


def merge_fft_high_precision(
    parts: Sequence[Sequence[Any]], *, dps: int = 100
) -> list[Any]:
    """High-precision counterpart of :func:`merge_fft`."""

    if dps < 20:
        raise ValueError("dps must be at least 20")
    mpmath = _require_mpmath()
    with mpmath.workdps(dps):
        converted = [
            [mpmath.mpc(value) for value in part]
            for part in parts
        ]
        return [+value for value in _mp_merge_fft(converted, mpmath)]


def ifft_high_precision(
    values: Sequence[Any],
    *,
    dps: int = 100,
    real_if_close: bool = True,
) -> list[Any]:
    """High-precision inverse of :func:`fft_high_precision`."""

    n = len(values)
    _check_degree(n)
    if dps < 20:
        raise ValueError("dps must be at least 20")
    mpmath = _require_mpmath()
    with mpmath.workdps(dps):
        converted = [mpmath.mpc(value) for value in values]

        def recurse(items: Sequence[Any]) -> list[Any]:
            if len(items) == 1:
                return [mpmath.mpc(items[0])]
            even_fft, odd_fft = _mp_split_fft(items, mpmath)
            return merge_coefficients((recurse(even_fft), recurse(odd_fft)))

        result = recurse(converted)
        if real_if_close:
            tolerance = mpmath.power(10, -(dps - 10))
            cleaned: list[Any] = []
            for value in result:
                if abs(mpmath.im(value)) <= tolerance * max(1, abs(mpmath.re(value))):
                    cleaned.append(+mpmath.re(value))
                else:
                    cleaned.append(+value)
            return cleaned
        return [+value for value in result]


def fft_reference(coefficients: Sequence[Any], *, dps: int = 100) -> list[Any]:
    """Direct ``O(n**2)`` high-precision evaluation oracle.

    Prefer :func:`fft_high_precision` for real work.  This simple formula is
    retained as an independent small-degree test oracle.
    """

    n = len(coefficients)
    _check_degree(n)
    if dps < 20:
        raise ValueError("dps must be at least 20")
    mpmath = _require_mpmath()
    with mpmath.workdps(dps):
        coeffs = [mpmath.mpc(value) for value in coefficients]
        result: list[Any] = []
        for exponent in root_exponents(n):
            root = mpmath.exp(mpmath.j * mpmath.pi * exponent / n)
            value = mpmath.mpc(0)
            for coefficient in reversed(coeffs):
                value = value * root + coefficient
            result.append(+value)
        return result


# Full-spectrum transforms have one complex value per ring coefficient.
fft_ratio = 1


__all__ = [
    "MAX_DEGREE",
    "add",
    "add_fft",
    "adj",
    "adj_fft",
    "cyclotomic_roots",
    "div",
    "div_fft",
    "fft",
    "fft_high_precision",
    "fft_reference",
    "fft_ratio",
    "ifft",
    "ifft_high_precision",
    "merge",
    "merge_coefficients",
    "merge_fft",
    "merge_fft_high_precision",
    "mul",
    "mul_fft",
    "neg",
    "neg_fft",
    "negacyclic_convolution",
    "root_exponents",
    "scale",
    "split",
    "split_coefficients",
    "split_fft",
    "split_fft_high_precision",
    "sub",
    "sub_fft",
]
