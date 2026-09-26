"""Exact polynomial arithmetic used by the Falcon++ reference code.

Ring elements are represented as coefficient lists in ascending order.  A
list of length ``n`` denotes an element of ``Z[x] / (x**n + 1)``.  Functions
whose name contains ``poly`` (for example :func:`poly_xgcd_mod`) operate on
ordinary polynomials instead; their result is trimmed to its canonical degree.

This module uses exact Python integers throughout.  The signed integer-packing
and recursive modular-inverse kernels do not introduce floating-point error;
schoolbook/Karatsuba and polynomial EGCD remain independent reference paths.
"""

from __future__ import annotations

from collections.abc import Sequence


Polynomial = list[int]

# Below this degree, schoolbook multiplication is faster and easier on the
# allocator.  Above it, exact Karatsuba keeps NTRUSolve practical for n=1024.
_KARATSUBA_THRESHOLD = 32
# Very short/sparse inputs avoid packing overhead.  These cutoffs affect only
# speed: both branches compute the identical, unbounded-integer convolution.
_PACKED_CONVOLUTION_MIN_LENGTH = 32
_PACKED_CONVOLUTION_MIN_NONZERO_PAIRS = 256
# All primes used by the eight registered original/additional parameter sets.
# Keep this arithmetic module independent of parameters.py; a registry test
# checks this explicit fast-domain allowlist against the current parameters.
_NORM_INVERSE_PRIME_MODULI = frozenset((509, 953, 1021, 1949))


def validate_power_of_two(n: int) -> None:
    """Raise ``ValueError`` unless ``n`` is a positive power of two."""

    if not isinstance(n, int) or isinstance(n, bool) or n <= 0 or n & (n - 1):
        raise ValueError(f"ring degree must be a positive power of two, got {n!r}")


def _validate_modulus(q: int) -> None:
    if not isinstance(q, int) or isinstance(q, bool) or q <= 1:
        raise ValueError(f"modulus must be an integer greater than one, got {q!r}")


def _same_ring_degree(a: Sequence[int], b: Sequence[int]) -> int:
    if not a or not b:
        raise ValueError("ring polynomials must be non-empty")
    if len(a) != len(b):
        raise ValueError(f"ring-degree mismatch: {len(a)} != {len(b)}")
    return len(a)


def add(a: Sequence[int], b: Sequence[int]) -> Polynomial:
    """Return ``a + b`` in a common, fixed-degree coefficient representation."""

    _same_ring_degree(a, b)
    return [int(ai) + int(bi) for ai, bi in zip(a, b, strict=True)]


def sub(a: Sequence[int], b: Sequence[int]) -> Polynomial:
    """Return ``a - b`` in a common, fixed-degree coefficient representation."""

    _same_ring_degree(a, b)
    return [int(ai) - int(bi) for ai, bi in zip(a, b, strict=True)]


def neg(a: Sequence[int]) -> Polynomial:
    """Return ``-a`` without changing the represented ring degree."""

    if not a:
        raise ValueError("ring polynomial must be non-empty")
    return [-int(ai) for ai in a]


def scalar_mul(a: Sequence[int], scalar: int) -> Polynomial:
    """Multiply every coefficient of ``a`` by an integer scalar."""

    if not a:
        raise ValueError("ring polynomial must be non-empty")
    return [int(scalar) * int(ai) for ai in a]


def poly_mod_xn1(
    coefficients: Sequence[int],
    n: int,
    modulus: int | None = None,
) -> Polynomial:
    """Reduce an ordinary polynomial modulo ``x**n + 1``.

    ``coefficients`` may have arbitrary length.  If ``modulus`` is supplied,
    the returned coefficients are the canonical residues in ``[0, modulus)``.
    """

    if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
        raise ValueError(f"ring degree must be positive, got {n!r}")
    if modulus is not None:
        _validate_modulus(modulus)

    result = [0] * n
    for degree, coefficient in enumerate(coefficients):
        block, index = divmod(degree, n)
        if block & 1:
            result[index] -= int(coefficient)
        else:
            result[index] += int(coefficient)

    if modulus is not None:
        result = [coefficient % modulus for coefficient in result]
    return result


def _schoolbook_convolution(a: Sequence[int], b: Sequence[int]) -> Polynomial:
    """Return the exact, ordinary convolution (internal reference kernel)."""

    if not a or not b:
        return []
    result = [0] * (len(a) + len(b) - 1)
    for i, ai_raw in enumerate(a):
        ai = int(ai_raw)
        if ai == 0:
            continue
        for j, bj_raw in enumerate(b):
            bj = int(bj_raw)
            if bj:
                result[i + j] += ai * bj
    return result


def _karatsuba_equal(a: Sequence[int], b: Sequence[int]) -> Polynomial:
    """Exact convolution for equal, power-of-two padded operand lengths."""

    n = len(a)
    if n <= _KARATSUBA_THRESHOLD:
        return _schoolbook_convolution(a, b)

    half = n // 2
    a_low, a_high = a[:half], a[half:]
    b_low, b_high = b[:half], b[half:]
    low = _karatsuba_equal(a_low, b_low)
    high = _karatsuba_equal(a_high, b_high)
    a_sum = [int(a_low[i]) + int(a_high[i]) for i in range(half)]
    b_sum = [int(b_low[i]) + int(b_high[i]) for i in range(half)]
    middle = _karatsuba_equal(a_sum, b_sum)
    for i in range(len(middle)):
        middle[i] -= low[i] + high[i]

    result = [0] * (2 * n - 1)
    for i, coefficient in enumerate(low):
        result[i] += coefficient
    for i, coefficient in enumerate(middle):
        result[i + half] += coefficient
    for i, coefficient in enumerate(high):
        result[i + n] += coefficient
    return result


def _reference_convolution(a: Sequence[int], b: Sequence[int]) -> Polynomial:
    """Original schoolbook/padded-Karatsuba kernel, retained as an oracle."""

    if not a or not b:
        return []
    target = max(len(a), len(b))
    size = 1 << (target - 1).bit_length()
    if size <= _KARATSUBA_THRESHOLD:
        return _schoolbook_convolution(a, b)
    padded_a = [int(coefficient) for coefficient in a] + [0] * (size - len(a))
    padded_b = [int(coefficient) for coefficient in b] + [0] * (size - len(b))
    # Remove padding-only high coefficients so callers see the ordinary
    # product's canonical maximum length.
    return _karatsuba_equal(padded_a, padded_b)[: len(a) + len(b) - 1]


def _packed_convolution(a: Sequence[int], b: Sequence[int]) -> Polynomial:
    """Exact signed Kronecker substitution with a proven coefficient bound.

    Every output coefficient has absolute value at most
    ``C = min(len(a), len(b)) * max(abs(a)) * max(abs(b))``.  Choose the
    power-of-two base ``B > 2*C``; then balanced base-B digit recovery is
    unique, including for negative packed operands and cross-lane borrows.
    No modular reduction, approximate transform, or coefficient truncation is
    used.  Fixed output length preserves the ordinary-convolution API.
    """

    if not a or not b:
        return []
    integer_a = [int(coefficient) for coefficient in a]
    integer_b = [int(coefficient) for coefficient in b]
    result_length = len(integer_a) + len(integer_b) - 1
    max_a = max(map(abs, integer_a))
    max_b = max(map(abs, integer_b))
    if not max_a or not max_b:
        return [0] * result_length
    bound = min(len(integer_a), len(integer_b)) * max_a * max_b
    # Byte-aligned lanes let int.from_bytes/to_bytes perform the bulk packing
    # in linear time.  Repeated shifts of a growing packed integer would add
    # quadratic Python/bignum copying overhead at the larger ring degrees.
    byte_width = ((2 * bound).bit_length() + 7) // 8
    bits = 8 * byte_width
    base = 1 << bits
    mask = base - 1
    half_base = base >> 1

    def pack(coefficients: Sequence[int]) -> int:
        lanes = []
        borrow = 0
        for coefficient in coefficients:
            digit = coefficient + borrow
            borrow = -int(digit < 0)
            lanes.append((digit & mask).to_bytes(byte_width, "little"))
        # Signed decoding of the top lane accounts for a final negative
        # borrow.  The coefficient bound guarantees that lane's sign bit
        # agrees with the sign of the complete polynomial evaluation.
        return int.from_bytes(b"".join(lanes), "little", signed=True)

    packed_a = pack(integer_a)
    if integer_a == integer_b:
        packed_b = packed_a
    else:
        packed_b = pack(integer_b)
    product = packed_a * packed_b
    encoded = product.to_bytes(result_length * byte_width, "little", signed=True)
    result = []
    carry = 0
    for offset in range(0, len(encoded), byte_width):
        coefficient = int.from_bytes(encoded[offset:offset + byte_width], "little") + carry
        if coefficient >= half_base:
            coefficient -= base
            carry = 1
        else:
            carry = 0
        result.append(coefficient)
    # Negative two's-complement sign extension is cancelled by the final
    # balanced-digit carry; positive extension requires no final carry.
    if carry != int(product < 0):
        raise ArithmeticError("internal signed convolution bound failure")
    return result


def _exact_convolution(a: Sequence[int], b: Sequence[int]) -> Polynomial:
    """Select an exact multiplication kernel without changing coefficients."""

    if not a or not b:
        return []
    if min(len(a), len(b)) < _PACKED_CONVOLUTION_MIN_LENGTH:
        return _reference_convolution(a, b)
    nonzero_a = sum(int(coefficient) != 0 for coefficient in a)
    nonzero_b = sum(int(coefficient) != 0 for coefficient in b)
    if nonzero_a * nonzero_b < _PACKED_CONVOLUTION_MIN_NONZERO_PAIRS:
        return _schoolbook_convolution(a, b)
    return _packed_convolution(a, b)


def negacyclic_mul(
    a: Sequence[int],
    b: Sequence[int],
    modulus: int | None = None,
) -> Polynomial:
    """Multiply in ``Z[x] / (x**n + 1)`` exactly.

    The dispatcher uses schoolbook, Karatsuba, or signed integer packing.
    Modular reduction, when requested, happens after the exact convolution
    and therefore cannot change the mathematical result.
    """

    n = _same_ring_degree(a, b)
    if modulus is not None:
        _validate_modulus(modulus)

    convolution = _exact_convolution(a, b)
    result = convolution[:n]
    # The product of two degree-(n-1) polynomials has degree at most 2n-2,
    # hence there is exactly one possible negacyclic wrap.
    for degree in range(n, len(convolution)):
        result[degree - n] -= convolution[degree]

    if modulus is not None:
        result = [coefficient % modulus for coefficient in result]
    return result


def adjoint(a: Sequence[int]) -> Polynomial:
    """Return the canonical involution ``a*(x) = a(x**-1)``.

    In the negacyclic ring, ``x**-i = -x**(n-i)`` for ``0 < i < n``.
    """

    if not a:
        raise ValueError("ring polynomial must be non-empty")
    return [int(a[0]), *(-int(a[i]) for i in range(len(a) - 1, 0, -1))]


def galois_conjugate(a: Sequence[int]) -> Polynomial:
    """Return ``a(-x)`` in coefficient representation."""

    if not a:
        raise ValueError("ring polynomial must be non-empty")
    return [int(coefficient) if i % 2 == 0 else -int(coefficient)
            for i, coefficient in enumerate(a)]


def lift(a: Sequence[int]) -> Polynomial:
    """Embed ``a(y)`` into the double-degree ring as ``a(x**2)``."""

    if not a:
        raise ValueError("polynomial to lift must be non-empty")
    result = [0] * (2 * len(a))
    result[::2] = [int(coefficient) for coefficient in a]
    return result


def field_norm(a: Sequence[int]) -> Polynomial:
    """Return the relative norm to the half-degree negacyclic subring.

    For ``a(x) = a_even(x**2) + x*a_odd(x**2)``, this computes
    ``a_even(y)**2 - y*a_odd(y)**2`` modulo ``y**(n/2) + 1``.
    """

    n = len(a)
    validate_power_of_two(n)
    if n < 2:
        raise ValueError("field norm requires ring degree at least two")

    even = [int(a[i]) for i in range(0, n, 2)]
    odd = [int(a[i]) for i in range(1, n, 2)]
    even_squared = negacyclic_mul(even, even)
    odd_squared = negacyclic_mul(odd, odd)

    # Subtract y*odd_squared in Z[y]/(y**(n/2) + 1).  The wrapped
    # highest-degree term changes sign once more because y**(n/2) = -1.
    result = even_squared[:]
    result[0] += odd_squared[-1]
    for i in range(1, len(result)):
        result[i] -= odd_squared[i - 1]
    return result


def centered_mod(value: int, q: int) -> int:
    """Return the canonical centered representative of ``value modulo q``.

    For odd ``q`` (the Falcon++ case), the range is
    ``[-(q-1)/2, (q-1)/2]``.  For even moduli, the positive representative
    ``q/2`` is chosen for the otherwise ambiguous endpoint.
    """

    _validate_modulus(q)
    residue = int(value) % q
    return residue - q if residue > q // 2 else residue


def coefficients_mod(
    a: Sequence[int],
    q: int,
    *,
    centered: bool = False,
) -> Polynomial:
    """Reduce all coefficients modulo ``q`` canonically or centrally."""

    _validate_modulus(q)
    if centered:
        return [centered_mod(coefficient, q) for coefficient in a]
    return [int(coefficient) % q for coefficient in a]


def _trim_mod(a: Sequence[int], q: int) -> Polynomial:
    result = [int(coefficient) % q for coefficient in a]
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return result or [0]


def _ordinary_add_mod(a: Sequence[int], b: Sequence[int], q: int) -> Polynomial:
    length = max(len(a), len(b))
    result = [0] * length
    for i in range(length):
        ai = int(a[i]) if i < len(a) else 0
        bi = int(b[i]) if i < len(b) else 0
        result[i] = (ai + bi) % q
    return _trim_mod(result, q)


def _ordinary_sub_mod(a: Sequence[int], b: Sequence[int], q: int) -> Polynomial:
    length = max(len(a), len(b))
    result = [0] * length
    for i in range(length):
        ai = int(a[i]) if i < len(a) else 0
        bi = int(b[i]) if i < len(b) else 0
        result[i] = (ai - bi) % q
    return _trim_mod(result, q)


def _ordinary_mul_mod(a: Sequence[int], b: Sequence[int], q: int) -> Polynomial:
    if not a or not b:
        return [0]
    result = [0] * (len(a) + len(b) - 1)
    for i, ai_raw in enumerate(a):
        ai = int(ai_raw) % q
        if ai == 0:
            continue
        for j, bj_raw in enumerate(b):
            result[i + j] = (result[i + j] + ai * (int(bj_raw) % q)) % q
    return _trim_mod(result, q)


def poly_divmod_mod(
    dividend: Sequence[int],
    divisor: Sequence[int],
    q: int,
) -> tuple[Polynomial, Polynomial]:
    """Divide ordinary polynomials over ``F_q``.

    Coefficients are in ascending order.  Falcon++ uses prime ``q``; for a
    composite modulus this routine raises ``ValueError`` if the leading
    coefficient of ``divisor`` is not invertible.
    """

    _validate_modulus(q)
    if not divisor:
        raise ZeroDivisionError("polynomial division by the zero polynomial")

    numerator = _trim_mod(dividend or [0], q)
    denominator = _trim_mod(divisor, q)
    if denominator == [0]:
        raise ZeroDivisionError("polynomial division by the zero polynomial")
    if len(numerator) < len(denominator):
        return [0], numerator

    try:
        inverse_lead = pow(denominator[-1], -1, q)
    except ValueError as exc:
        raise ValueError(
            "divisor leading coefficient is not invertible modulo q"
        ) from exc

    quotient = [0] * (len(numerator) - len(denominator) + 1)
    remainder = numerator[:]
    while remainder != [0] and len(remainder) >= len(denominator):
        shift = len(remainder) - len(denominator)
        factor = remainder[-1] * inverse_lead % q
        quotient[shift] = factor
        for i, coefficient in enumerate(denominator):
            remainder[i + shift] = (
                remainder[i + shift] - factor * coefficient
            ) % q
        remainder = _trim_mod(remainder, q)
    return _trim_mod(quotient, q), remainder


def poly_xgcd_mod(
    a: Sequence[int],
    b: Sequence[int],
    q: int,
) -> tuple[Polynomial, Polynomial, Polynomial]:
    """Extended GCD of ordinary polynomials over ``F_q``.

    Return monic ``(gcd, s, t)`` satisfying ``s*a + t*b = gcd (mod q)``.
    """

    _validate_modulus(q)
    old_r, r = _trim_mod(a or [0], q), _trim_mod(b or [0], q)
    old_s, s = [1], [0]
    old_t, t = [0], [1]

    while r != [0]:
        quotient, remainder = poly_divmod_mod(old_r, r, q)
        old_r, r = r, remainder
        old_s, s = s, _ordinary_sub_mod(
            old_s, _ordinary_mul_mod(quotient, s, q), q
        )
        old_t, t = t, _ordinary_sub_mod(
            old_t, _ordinary_mul_mod(quotient, t, q), q
        )

    if old_r == [0]:
        return [0], [0], [0]

    inverse_lead = pow(old_r[-1], -1, q)
    gcd = [(coefficient * inverse_lead) % q for coefficient in old_r]
    bezout_a = [(coefficient * inverse_lead) % q for coefficient in old_s]
    bezout_b = [(coefficient * inverse_lead) % q for coefficient in old_t]
    return (
        _trim_mod(gcd, q),
        _trim_mod(bezout_a, q),
        _trim_mod(bezout_b, q),
    )


def poly_gcd_mod(a: Sequence[int], b: Sequence[int], q: int) -> Polynomial:
    """Return the monic ordinary-polynomial GCD over ``F_q``."""

    return poly_xgcd_mod(a, b, q)[0]


def _inverse_mod_q_egcd(f: Sequence[int], q: int) -> Polynomial:
    """Original EGCD inversion, including its general-modulus error behavior."""

    if not f:
        raise ValueError("ring polynomial must be non-empty")
    _validate_modulus(q)
    n = len(f)
    modulus_polynomial = [1, *([0] * (n - 1)), 1]
    gcd, bezout_f, _ = poly_xgcd_mod(f, modulus_polynomial, q)
    if gcd != [1]:
        raise ValueError("polynomial is not invertible modulo (q, x**n + 1)")

    inverse = poly_mod_xn1(bezout_f, n, q)
    expected_one = [1, *([0] * (n - 1))]
    if negacyclic_mul(f, inverse, q) != expected_one:
        raise ArithmeticError("internal error while verifying polynomial inverse")
    return inverse


def _inverse_mod_q_recursive(f: Sequence[int], q: int) -> Polynomial:
    """Norm-tower inversion for a power-of-two degree and known prime q.

    Writing ``f=e(y)+x*o(y)``, ``y=x**2``, gives
    ``f*f(-x)=e(y)**2-y*o(y)**2`` in the half-degree ring.  Thus
    ``f**-1=f(-x)*lift(norm(f)**-1)``.  Reducing at every level bounds
    intermediate coefficients, while two half-degree products reconstruct
    the even and odd coefficients of the inverse.  The caller validates the
    domain and independently checks the final inverse identity.
    """

    n = len(f)
    if n == 1:
        try:
            return [pow(int(f[0]) % q, -1, q)]
        except ValueError as exc:
            raise ValueError(
                "polynomial is not invertible modulo (q, x**n + 1)"
            ) from exc
    even = [int(coefficient) % q for coefficient in f[::2]]
    odd = [int(coefficient) % q for coefficient in f[1::2]]
    even_squared = negacyclic_mul(even, even, q)
    odd_squared = negacyclic_mul(odd, odd, q)
    norm = [(even_squared[0] + odd_squared[-1]) % q]
    norm.extend(
        (even_squared[i] - odd_squared[i - 1]) % q
        for i in range(1, n // 2)
    )
    inverse_norm = _inverse_mod_q_recursive(norm, q)
    inverse_even = negacyclic_mul(even, inverse_norm, q)
    inverse_odd = negacyclic_mul(odd, inverse_norm, q)
    inverse = [0] * n
    inverse[::2] = inverse_even
    inverse[1::2] = [(-coefficient) % q for coefficient in inverse_odd]
    return inverse


def inverse_mod_q(f: Sequence[int], q: int) -> Polynomial:
    """Return ``f**-1`` in ``F_q[x] / (x**n + 1)``, checked exactly.

    Falcon++'s known prime moduli and power-of-two degrees use recursive
    relative-norm inversion.  All other domains retain the original EGCD
    path, including its historical exceptions for composite moduli.  Neither
    path needs roots of unity or an NTT.
    """

    if not f:
        raise ValueError("ring polynomial must be non-empty")
    _validate_modulus(q)
    n = len(f)
    if q not in _NORM_INVERSE_PRIME_MODULI or n & (n - 1):
        return _inverse_mod_q_egcd(f, q)
    inverse = _inverse_mod_q_recursive(f, q)
    if negacyclic_mul(f, inverse, q) != [1, *([0] * (n - 1))]:
        raise ArithmeticError("internal error while verifying polynomial inverse")
    return inverse


def is_invertible_mod_q(f: Sequence[int], q: int) -> bool:
    """Return whether ``f`` is a unit in ``F_q[x] / (x**n + 1)``."""

    try:
        inverse_mod_q(f, q)
    except ValueError:
        return False
    return True


def verify_ntru(
    f: Sequence[int],
    g: Sequence[int],
    capital_f: Sequence[int],
    capital_g: Sequence[int],
    q: int,
) -> bool:
    """Check the exact Falcon-sign convention ``f*G - g*F = q``."""

    n = _same_ring_degree(f, g)
    if len(capital_f) != n or len(capital_g) != n:
        raise ValueError("all four NTRU polynomials must have the same degree")
    lhs = sub(
        negacyclic_mul(f, capital_g),
        negacyclic_mul(g, capital_f),
    )
    return lhs == [int(q), *([0] * (n - 1))]


# Explicit alias for callers that want to distinguish quotient-ring inversion
# from integer inversion at the call site.
ring_inverse_mod_q = inverse_mod_q


__all__ = [
    "Polynomial",
    "add",
    "adjoint",
    "centered_mod",
    "coefficients_mod",
    "field_norm",
    "galois_conjugate",
    "inverse_mod_q",
    "is_invertible_mod_q",
    "lift",
    "neg",
    "negacyclic_mul",
    "poly_divmod_mod",
    "poly_gcd_mod",
    "poly_mod_xn1",
    "poly_xgcd_mod",
    "ring_inverse_mod_q",
    "scalar_mul",
    "sub",
    "validate_power_of_two",
    "verify_ntru",
]
