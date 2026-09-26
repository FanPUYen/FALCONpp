"""In-memory key and signature containers for Falcon++.

The project deliberately postpones a secret-key wire format.  A
:class:`SecretKey` therefore retains all four NTRU polynomials and the
expanded FFT sampling data in memory.  Only public keys and signatures have a
wire representation here.

These classes are validation-oriented research interfaces.  They are not
constant-time containers and do not attempt to erase secret material.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from typing import Any, Iterator, Sequence

from .hash_to_point import (
    decode_public_key_payload,
    encode_public_key_payload,
)
from .parameters import SALT_BYTES, FalconPPParameters, get_parameters
from .polynomial import inverse_mod_q, negacyclic_mul, verify_ntru
from .randomness import validate_salt


IntegerPolynomial = tuple[int, ...]
FFTPolynomial = tuple[Any, ...]
FFTBasis = tuple[
    tuple[FFTPolynomial, FFTPolynomial],
    tuple[FFTPolynomial, FFTPolynomial],
]


def _wire_memoryview(value: Any, name: str) -> memoryview:
    """Return a one-dimensional octet view for an external wire value."""

    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError(f"{name} must be bytes-like")
    try:
        view = memoryview(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must expose a live byte buffer") from exc
    if view.ndim != 1 or view.itemsize != 1 or view.format not in {"B", "b", "c"}:
        raise TypeError(f"{name} must be a one-dimensional byte buffer")
    return view


def _canonical_bytes_like(
    value: Any,
    name: str,
    *,
    expected_length: int | None = None,
    minimum_length: int | None = None,
) -> bytes:
    """Freeze an explicitly bytes-like wire value.

    Calling ``bytes(value)`` without this guard is too permissive at an
    adversarial boundary: integers mean a zero-filled allocation and lists of
    integers are silently accepted as encodings.
    """

    view = _wire_memoryview(value, name)
    if expected_length is not None and view.nbytes != expected_length:
        raise ValueError(
            f"{name} must contain exactly {expected_length} bytes; "
            f"received {view.nbytes}"
        )
    if minimum_length is not None and view.nbytes < minimum_length:
        raise ValueError(
            f"{name} must contain at least {minimum_length} bytes; "
            f"received {view.nbytes}"
        )
    if type(value) is bytes:
        return value
    # Copy through the buffer protocol so even a bytes/bytearray subclass
    # cannot inject an arbitrary Python ``__bytes__`` implementation.
    return view.tobytes()


def _integer_polynomial(
    values: Sequence[int],
    parameters: FalconPPParameters,
    name: str,
) -> IntegerPolynomial:
    """Return a checked immutable integer polynomial of the right degree."""

    try:
        polynomial = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of integers") from exc
    if len(polynomial) != parameters.n:
        raise ValueError(
            f"{name} must have degree {parameters.n}; received "
            f"{len(polynomial)} coefficients"
        )
    for index, coefficient in enumerate(polynomial):
        if isinstance(coefficient, bool) or not isinstance(coefficient, int):
            raise TypeError(f"{name}[{index}] is not an integer")
    return polynomial


def _freeze_fft_basis(basis: Any, degree: int) -> FFTBasis:
    """Validate and recursively freeze a 2-by-2 FFT polynomial matrix."""

    try:
        # Do not coerce values through Python ``complex`` here: the formal
        # signing path stores mpmath values and would silently lose precision.
        rows = tuple(tuple(tuple(entry) for entry in row)
                     for row in basis)
    except TypeError as exc:
        raise TypeError("basis_fft must be a 2-by-2 matrix of FFT vectors") from exc
    if len(rows) != 2 or any(len(row) != 2 for row in rows):
        raise ValueError("basis_fft must be a 2-by-2 matrix")
    if any(len(rows[i][j]) != degree for i in range(2) for j in range(2)):
        raise ValueError("each FFT basis entry must have the key's ring degree")
    return rows  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class PublicKey:
    """A Falcon++ public polynomial and its canonical payload."""

    parameters: FalconPPParameters
    h: IntegerPolynomial
    payload: bytes = field(repr=False)

    def __post_init__(self) -> None:
        parameters = get_parameters(self.parameters)
        h = _integer_polynomial(self.h, parameters, "h")
        if any(not 0 <= coefficient < parameters.q for coefficient in h):
            raise ValueError("public-key coefficients must be canonical modulo q")
        payload = _canonical_bytes_like(
            self.payload,
            "public-key payload",
            expected_length=parameters.public_key_payload_bytes,
        )
        canonical = encode_public_key_payload(h, parameters)
        if payload != canonical:
            raise ValueError("public-key payload is not the canonical encoding of h")
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "h", h)
        object.__setattr__(self, "payload", payload)

    @classmethod
    def from_coefficients(
        cls,
        h: Sequence[int],
        parameters: int | str | FalconPPParameters,
    ) -> "PublicKey":
        params = get_parameters(parameters)
        coefficients = _integer_polynomial(h, params, "h")
        payload = encode_public_key_payload(coefficients, params)
        return cls(params, coefficients, payload)

    @classmethod
    def from_payload(
        cls,
        payload: bytes | bytearray | memoryview,
        parameters: int | str | FalconPPParameters,
    ) -> "PublicKey":
        params = get_parameters(parameters)
        encoded = _canonical_bytes_like(
            payload,
            "public-key payload",
            expected_length=params.public_key_payload_bytes,
        )
        h = tuple(decode_public_key_payload(encoded, params))
        return cls(params, h, encoded)

    def __bytes__(self) -> bytes:
        return self.payload


@dataclass(frozen=True, slots=True)
class SecretKey:
    """Expanded in-memory secret key.

    ``sampler_tree`` is intentionally typed as ``Any`` to keep this data
    container independent of the numerical implementation.  Construction
    still checks the exact NTRU equation and public-key relation.
    """

    parameters: FalconPPParameters
    f: IntegerPolynomial
    g: IntegerPolynomial
    capital_f: IntegerPolynomial
    capital_g: IntegerPolynomial
    public_key: PublicKey
    basis_fft: FFTBasis = field(repr=False)
    sampler_tree: Any = field(repr=False)
    _inverse_f_hint: InitVar[Sequence[int] | None] = field(default=None, kw_only=True)

    def __post_init__(self, _inverse_f_hint: Sequence[int] | None = None) -> None:
        parameters = get_parameters(self.parameters)
        f = _integer_polynomial(self.f, parameters, "f")
        g = _integer_polynomial(self.g, parameters, "g")
        capital_f = _integer_polynomial(self.capital_f, parameters, "F")
        capital_g = _integer_polynomial(self.capital_g, parameters, "G")
        if self.public_key.parameters != parameters:
            raise ValueError("secret and public keys use different parameter sets")
        if not verify_ntru(f, g, capital_f, capital_g, parameters.q):
            raise ValueError("secret polynomials do not satisfy f*G - g*F = q")
        if _inverse_f_hint is None:
            try:
                inverse_f = inverse_mod_q(f, parameters.q)
            except ValueError as exc:
                raise ValueError("f is not invertible modulo (q, x^n + 1)") from exc
        else:
            # KeyGen already computed an inverse.  Verify the supplied witness
            # by exact multiplication instead of performing a second extended
            # Euclidean inversion.  Even an external caller cannot bypass the
            # invertibility check with this optional, non-stored init argument.
            inverse_f = _integer_polynomial(_inverse_f_hint, parameters, "inverse_f")
            if any(not 0 <= coefficient < parameters.q for coefficient in inverse_f):
                raise ValueError("inverse witness must be canonical modulo q")
            unit = [1, *([0] * (parameters.n - 1))]
            if negacyclic_mul(f, inverse_f, parameters.q) != unit:
                raise ValueError("supplied inverse witness does not satisfy f*inverse = 1")
        expected_h = tuple(
            negacyclic_mul(g, inverse_f, parameters.q)
        )
        if self.public_key.h != expected_h:
            raise ValueError("public key is not h = g/f modulo q")
        basis_fft = _freeze_fft_basis(self.basis_fft, parameters.n)

        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "f", f)
        object.__setattr__(self, "g", g)
        object.__setattr__(self, "capital_f", capital_f)
        object.__setattr__(self, "capital_g", capital_g)
        object.__setattr__(self, "basis_fft", basis_fft)

    @property
    def F(self) -> IntegerPolynomial:  # noqa: N802 - standard NTRU notation
        return self.capital_f

    @property
    def G(self) -> IntegerPolynomial:  # noqa: N802 - standard NTRU notation
        return self.capital_g


@dataclass(frozen=True, slots=True)
class Signature:
    """A structured signature ``(salt, canonical_encoded_s2)``.

    The parameter set is external, as it is for Falcon.  Consequently the
    byte representation is simply the canonical 41-byte salt followed by the
    codec payload; no hidden parameter identifier is inserted.
    """

    salt: bytes
    encoded_s2: bytes

    def __post_init__(self) -> None:
        # Salt has a fixed external size, so reject oversized buffers before
        # validate_salt() materializes them.
        salt = validate_salt(
            _canonical_bytes_like(
                self.salt,
                "salt",
                expected_length=SALT_BYTES,
            )
        )
        encoded = _canonical_bytes_like(self.encoded_s2, "encoded_s2")
        if not encoded:
            raise ValueError("encoded_s2 must not be empty")
        object.__setattr__(self, "salt", salt)
        object.__setattr__(self, "encoded_s2", encoded)

    @property
    def wire(self) -> bytes:
        return self.salt + self.encoded_s2

    def __bytes__(self) -> bytes:
        return self.wire

    @classmethod
    def from_wire(
        cls,
        wire: bytes | bytearray | memoryview,
        *,
        encoded_length: int | None = None,
    ) -> "Signature":
        salt_bytes = SALT_BYTES
        if encoded_length is not None:
            if isinstance(encoded_length, bool) or not isinstance(encoded_length, int):
                raise TypeError("encoded_length must be an integer")
            if encoded_length <= 0:
                raise ValueError("encoded_length must be positive")
            expected = salt_bytes + encoded_length
            # Check the buffer metadata before copying.  Formal verification
            # therefore never duplicates an attacker-controlled oversized
            # object merely to discover that its length is wrong.
            data = _canonical_bytes_like(
                wire,
                "signature wire value",
                expected_length=expected,
            )
        else:
            # Raw mode is explicitly variable-length, so there is no external
            # upper bound to enforce.  The lower bound can still be checked
            # through buffer metadata before materializing the input.
            data = _canonical_bytes_like(
                wire,
                "signature wire value",
                minimum_length=salt_bytes + 1,
            )
        return cls(data[:salt_bytes], data[salt_bytes:])


@dataclass(frozen=True, slots=True)
class KeyGenerationStatistics:
    """Rejection counters accumulated while producing one accepted key."""

    sampled_pairs: int
    norm_rejections: int = 0
    invertibility_rejections: int = 0
    gram_schmidt_rejections: int = 0
    ntru_rejections: int = 0

    @property
    def rejected_pairs(self) -> int:
        return self.sampled_pairs - 1


@dataclass(frozen=True, slots=True)
class KeyPair:
    """Accepted key pair plus transparent KeyGen rejection statistics."""

    secret_key: SecretKey
    public_key: PublicKey
    statistics: KeyGenerationStatistics

    def __post_init__(self) -> None:
        if self.secret_key.public_key != self.public_key:
            raise ValueError("key-pair public and secret components disagree")

    def __iter__(self) -> Iterator[SecretKey | PublicKey]:
        """Allow the conventional ``sk, pk = keypair`` spelling."""

        yield self.secret_key
        yield self.public_key


__all__ = [
    "FFTBasis",
    "FFTPolynomial",
    "IntegerPolynomial",
    "KeyGenerationStatistics",
    "KeyPair",
    "PublicKey",
    "SecretKey",
    "Signature",
]
