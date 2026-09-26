"""Canonical public-key payloads and Falcon-style hash-to-point.

The paper fixes the random-oracle input to ``H(h, salt, message)``.  At the
byte level this implementation uses exactly::

    canonical_h_payload || salt_41_bytes || message

There is no hidden length prefix, parameter identifier, or domain-separation
tag.  As in Falcon, successive SHAKE256 bytes are parsed as big-endian 16-bit
words and rejection-sampled modulo ``q``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from .parameters import FalconPPParameters, get_parameters
from .randomness import validate_salt


_SHAKE_CACHE_BYTES = 4096


def encode_public_key_payload(
    coefficients: Sequence[int],
    parameters: int | str | FalconPPParameters,
) -> bytes:
    """Encode ``h`` with fixed-width coefficients in Falcon bitstream order.

    Each coefficient is represented with ``ceil(log2(q))`` bits.  Coefficient
    zero is first, bits are most-significant first, and any final unused low
    bits in the last byte are zero.  The payload length is derived from the
    selected degree and modulus, without a parameter identifier or header.
    """

    params = get_parameters(parameters)
    if len(coefficients) != params.n:
        raise ValueError(
            f"public key must contain {params.n} coefficients; "
            f"received {len(coefficients)}"
        )

    width = params.public_key_coefficient_bits
    output = bytearray()
    accumulator = 0
    accumulated_bits = 0

    for index, coefficient in enumerate(coefficients):
        if isinstance(coefficient, bool) or not isinstance(coefficient, int):
            raise TypeError(f"public-key coefficient {index} is not an integer")
        if not 0 <= coefficient < params.q:
            raise ValueError(
                f"public-key coefficient {index}={coefficient} is outside "
                f"0..{params.q - 1}"
            )
        accumulator = (accumulator << width) | coefficient
        accumulated_bits += width
        while accumulated_bits >= 8:
            accumulated_bits -= 8
            output.append((accumulator >> accumulated_bits) & 0xFF)
            if accumulated_bits:
                accumulator &= (1 << accumulated_bits) - 1
            else:
                accumulator = 0

    if accumulated_bits:
        output.append((accumulator << (8 - accumulated_bits)) & 0xFF)

    encoded = bytes(output)
    if len(encoded) != params.public_key_payload_bytes:  # defensive invariant
        raise AssertionError("internal public-key length mismatch")
    return encoded


def decode_public_key_payload(
    payload: bytes | bytearray | memoryview,
    parameters: int | str | FalconPPParameters,
) -> list[int]:
    """Decode and validate a canonical fixed-width public-key payload."""

    params = get_parameters(parameters)
    expected = params.public_key_payload_bytes
    encoded = _as_fixed_bytes(payload, "public-key payload", expected)

    width = params.public_key_coefficient_bits
    mask = (1 << width) - 1
    accumulator = 0
    accumulated_bits = 0
    coefficients: list[int] = []

    for octet in encoded:
        accumulator = (accumulator << 8) | octet
        accumulated_bits += 8
        while accumulated_bits >= width and len(coefficients) < params.n:
            accumulated_bits -= width
            coefficient = (accumulator >> accumulated_bits) & mask
            if coefficient >= params.q:
                raise ValueError(
                    f"non-canonical public-key coefficient {coefficient} "
                    f"at index {len(coefficients)}"
                )
            coefficients.append(coefficient)
            if accumulated_bits:
                accumulator &= (1 << accumulated_bits) - 1
            else:
                accumulator = 0

    if len(coefficients) != params.n:
        raise ValueError("public-key payload ended before all coefficients")
    if accumulated_bits and accumulator:
        raise ValueError("non-zero public-key padding bits")
    return coefficients


def hash_to_point(
    canonical_h_payload: bytes | bytearray | memoryview,
    salt: bytes | bytearray | memoryview,
    message: bytes | bytearray | memoryview,
    parameters: int | str | FalconPPParameters,
) -> list[int]:
    """Map ``h || salt || message`` to ``R_q`` using Falcon rejection rules."""

    params = get_parameters(parameters)
    h_payload = _as_fixed_bytes(
        canonical_h_payload,
        "canonical_h_payload",
        params.public_key_payload_bytes,
    )
    # Decoding is intentional: fixed length alone does not rule out q..2^b-1.
    decode_public_key_payload(h_payload, params)
    salt_bytes = validate_salt(salt)
    message_bytes = _as_bytes(message, "message")

    reader = _ShakePrefixReader(h_payload + salt_bytes + message_bytes)
    rejection_limit = ((1 << 16) // params.q) * params.q
    point: list[int] = []
    while len(point) < params.n:
        word_bytes = reader.read(2)
        word = (word_bytes[0] << 8) | word_bytes[1]
        if word < rejection_limit:
            point.append(word % params.q)
    return point


def hash_to_point_from_coefficients(
    h: Sequence[int],
    salt: bytes | bytearray | memoryview,
    message: bytes | bytearray | memoryview,
    parameters: int | str | FalconPPParameters,
) -> list[int]:
    """Convenience wrapper that canonically packs ``h`` before hashing."""

    params = get_parameters(parameters)
    return hash_to_point(
        encode_public_key_payload(h, params), salt, message, params
    )


def _as_bytes(value: object, label: str) -> bytes:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError(f"{label} must be bytes-like")
    if type(value) is bytes:
        return value
    # Avoid invoking a user-defined ``__bytes__`` implementation on a bytes
    # or bytearray subclass at a wire boundary.  Copying through the buffer
    # protocol also freezes mutable inputs before parsing or hashing.
    return memoryview(value).tobytes()


def _as_fixed_bytes(value: object, label: str, expected_length: int) -> bytes:
    """Reject a wrong-length wire buffer before copying its contents."""

    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError(f"{label} must be bytes-like")
    if type(value) is bytes:
        if len(value) != expected_length:
            raise ValueError(
                f"{label} must contain {expected_length} bytes; "
                f"received {len(value)}"
            )
        return value
    view = memoryview(value)
    if view.nbytes != expected_length:
        raise ValueError(
            f"{label} must contain {expected_length} bytes; "
            f"received {view.nbytes}"
        )
    return view.tobytes()


class _ShakePrefixReader:
    """Small consuming reader for the standard-library SHAKE prefix API."""

    __slots__ = ("_shake", "_cache", "_offset")

    def __init__(self, data: bytes):
        self._shake = hashlib.shake_256(data)
        self._cache = b""
        self._offset = 0

    def read(self, length: int) -> bytes:
        if length < 0:
            raise ValueError("length must be non-negative")
        end = self._offset + length
        if end > len(self._cache):
            target = max(end, _SHAKE_CACHE_BYTES, 2 * len(self._cache))
            self._cache = self._shake.digest(target)
        result = self._cache[self._offset : end]
        self._offset = end
        return result


# Short aliases used by key/signature modules.
encode_h = encode_public_key_payload
decode_h = decode_public_key_payload
serialize_public_key = encode_public_key_payload
deserialize_public_key = decode_public_key_payload


__all__ = [
    "encode_public_key_payload",
    "decode_public_key_payload",
    "encode_h",
    "decode_h",
    "serialize_public_key",
    "deserialize_public_key",
    "hash_to_point",
    "hash_to_point_from_coefficients",
]
