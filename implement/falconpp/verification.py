"""Strict Falcon++ signature verification.

Verification treats the public key and signature as adversarial byte strings:
both must decode canonically, the 325-bit salt representation is checked, and
the fixed-length formal rANS payload must be consumed completely.  The ring
equation and squared norm are then evaluated with exact Python integers.
Until the fixed rANS policy is frozen, formal verification requires the same
explicit ``allow_provisional_wire=True`` acknowledgement as signing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .hash_to_point import hash_to_point
from .keys import PublicKey, Signature
from .parameters import FalconPPParameters, get_parameters
from .polynomial import centered_mod, negacyclic_mul
from .rans import RANSError
from .signing import (
    SignatureCodec,
    _require_signature_wire_acknowledgement,
    make_signature_codec,
)


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """Detailed verifier result; malformed external input never raises."""

    valid: bool
    reason: str
    squared_norm: int | None = None
    s1: tuple[int, ...] | None = None
    s2: tuple[int, ...] | None = None

    def __bool__(self) -> bool:
        return self.valid


def _resolve_public_key(
    public_key: PublicKey | bytes | bytearray | memoryview,
    parameters: int | str | FalconPPParameters | None,
) -> PublicKey:
    if isinstance(public_key, PublicKey):
        if parameters is not None and get_parameters(parameters) != public_key.parameters:
            raise ValueError("public key and verifier use different parameter sets")
        return public_key
    if parameters is None:
        raise ValueError("parameters are required when the public key is bytes")
    return PublicKey.from_payload(public_key, parameters)


def verify_signature_detailed(
    public_key: PublicKey | bytes | bytearray | memoryview,
    message: bytes | bytearray | memoryview,
    signature: Signature | bytes | bytearray | memoryview,
    *,
    parameters: int | str | FalconPPParameters | None = None,
    codec: SignatureCodec | None = None,
    allow_provisional_wire: bool = False,
) -> VerificationResult:
    """Verify a canonical Falcon++ signature and return an audit-friendly result."""

    try:
        resolved_key = _resolve_public_key(public_key, parameters)
    except (TypeError, ValueError, KeyError):
        return VerificationResult(False, "invalid-public-key")

    if not isinstance(message, (bytes, bytearray, memoryview)):
        return VerificationResult(False, "invalid-message-type")
    message_bytes = bytes(message)
    params = resolved_key.parameters

    signature_codec = codec or make_signature_codec(params, formal=True)
    if signature_codec.parameters != params:
        return VerificationResult(False, "parameter-mismatch")
    _require_signature_wire_acknowledgement(
        signature_codec, allow_provisional_wire
    )

    try:
        if isinstance(signature, Signature):
            structured = signature
            if (
                signature_codec.payload_length is not None
                and len(structured.encoded_s2) != signature_codec.payload_length
            ):
                return VerificationResult(False, "invalid-signature-length")
        else:
            structured = Signature.from_wire(
                signature, encoded_length=signature_codec.payload_length
            )
        s2 = signature_codec.decode(structured.encoded_s2)
        # Keep the re-encoding test explicit at the scheme boundary even
        # though strict rANS decoding already enforces it internally.
        if signature_codec.encode(s2) != structured.encoded_s2:
            return VerificationResult(False, "non-canonical-signature")
    except (RANSError, TypeError, ValueError, OverflowError):
        return VerificationResult(False, "malformed-signature")

    try:
        point = hash_to_point(
            resolved_key.payload, structured.salt, message_bytes, params
        )
        product = negacyclic_mul(resolved_key.h, s2, params.q)
        s1 = tuple(
            centered_mod(c - hs2, params.q)
            for c, hs2 in zip(point, product, strict=True)
        )
        squared_norm = sum(value * value for value in (*s1, *s2))
    except (ArithmeticError, TypeError, ValueError):
        return VerificationResult(False, "verification-arithmetic-failure")

    # An explicit equation check makes the coefficient convention auditable.
    relation = negacyclic_mul(resolved_key.h, s2, params.q)
    if any(
        (left + right - target) % params.q
        for left, right, target in zip(s1, relation, point, strict=True)
    ):
        return VerificationResult(
            False, "relation-failure", squared_norm, s1, tuple(s2)
        )
    if squared_norm > params.signature_norm_bound:
        return VerificationResult(
            False, "norm-bound", squared_norm, s1, tuple(s2)
        )
    return VerificationResult(True, "ok", squared_norm, s1, tuple(s2))


def verify_signature(
    public_key: PublicKey | bytes | bytearray | memoryview,
    message: bytes | bytearray | memoryview,
    signature: Signature | bytes | bytearray | memoryview,
    *,
    parameters: int | str | FalconPPParameters | None = None,
    codec: SignatureCodec | None = None,
    allow_provisional_wire: bool = False,
) -> bool:
    """Boolean convenience wrapper around :func:`verify_signature_detailed`."""

    return verify_signature_detailed(
        public_key,
        message,
        signature,
        parameters=parameters,
        codec=codec,
        allow_provisional_wire=allow_provisional_wire,
    ).valid


# Short spelling used by the object-oriented facade.
verify = verify_signature


__all__ = [
    "VerificationResult",
    "verify",
    "verify_signature",
    "verify_signature_detailed",
]
