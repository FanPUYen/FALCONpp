"""Randomness primitives for the readable Falcon++ implementation.

Production callers seed :class:`Shake256PRNG` from the operating-system CSPRNG.
Tests can inject a byte seed and obtain a reproducible stream.  The stream is
the standard ``SHAKE256(seed)`` XOF output; this deliberately avoids depending
on a platform RNG after construction and avoids an optional crypto package.
"""

from __future__ import annotations

import hashlib
import os
import threading
from typing import Protocol, runtime_checkable

from .parameters import SALT_BYTES, SALT_UNUSED_HIGH_BITS


DEFAULT_SEED_BYTES = 48
_INITIAL_CACHE_BYTES = 4096


@runtime_checkable
class RandomByteSource(Protocol):
    """Structural interface used by the samplers and salt generator."""

    def read(self, length: int) -> bytes:
        """Return exactly ``length`` random bytes."""


class Shake256PRNG:
    """A deterministic byte stream backed by ``hashlib.shake_256``.

    Python's standard SHAKE object exposes prefix extraction instead of a
    consuming ``read`` method.  This wrapper grows a cached prefix
    geometrically, so splitting a read never changes the resulting stream.
    Instances should be scoped to one key-generation or signing operation;
    that also bounds the retained cache.
    """

    __slots__ = ("_shake", "_cache", "_offset", "_lock")

    def __init__(self, seed: bytes | bytearray | memoryview):
        seed_bytes = _as_bytes(seed, "seed")
        self._shake = hashlib.shake_256(seed_bytes)
        self._cache = b""
        self._offset = 0
        self._lock = threading.Lock()

    @classmethod
    def from_system(cls, seed_bytes: int = DEFAULT_SEED_BYTES) -> "Shake256PRNG":
        """Seed a new stream with bytes obtained from ``os.urandom``."""

        _validate_length(seed_bytes, "seed_bytes")
        if seed_bytes < 32:
            raise ValueError("a production seed must contain at least 32 bytes")
        return cls(os.urandom(seed_bytes))

    @property
    def bytes_consumed(self) -> int:
        """Number of stream bytes returned so far."""

        return self._offset

    def read(self, length: int) -> bytes:
        """Consume and return exactly ``length`` bytes from the SHAKE stream."""

        _validate_length(length, "length")
        if length == 0:
            return b""
        with self._lock:
            end = self._offset + length
            if end > len(self._cache):
                target = max(end, _INITIAL_CACHE_BYTES, 2 * len(self._cache))
                self._cache = self._shake.digest(target)
            result = self._cache[self._offset : end]
            self._offset = end
            return result

    def random_bytes(self, length: int) -> bytes:
        """Alias for :meth:`read`, matching common sampler APIs."""

        return self.read(length)

    def __call__(self, length: int) -> bytes:
        return self.read(length)

    def randbits(self, bit_count: int) -> int:
        """Return a uniform integer in ``range(2**bit_count)``."""

        _validate_length(bit_count, "bit_count")
        if bit_count == 0:
            return 0
        byte_count = (bit_count + 7) // 8
        value = int.from_bytes(self.read(byte_count), "big")
        return value & ((1 << bit_count) - 1)

    def randbelow(self, upper_bound: int) -> int:
        """Return a uniform integer in ``range(upper_bound)`` by rejection."""

        if isinstance(upper_bound, bool) or not isinstance(upper_bound, int):
            raise TypeError("upper_bound must be an integer")
        if upper_bound <= 0:
            raise ValueError("upper_bound must be positive")
        if upper_bound == 1:
            return 0
        bits = (upper_bound - 1).bit_length()
        while True:
            candidate = self.randbits(bits)
            if candidate < upper_bound:
                return candidate


def make_prng(
    seed: bytes | bytearray | memoryview | None = None,
) -> Shake256PRNG:
    """Create a deterministic seeded stream or a production system-seeded one."""

    if seed is None:
        return Shake256PRNG.from_system()
    return Shake256PRNG(seed)


def read_random_bytes(
    source: RandomByteSource | object | None,
    length: int,
) -> bytes:
    """Read from common random-source shapes and enforce the output length.

    Supported sources expose ``read(n)`` or ``random_bytes(n)``, or are plain
    callables accepting ``n``.  ``None`` creates a fresh OS-seeded SHAKE stream.
    This adapter keeps the cryptographic modules independent of a specific RNG
    class while retaining strict length checks.
    """

    _validate_length(length, "length")
    if source is None:
        output = Shake256PRNG.from_system().read(length)
    elif callable(getattr(source, "read", None)):
        output = source.read(length)  # type: ignore[attr-defined]
    elif callable(getattr(source, "random_bytes", None)):
        output = source.random_bytes(length)  # type: ignore[attr-defined]
    elif callable(source):
        output = source(length)  # type: ignore[operator]
    else:
        raise TypeError(
            "random source must expose read/random_bytes or be callable"
        )
    output_bytes = _as_bytes(output, "random-source output")
    if len(output_bytes) != length:
        raise ValueError(
            f"random source returned {len(output_bytes)} bytes; expected {length}"
        )
    return output_bytes


def generate_salt(source: RandomByteSource | object | None = None) -> bytes:
    """Generate a uniformly random 325-bit salt in its 41-byte encoding.

    The most significant three bits of the first byte are reserved and set to
    zero.  Clearing them maps uniform 328-bit input onto uniform 325-bit salts.
    """

    salt = bytearray(read_random_bytes(source, SALT_BYTES))
    salt[0] &= 0xFF >> SALT_UNUSED_HIGH_BITS
    return bytes(salt)


def is_valid_salt(salt: object) -> bool:
    """Return whether ``salt`` is the canonical 41-byte/325-bit encoding."""

    try:
        value = _as_fixed_bytes(salt, "salt", SALT_BYTES)
    except (TypeError, ValueError):
        return False
    reserved_mask = ((1 << SALT_UNUSED_HIGH_BITS) - 1) << (
        8 - SALT_UNUSED_HIGH_BITS
    )
    return not value[0] & reserved_mask


def validate_salt(salt: bytes | bytearray | memoryview) -> bytes:
    """Return an immutable canonical salt or raise ``ValueError``."""

    value = _as_fixed_bytes(salt, "salt", SALT_BYTES)
    if not is_valid_salt(value):
        raise ValueError("the three reserved high salt bits must be zero")
    return value


def _as_bytes(value: object, label: str) -> bytes:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError(f"{label} must be bytes-like")
    if type(value) is bytes:
        return value
    return memoryview(value).tobytes()


def _as_fixed_bytes(value: object, label: str, expected_length: int) -> bytes:
    """Validate a fixed-size buffer before making any nontrivial copy."""

    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError(f"{label} must be bytes-like")
    if type(value) is bytes:
        if len(value) != expected_length:
            raise ValueError(
                f"{label} must contain exactly {expected_length} bytes"
            )
        return value
    view = memoryview(value)
    if view.nbytes != expected_length:
        raise ValueError(f"{label} must contain exactly {expected_length} bytes")
    # The length check above bounds this copy and bypasses arbitrary
    # ``__bytes__`` implementations on bytes/bytearray subclasses.
    return view.tobytes()


def _validate_length(value: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value < 0:
        raise ValueError(f"{label} must be non-negative")


__all__ = [
    "DEFAULT_SEED_BYTES",
    "RandomByteSource",
    "Shake256PRNG",
    "make_prng",
    "read_random_bytes",
    "generate_salt",
    "is_valid_salt",
    "validate_salt",
]
