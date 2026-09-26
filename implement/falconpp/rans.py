"""Readable, canonical byte-rANS coding for Falcon++ experiments.

This module is an independent implementation of the generic construction in
Algorithms 10--11 of the Falconws paper.  It intentionally does *not* reuse
the HuFu-specific dimensions, empirical frequency tables, or seven-bit split
from the accompanying C code.

There are two layers:

``rans_encode`` / ``rans_decode``
    A conventional byte-aligned rANS stack with a fixed initial state.  Strict
    decoding checks the packed state, the final state, complete input
    consumption, and (by default) byte-for-byte canonical re-encoding.

``encode_raw`` / ``decode_raw``
    A signed-integer codec.  Each magnitude is split into a small raw low part
    and an entropy-coded high quotient.  An escape symbol plus canonical
    unsigned LEB128 covers every coefficient permitted by the model; there is
    no hidden tail rejection.

``encode_padded`` / ``decode_padded``
    The fixed-length ``00 ... 00 || 80 || raw-stream`` wrapper proposed by
    Falconws.  The fixed length is a parameter-set property supplied by the
    caller, never a value parsed from an untrusted signature.

The default model uses a 16-bit frequency scale as an engineering default.
It is configurable because the final Falcon++ wire-format policy has not yet
been frozen.  ``analyze_padded_length`` returns a certified (but deliberately
conservative) length bound and explicitly flags that policy decision.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from decimal import ROUND_CEILING, Decimal, localcontext
from hashlib import sha256
from math import ceil, log2
from typing import Iterable, Sequence


RANS_BYTE_L = 1 << 23
RANS_RADIX_BITS = 8
RANS_RADIX = 1 << RANS_RADIX_BITS
RAW_CODEC_VERSION = 1
PAD_MARKER = 0x80
PADDED_LENGTH_ALGORITHM = "norm-layer-cake-rans-v1"


class RANSError(ValueError):
    """Base class for rANS format, model, and value errors."""


class RANSModelError(RANSError):
    """Raised when a frequency table or signed model is invalid."""


class RANSEncodeError(RANSError):
    """Raised when values cannot be represented under the requested model."""


class RANSDecodeError(RANSError):
    """Raised for malformed, non-canonical, or truncated encodings."""


def _require_plain_int(value: int, name: str) -> int:
    # bool is an int subclass but accepting it in a wire-format API tends to
    # hide caller mistakes.
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    return value


def _require_bytes_like(
    value: bytes | bytearray | memoryview,
    name: str = "data",
) -> bytes:
    """Copy a wire value without invoking permissive ``bytes(value)`` forms."""

    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError(f"{name} must be bytes, bytearray, or memoryview")
    if type(value) is bytes:
        return value
    return memoryview(value).tobytes()


def _require_exact_length_bytes_like(
    value: bytes | bytearray | memoryview,
    expected_length: int,
    name: str = "data",
) -> bytes:
    """Validate a fixed-size wire object before making a bounded copy.

    In particular, an attacker-controlled oversized ``bytearray`` or
    ``memoryview`` is rejected from its buffer metadata.  A bytes subclass is
    copied through the buffer protocol, never through an overridable
    ``__bytes__`` method.
    """

    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError(f"{name} must be bytes, bytearray, or memoryview")
    view = memoryview(value)
    if view.nbytes != expected_length:
        raise RANSDecodeError(
            f"fixed rANS object has length {view.nbytes}, expected {expected_length}"
        )
    if type(value) is bytes:
        return value
    return view.tobytes()


@dataclass(frozen=True)
class RANSFrequencyTable:
    """An immutable normalized frequency table.

    Frequencies are ordered by integer symbol value and must be strictly
    positive.  Their sum is exactly ``2**scale_bits``.
    """

    frequencies: tuple[int, ...]
    scale_bits: int = 16
    cumulative: tuple[int, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        frequencies = tuple(self.frequencies)
        object.__setattr__(self, "frequencies", frequencies)
        if isinstance(self.scale_bits, bool) or not isinstance(self.scale_bits, int):
            raise RANSModelError("scale_bits must be an integer")
        if not 1 <= self.scale_bits <= 16:
            raise RANSModelError("scale_bits must lie in [1, 16]")
        if not frequencies:
            raise RANSModelError("the frequency table must not be empty")
        if any(isinstance(f, bool) or not isinstance(f, int) or f <= 0 for f in frequencies):
            raise RANSModelError("all symbol frequencies must be positive integers")
        scale = 1 << self.scale_bits
        if sum(frequencies) != scale:
            raise RANSModelError(
                f"frequencies sum to {sum(frequencies)}, expected {scale}"
            )
        starts: list[int] = []
        total = 0
        for frequency in frequencies:
            starts.append(total)
            total += frequency
        object.__setattr__(self, "cumulative", tuple(starts))

    @property
    def scale(self) -> int:
        return 1 << self.scale_bits

    @property
    def mask(self) -> int:
        return self.scale - 1

    @property
    def symbol_count(self) -> int:
        return len(self.frequencies)

    @property
    def fingerprint(self) -> str:
        """Stable short identifier for recording a frozen model in artifacts."""

        width = max(1, (max(self.frequencies).bit_length() + 7) // 8)
        material = bytearray((self.scale_bits,))
        material.extend(len(self.frequencies).to_bytes(4, "little"))
        for frequency in self.frequencies:
            material.extend(frequency.to_bytes(width, "little"))
        return sha256(material).hexdigest()[:16]


@dataclass(frozen=True)
class SignedRANSModel:
    """Model for signed coefficients with an rANS-coded high quotient.

    Symbols ``0 .. direct_quotients-1`` encode that quotient directly.  The
    last symbol is ESCAPE; a following canonical ULEB128 value encodes
    ``quotient - direct_quotients``.  Consequently every signed integer in
    ``[-coefficient_bound, coefficient_bound]`` is representable.
    """

    table: RANSFrequencyTable
    low_bits: int
    direct_quotients: int
    coefficient_bound: int
    sigma: str
    construction_precision: int = 80

    def __post_init__(self) -> None:
        if isinstance(self.low_bits, bool) or not isinstance(self.low_bits, int):
            raise RANSModelError("low_bits must be an integer")
        if not 0 <= self.low_bits <= 15:
            raise RANSModelError("low_bits must lie in [0, 15]")
        if isinstance(self.direct_quotients, bool) or not isinstance(
            self.direct_quotients, int
        ):
            raise RANSModelError("direct_quotients must be an integer")
        if self.direct_quotients < 1:
            raise RANSModelError("direct_quotients must be positive")
        if self.table.symbol_count != self.direct_quotients + 1:
            raise RANSModelError(
                "the table must contain direct_quotients symbols plus ESCAPE"
            )
        if isinstance(self.coefficient_bound, bool) or not isinstance(
            self.coefficient_bound, int
        ):
            raise RANSModelError("coefficient_bound must be an integer")
        if self.coefficient_bound < 0:
            raise RANSModelError("coefficient_bound must be non-negative")
        sigma = Decimal(self.sigma)
        if not sigma.is_finite() or sigma <= 0:
            raise RANSModelError("sigma must be finite and positive")
        if isinstance(self.construction_precision, bool) or not isinstance(
            self.construction_precision, int
        ):
            raise RANSModelError("construction_precision must be an integer")
        if self.construction_precision < 32:
            raise RANSModelError("construction_precision must be at least 32 digits")

    @property
    def escape_symbol(self) -> int:
        return self.direct_quotients

    @property
    def low_mask(self) -> int:
        return (1 << self.low_bits) - 1

    @property
    def unit_bits(self) -> int:
        return self.low_bits + 1  # low magnitude bits followed by a sign bit

    @property
    def fingerprint(self) -> str:
        material = (
            f"falconpp-rans-v{RAW_CODEC_VERSION}|{self.sigma}|"
            f"{self.low_bits}|{self.direct_quotients}|"
            f"{self.coefficient_bound}|{self.table.fingerprint}"
        ).encode("ascii")
        return sha256(material).hexdigest()[:16]


@dataclass(frozen=True)
class PaddedLengthAnalysis:
    """Deterministic sizing result for a norm-bounded coefficient vector."""

    length_algorithm: str
    model_fingerprint: str
    frequency_table_fingerprint: str
    dimension: int
    norm_bound: int
    certified_upper_bound: int
    estimated_mean_length: int
    rans_stream_upper_bound: int
    side_bits_bytes: int
    escape_payload_upper_bound: int
    escape_payload_estimated_bytes: int
    certified: bool
    size_tight: bool
    wire_policy_resolved: bool
    note: str


def largest_remainder_frequencies(
    weights: Sequence[Decimal | int | str | float],
    *,
    scale_bits: int = 16,
    min_frequency: int = 1,
    decimal_precision: int = 80,
) -> tuple[int, ...]:
    """Quantize non-negative weights deterministically by largest remainder.

    A ``min_frequency`` seat is reserved for every symbol first, then the
    remaining mass is apportioned proportionally.  Equal fractional
    remainders are resolved by ascending symbol index.  Decimal arithmetic
    makes construction reproducible and independent of binary libm rounding.
    """

    if isinstance(scale_bits, bool) or not isinstance(scale_bits, int):
        raise RANSModelError("scale_bits must be an integer")
    if not 1 <= scale_bits <= 16:
        raise RANSModelError("scale_bits must lie in [1, 16]")
    if isinstance(min_frequency, bool) or not isinstance(min_frequency, int):
        raise RANSModelError("min_frequency must be an integer")
    if min_frequency < 1:
        raise RANSModelError("min_frequency must be at least one")
    if isinstance(decimal_precision, bool) or not isinstance(decimal_precision, int):
        raise RANSModelError("decimal_precision must be an integer")
    if decimal_precision < 32:
        raise RANSModelError("decimal_precision must be at least 32 digits")
    if not weights:
        raise RANSModelError("at least one weight is required")

    total_frequency = 1 << scale_bits
    if len(weights) * min_frequency > total_frequency:
        raise RANSModelError("the minimum frequencies exceed the available scale")

    with localcontext() as context:
        context.prec = decimal_precision
        converted = tuple(Decimal(str(weight)) for weight in weights)
        if any(not weight.is_finite() or weight < 0 for weight in converted):
            raise RANSModelError("weights must be finite and non-negative")
        weight_sum = sum(converted, Decimal(0))
        if weight_sum <= 0:
            raise RANSModelError("at least one weight must be positive")

        distributable = total_frequency - len(converted) * min_frequency
        quotas = tuple(weight * distributable / weight_sum for weight in converted)
        floors = tuple(int(quota) for quota in quotas)
        frequencies = [min_frequency + floor for floor in floors]
        seats_left = total_frequency - sum(frequencies)
        order = sorted(
            range(len(converted)),
            key=lambda index: (-(quotas[index] - floors[index]), index),
        )
        for index in order[:seats_left]:
            frequencies[index] += 1

    result = tuple(frequencies)
    if sum(result) != total_frequency:  # defensive invariant
        raise AssertionError("largest-remainder apportionment did not close")
    return result


def build_gaussian_model(
    sigma: Decimal | int | str | float,
    coefficient_bound: int,
    *,
    scale_bits: int = 16,
    low_bits: int = 4,
    direct_quotients: int = 16,
    decimal_precision: int = 80,
) -> SignedRANSModel:
    """Construct a deterministic quotient model for a centered Gaussian.

    The unnormalized integer mass is ``exp(-x^2/(2*sigma^2))``.  Bucket zero
    accounts for the single value zero and both signs of positive magnitudes;
    all other buckets account for both signs.  The final bucket is ESCAPE and
    aggregates every supported quotient not encoded directly.

    This analytic table is intentionally distinct from the empirical,
    HuFu-specific tables in the public Falconws repository.
    """

    coefficient_bound = _require_plain_int(coefficient_bound, "coefficient_bound")
    if coefficient_bound < 0:
        raise RANSModelError("coefficient_bound must be non-negative")
    scale_bits = _require_plain_int(scale_bits, "scale_bits")
    decimal_precision = _require_plain_int(decimal_precision, "decimal_precision")
    low_bits = _require_plain_int(low_bits, "low_bits")
    direct_quotients = _require_plain_int(direct_quotients, "direct_quotients")
    if not 1 <= scale_bits <= 16:
        raise RANSModelError("scale_bits must lie in [1, 16]")
    if decimal_precision < 32:
        raise RANSModelError("decimal_precision must be at least 32 digits")
    if not 0 <= low_bits <= 15:
        raise RANSModelError("low_bits must lie in [0, 15]")
    if direct_quotients < 1:
        raise RANSModelError("direct_quotients must be positive")

    sigma_text = str(sigma)
    with localcontext() as context:
        context.prec = decimal_precision
        sigma_decimal = Decimal(sigma_text)
        if not sigma_decimal.is_finite() or sigma_decimal <= 0:
            raise RANSModelError("sigma must be finite and positive")
        sigma_text = format(sigma_decimal.normalize(), "f")
        denominator = Decimal(2) * sigma_decimal * sigma_decimal
        weights = [Decimal(0) for _ in range(direct_quotients + 1)]
        for magnitude in range(coefficient_bound + 1):
            quotient = magnitude >> low_bits
            symbol = quotient if quotient < direct_quotients else direct_quotients
            mass = (-(Decimal(magnitude * magnitude) / denominator)).exp()
            weights[symbol] += mass if magnitude == 0 else Decimal(2) * mass

        # ESCAPE remains part of the format even when this particular bound
        # does not reach it.  A zero analytic mass still receives the reserved
        # minimum frequency in largest_remainder_frequencies().
        frequencies = largest_remainder_frequencies(
            weights,
            scale_bits=scale_bits,
            min_frequency=1,
            decimal_precision=decimal_precision,
        )

    return SignedRANSModel(
        table=RANSFrequencyTable(frequencies, scale_bits),
        low_bits=low_bits,
        direct_quotients=direct_quotients,
        coefficient_bound=coefficient_bound,
        sigma=sigma_text,
        construction_precision=decimal_precision,
    )


def rans_encode(
    symbols: Iterable[int], table: RANSFrequencyTable
) -> bytes:
    """Encode symbols using a fixed initial state and byte-aligned rANS."""

    materialized = tuple(symbols)
    state = RANS_BYTE_L
    emitted = bytearray()
    scale = table.scale

    for symbol in reversed(materialized):
        symbol = _require_plain_int(symbol, "symbol")
        if not 0 <= symbol < table.symbol_count:
            raise RANSEncodeError(f"symbol {symbol} is outside the table")
        frequency = table.frequencies[symbol]
        start = table.cumulative[symbol]
        x_max = ((RANS_BYTE_L >> table.scale_bits) << RANS_RADIX_BITS) * frequency
        while state >= x_max:
            emitted.append(state & (RANS_RADIX - 1))
            state >>= RANS_RADIX_BITS
        state = (state // frequency) * scale + start + state % frequency

    if not RANS_BYTE_L <= state < RANS_BYTE_L * RANS_RADIX:
        raise AssertionError("encoder state left the byte-rANS normalization interval")
    return state.to_bytes(4, "little") + bytes(reversed(emitted))


def rans_decode(
    data: bytes | bytearray | memoryview,
    symbol_count: int,
    table: RANSFrequencyTable,
    *,
    require_canonical: bool = True,
) -> tuple[int, ...]:
    """Decode exactly ``symbol_count`` symbols with strict state checks."""

    symbol_count = _require_plain_int(symbol_count, "symbol_count")
    if symbol_count < 0:
        raise RANSDecodeError("symbol_count must be non-negative")
    encoded = _require_bytes_like(data)
    if len(encoded) < 4:
        raise RANSDecodeError("truncated packed rANS state")
    state = int.from_bytes(encoded[:4], "little")
    if not RANS_BYTE_L <= state < RANS_BYTE_L * RANS_RADIX:
        raise RANSDecodeError("packed rANS state is outside the valid interval")

    cursor = 4
    symbols: list[int] = []
    for _ in range(symbol_count):
        slot = state & table.mask
        symbol = bisect_right(table.cumulative, slot) - 1
        if symbol < 0 or symbol >= table.symbol_count:
            raise RANSDecodeError("rANS slot does not map to a symbol")
        start = table.cumulative[symbol]
        frequency = table.frequencies[symbol]
        if slot >= start + frequency:
            raise RANSDecodeError("rANS slot lies outside its symbol interval")

        state = frequency * (state >> table.scale_bits) + slot - start
        while state < RANS_BYTE_L:
            if cursor >= len(encoded):
                raise RANSDecodeError("truncated rANS renormalization bytes")
            state = (state << RANS_RADIX_BITS) | encoded[cursor]
            cursor += 1
        symbols.append(symbol)

    if state != RANS_BYTE_L:
        raise RANSDecodeError("decoder final state does not equal the fixed initial state")
    if cursor != len(encoded):
        raise RANSDecodeError("rANS stream contains unconsumed bytes")

    result = tuple(symbols)
    if require_canonical and rans_encode(result, table) != encoded:
        raise RANSDecodeError("rANS stream is not the canonical encoding")
    return result


def _encode_uvarint(value: int) -> bytes:
    value = _require_plain_int(value, "uvarint value")
    if value < 0:
        raise RANSEncodeError("uvarint cannot encode a negative integer")
    result = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            result.append(byte | 0x80)
        else:
            result.append(byte)
            return bytes(result)


def _decode_uvarint(
    data: bytes, cursor: int, *, max_value: int, field_name: str
) -> tuple[int, int]:
    if cursor < 0 or cursor > len(data):
        raise RANSDecodeError(f"invalid cursor for {field_name}")
    value = 0
    shift = 0
    start = cursor
    max_bytes = max(1, (max_value.bit_length() + 6) // 7)
    for _ in range(max_bytes):
        if cursor >= len(data):
            raise RANSDecodeError(f"truncated {field_name}")
        byte = data[cursor]
        cursor += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            if value > max_value:
                raise RANSDecodeError(f"{field_name} exceeds its bound")
            if data[start:cursor] != _encode_uvarint(value):
                raise RANSDecodeError(f"non-canonical {field_name}")
            return value, cursor
        shift += 7
    raise RANSDecodeError(f"overlong or out-of-range {field_name}")


def _pack_fixed_width(values: Sequence[int], width: int) -> bytes:
    if width < 1:
        raise RANSEncodeError("packed width must be positive")
    mask = (1 << width) - 1
    accumulator = 0
    bits = 0
    output = bytearray()
    for value in values:
        if not 0 <= value <= mask:
            raise RANSEncodeError("packed value does not fit its fixed width")
        accumulator |= value << bits
        bits += width
        while bits >= 8:
            output.append(accumulator & 0xFF)
            accumulator >>= 8
            bits -= 8
    if bits:
        output.append(accumulator)
    return bytes(output)


def _unpack_fixed_width(data: bytes, count: int, width: int) -> tuple[int, ...]:
    expected_length = (count * width + 7) // 8
    if len(data) != expected_length:
        raise RANSDecodeError("invalid fixed-width side-data length")
    mask = (1 << width) - 1
    accumulator = 0
    bits = 0
    cursor = 0
    values: list[int] = []
    for _ in range(count):
        while bits < width:
            accumulator |= data[cursor] << bits
            cursor += 1
            bits += 8
        values.append(accumulator & mask)
        accumulator >>= width
        bits -= width
    if accumulator != 0:
        raise RANSDecodeError("non-zero unused bits in fixed-width side data")
    return tuple(values)


def _check_norm(values: Sequence[int], norm_bound: int | None, error_type: type[RANSError]) -> None:
    if norm_bound is None:
        return
    norm_bound = _require_plain_int(norm_bound, "norm_bound")
    if norm_bound < 0:
        raise error_type("norm_bound must be non-negative")
    squared_norm = sum(value * value for value in values)
    if squared_norm > norm_bound * norm_bound:
        raise error_type(
            f"coefficient vector has squared norm {squared_norm}, "
            f"exceeding {norm_bound * norm_bound}"
        )


def encode_raw(
    values: Sequence[int],
    model: SignedRANSModel,
    *,
    norm_bound: int | None = None,
) -> bytes:
    """Encode a signed coefficient vector into a canonical variable stream.

    Layout::

        version || canonical-uvarint(rans-length) || rans-stream
                || packed(low-bits, sign) || escape-uvarints

    The coefficient count and model are external parameter-set information.
    """

    coefficients = tuple(_require_plain_int(value, "coefficient") for value in values)
    _check_norm(coefficients, norm_bound, RANSEncodeError)

    symbols: list[int] = []
    side_units: list[int] = []
    escapes = bytearray()
    for value in coefficients:
        magnitude = abs(value)
        if magnitude > model.coefficient_bound:
            raise RANSEncodeError(
                f"coefficient magnitude {magnitude} exceeds model bound "
                f"{model.coefficient_bound}"
            )
        quotient = magnitude >> model.low_bits
        low = magnitude & model.low_mask
        sign = int(value < 0)
        side_units.append(low | (sign << model.low_bits))
        if quotient < model.direct_quotients:
            symbols.append(quotient)
        else:
            symbols.append(model.escape_symbol)
            escapes.extend(_encode_uvarint(quotient - model.direct_quotients))

    ans_stream = rans_encode(symbols, model.table)
    side_stream = _pack_fixed_width(side_units, model.unit_bits)
    return (
        bytes((RAW_CODEC_VERSION,))
        + _encode_uvarint(len(ans_stream))
        + ans_stream
        + side_stream
        + bytes(escapes)
    )


def decode_raw(
    data: bytes | bytearray | memoryview,
    model: SignedRANSModel,
    expected_count: int,
    *,
    norm_bound: int | None = None,
    require_canonical: bool = True,
) -> tuple[int, ...]:
    """Strictly decode a raw signed-coefficient stream."""

    expected_count = _require_plain_int(expected_count, "expected_count")
    if expected_count < 0:
        raise RANSDecodeError("expected_count must be non-negative")
    encoded = _require_bytes_like(data)
    if not encoded:
        raise RANSDecodeError("empty raw rANS stream")
    if encoded[0] != RAW_CODEC_VERSION:
        raise RANSDecodeError("unsupported raw rANS codec version")

    side_length = (expected_count * model.unit_bits + 7) // 8
    ans_length, cursor = _decode_uvarint(
        encoded, 1, max_value=max(0, len(encoded) - 1), field_name="rANS length"
    )
    if ans_length < 4:
        raise RANSDecodeError("rANS stream is too short to contain a state")
    ans_end = cursor + ans_length
    side_end = ans_end + side_length
    if ans_end > len(encoded) or side_end > len(encoded):
        raise RANSDecodeError("raw stream is truncated")

    symbols = rans_decode(
        encoded[cursor:ans_end],
        expected_count,
        model.table,
        require_canonical=require_canonical,
    )
    side_units = _unpack_fixed_width(
        encoded[ans_end:side_end], expected_count, model.unit_bits
    )

    escape_cursor = side_end
    max_quotient = model.coefficient_bound >> model.low_bits
    max_residual = max(0, max_quotient - model.direct_quotients)
    coefficients: list[int] = []
    for symbol, unit in zip(symbols, side_units, strict=True):
        low = unit & model.low_mask
        sign = unit >> model.low_bits
        if symbol == model.escape_symbol:
            residual, escape_cursor = _decode_uvarint(
                encoded,
                escape_cursor,
                max_value=max_residual,
                field_name="escape residual",
            )
            quotient = model.direct_quotients + residual
        else:
            quotient = symbol
        magnitude = (quotient << model.low_bits) | low
        if magnitude > model.coefficient_bound:
            raise RANSDecodeError("decoded coefficient exceeds the model bound")
        if sign and magnitude == 0:
            raise RANSDecodeError("negative zero is not a canonical coefficient")
        coefficients.append(-magnitude if sign else magnitude)

    if escape_cursor != len(encoded):
        raise RANSDecodeError("raw stream contains trailing or unused escape data")
    result = tuple(coefficients)
    _check_norm(result, norm_bound, RANSDecodeError)
    if require_canonical and encode_raw(result, model, norm_bound=norm_bound) != encoded:
        raise RANSDecodeError("raw stream is not the canonical encoding")
    return result


def encode_padded(
    values: Sequence[int],
    model: SignedRANSModel,
    padded_length: int,
    *,
    norm_bound: int | None = None,
) -> bytes:
    """Encode into ``00 ... 00 || 80 || raw`` of exactly ``padded_length``."""

    padded_length = _require_plain_int(padded_length, "padded_length")
    if padded_length < 1:
        raise RANSEncodeError("padded_length must be positive")
    raw = encode_raw(values, model, norm_bound=norm_bound)
    padding_length = padded_length - len(raw) - 1
    if padding_length < 0:
        raise RANSEncodeError(
            f"raw stream needs {len(raw) + 1} bytes including the marker, "
            f"but padded_length is {padded_length}"
        )
    return bytes(padding_length) + bytes((PAD_MARKER,)) + raw


def decode_padded(
    data: bytes | bytearray | memoryview,
    model: SignedRANSModel,
    expected_count: int,
    padded_length: int,
    *,
    norm_bound: int | None = None,
    require_canonical: bool = True,
) -> tuple[int, ...]:
    """Decode a fixed-length padded stream; the expected length is external."""

    padded_length = _require_plain_int(padded_length, "padded_length")
    if padded_length < 1:
        raise RANSDecodeError("padded_length must be positive")
    encoded = _require_exact_length_bytes_like(data, padded_length)
    cursor = 0
    while cursor < len(encoded) and encoded[cursor] == 0:
        cursor += 1
    if cursor >= len(encoded) or encoded[cursor] != PAD_MARKER:
        raise RANSDecodeError("invalid zero padding or missing 0x80 marker")
    raw = encoded[cursor + 1 :]
    if not raw:
        raise RANSDecodeError("padding marker is not followed by a raw stream")
    result = decode_raw(
        raw,
        model,
        expected_count,
        norm_bound=norm_bound,
        require_canonical=require_canonical,
    )
    if require_canonical:
        canonical = encode_padded(
            result, model, padded_length, norm_bound=norm_bound
        )
        if canonical != encoded:
            raise RANSDecodeError("padded stream is not the canonical encoding")
    return result


def _uvarint_length(value: int) -> int:
    return len(_encode_uvarint(value))


def _escape_payload_bound(
    dimension: int, norm_bound: int, model: SignedRANSModel
) -> int:
    """Layer-cake upper bound for all escape ULEB128 payload bytes.

    For every byte position j, a j-byte residual has a known minimum
    coefficient magnitude.  At most floor(beta^2 / magnitude^2) coordinates
    can reach it.  Summing those nested counts is conservative but certified.
    """

    if dimension == 0 or norm_bound == 0:
        return 0
    base = 1 << model.low_bits
    budget = norm_bound * norm_bound
    result = 0
    byte_index = 1
    while True:
        minimum_residual = 0 if byte_index == 1 else 1 << (7 * (byte_index - 1))
        minimum_magnitude = (model.direct_quotients + minimum_residual) * base
        if minimum_magnitude > min(norm_bound, model.coefficient_bound):
            break
        result += min(dimension, budget // (minimum_magnitude * minimum_magnitude))
        byte_index += 1
    return result


def _expected_escape_payload_bytes(model: SignedRANSModel) -> Decimal:
    """Expected ULEB128 escape bytes for one truncated Gaussian draw.

    This uses the same centered, coefficient-bounded Gaussian model as
    :func:`build_gaussian_model`.  It is an estimate only; the separate
    layer-cake value remains the certified worst-case bound.
    """

    with localcontext() as context:
        context.prec = model.construction_precision
        sigma = Decimal(model.sigma)
        denominator = Decimal(2) * sigma * sigma
        normalizer = Decimal(1)  # magnitude zero has one sign
        weighted_bytes = Decimal(0)
        for magnitude in range(1, model.coefficient_bound + 1):
            signed_mass = Decimal(2) * (
                -(Decimal(magnitude * magnitude) / denominator)
            ).exp()
            normalizer += signed_mass
            quotient = magnitude >> model.low_bits
            if quotient >= model.direct_quotients:
                residual = quotient - model.direct_quotients
                weighted_bytes += signed_mass * len(_encode_uvarint(residual))
        return +(weighted_bytes / normalizer)


def analyze_padded_length(
    dimension: int,
    norm_bound: int,
    model: SignedRANSModel,
) -> PaddedLengthAnalysis:
    """Return a deterministic certified bound and a model-based mean estimate.

    The certified bound assumes every rANS symbol may emit
    ``ceil(scale_bits/8)`` renormalization bytes, plus the four-byte state.
    It uses the Euclidean norm budget to bound escape payload bytes.  This is
    safe for formal encoding, but can be substantially larger than real
    streams.  The final size-competitive wire length therefore remains an
    explicit parameter-policy decision (``wire_policy_resolved=False``).
    """

    dimension = _require_plain_int(dimension, "dimension")
    norm_bound = _require_plain_int(norm_bound, "norm_bound")
    if dimension < 0:
        raise RANSModelError("dimension must be non-negative")
    if norm_bound < 0:
        raise RANSModelError("norm_bound must be non-negative")
    if norm_bound > model.coefficient_bound:
        raise RANSModelError(
            "model coefficient_bound must cover every norm-permitted coefficient"
        )

    max_renorm_per_symbol = ceil(model.table.scale_bits / RANS_RADIX_BITS)
    ans_upper = 4 + dimension * max_renorm_per_symbol
    side_bytes = (dimension * model.unit_bits + 7) // 8
    escape_upper = _escape_payload_bound(dimension, norm_bound, model)
    raw_upper = (
        1  # raw codec version
        + _uvarint_length(ans_upper)
        + ans_upper
        + side_bytes
        + escape_upper
    )
    certified_upper = 1 + raw_upper  # fixed-padding marker

    scale = model.table.scale
    entropy_per_symbol = sum(
        (frequency / scale) * log2(scale / frequency)
        for frequency in model.table.frequencies
    )
    estimated_ans = 4 + ceil(dimension * entropy_per_symbol / 8)
    expected_escape_per_coefficient = _expected_escape_payload_bytes(model)
    with localcontext() as context:
        context.prec = model.construction_precision
        estimated_escape = int(
            (Decimal(dimension) * expected_escape_per_coefficient).to_integral_value(
                rounding=ROUND_CEILING
            )
        )
    estimated_raw = (
        1
        + _uvarint_length(estimated_ans)
        + estimated_ans
        + side_bytes
        + estimated_escape
    )
    estimated_padded = 1 + estimated_raw

    return PaddedLengthAnalysis(
        length_algorithm=PADDED_LENGTH_ALGORITHM,
        model_fingerprint=model.fingerprint,
        frequency_table_fingerprint=model.table.fingerprint,
        dimension=dimension,
        norm_bound=norm_bound,
        certified_upper_bound=certified_upper,
        estimated_mean_length=estimated_padded,
        rans_stream_upper_bound=ans_upper,
        side_bits_bytes=side_bytes,
        escape_payload_upper_bound=escape_upper,
        escape_payload_estimated_bytes=estimated_escape,
        certified=True,
        size_tight=False,
        wire_policy_resolved=False,
        note=(
            "The certified length is safe but conservative.  A final fixed "
            "wire length has not been selected; choosing a shorter length "
            "would require a proof or an explicit encoding-rejection analysis."
        ),
    )


__all__ = [
    "PAD_MARKER",
    "PADDED_LENGTH_ALGORITHM",
    "PaddedLengthAnalysis",
    "RANSDecodeError",
    "RANSEncodeError",
    "RANSError",
    "RANSFrequencyTable",
    "RANSModelError",
    "SignedRANSModel",
    "analyze_padded_length",
    "build_gaussian_model",
    "decode_padded",
    "decode_raw",
    "encode_padded",
    "encode_raw",
    "largest_remainder_frequencies",
    "rans_decode",
    "rans_encode",
]
