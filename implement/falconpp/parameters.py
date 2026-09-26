"""Registered parameter sets for the Falcon++ reference implementation.

The four original Section 6 records retain the values printed in the paper.
Four additional candidates retain their authoritative decimal core inputs,
derive their norm thresholds, and use separately reoptimized correction
constants.  Derived quantities are exposed as properties so that experiments
do not keep independent, potentially inconsistent copies of the same formula.

This is research code.  The parameter identifiers below are local identifiers
for reproducible experiments, not standardized Falcon identifiers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation, ROUND_FLOOR, localcontext
from fractions import Fraction
from math import isfinite, isqrt, log2, pi, sqrt
from types import MappingProxyType
from typing import Final, Iterator


HASH_QUERY_CAP: Final[int] = 1 << 96
SIGN_QUERY_CAP: Final[int] = 1 << 64
SAMPLER_LOSS_BITS: Final[int] = 2
SALT_BITS: Final[int] = 325
SALT_BYTES: Final[int] = 41
SALT_UNUSED_HIGH_BITS: Final[int] = SALT_BYTES * 8 - SALT_BITS
MOMENT_ORDER_MIN: Final[int] = 2
MOMENT_ORDER_MAX: Final[int] = 1024


def _exact_decimal_parameter(value: object, name: str) -> Decimal:
    """Retain the numeric source supplied to a registered parameter field.

    Text and :class:`Decimal` inputs remain exact decimal values.  A caller
    may still use the convenient float constructor API; in that case the
    stored source honestly represents that already-rounded binary64 value.
    """

    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and positive")
    try:
        if isinstance(value, Decimal):
            result = value
        elif isinstance(value, float):
            result = Decimal.from_float(value)
        else:
            result = Decimal(value)  # type: ignore[arg-type]
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and positive") from exc
    if not result.is_finite() or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _derived_signature_beta(n: int, sigma_sig: str) -> int:
    """Compute ``ceil(1.1*sigma_sig*sqrt(2*n))`` from its exact source.

    Squaring reduces the ceiling decision to integer arithmetic.  This is
    equivalent to evaluating the formula with the source Decimal, but also
    handles an exact integer boundary without depending on ambient precision.
    """

    width = Fraction(_exact_decimal_parameter(sigma_sig, "sigma_sig"))
    squared_bound = Fraction(121, 100) * width * width * (2 * n)
    lower = isqrt(squared_bound.numerator // squared_bound.denominator)
    if lower * lower * squared_bound.denominator == squared_bound.numerator:
        return lower
    return lower + 1


@dataclass(frozen=True, slots=True)
class FalconPPParameters:
    """One immutable Falcon++ parameter record.

    ``sigma_sig`` is the usual (probabilistic) standard deviation.  The paper
    also uses the Gaussian parameter ``s = sqrt(2*pi) * sigma_sig``; it is
    available through :attr:`s`.
    """

    param_id: int
    name: str
    category: str
    target_security: int
    n: int
    q: int
    gamma: float
    sigma_sig: float
    beta: int
    widehat_gamma: float
    moment_order: int
    # These fields are part of equality/hash semantics: two records whose
    # decimal sources round to the same float must not share cached formal
    # signing artefacts.  ``repr=False`` keeps the familiar compact display.
    gamma_decimal: Decimal = field(init=False, repr=False)
    sigma_sig_decimal: Decimal = field(init=False, repr=False)
    widehat_gamma_decimal: Decimal = field(init=False, repr=False)

    def __post_init__(self) -> None:
        gamma_decimal = _exact_decimal_parameter(self.gamma, "gamma")
        sigma_sig_decimal = _exact_decimal_parameter(self.sigma_sig, "sigma_sig")
        widehat_gamma_decimal = _exact_decimal_parameter(
            self.widehat_gamma, "widehat_gamma"
        )
        object.__setattr__(self, "gamma_decimal", gamma_decimal)
        object.__setattr__(self, "sigma_sig_decimal", sigma_sig_decimal)
        object.__setattr__(self, "widehat_gamma_decimal", widehat_gamma_decimal)
        # Preserve the original ergonomic API for diagnostics and callers
        # that deliberately select the binary64 path.
        object.__setattr__(self, "gamma", float(gamma_decimal))
        object.__setattr__(self, "sigma_sig", float(sigma_sig_decimal))
        object.__setattr__(self, "widehat_gamma", float(widehat_gamma_decimal))

        if isinstance(self.param_id, bool) or not 0 < self.param_id < 256:
            raise ValueError("param_id must be an integer in 1..255")
        if not self.name:
            raise ValueError("parameter-set name must not be empty")
        if self.n <= 0 or self.n & (self.n - 1):
            raise ValueError("n must be a positive power of two")
        if not 2 < self.q < (1 << 16) or self.q % 2 == 0:
            raise ValueError("q must be an odd modulus below 2^16")
        if not isfinite(self.gamma) or self.gamma <= 0:
            raise ValueError("gamma must be finite and positive")
        if not isfinite(self.sigma_sig) or self.sigma_sig <= 0:
            raise ValueError("sigma_sig must be finite and positive")
        if self.beta <= 0:
            raise ValueError("beta must be positive")
        if not isfinite(self.widehat_gamma) or self.widehat_gamma <= 0:
            raise ValueError("widehat_gamma must be finite and positive")
        if not MOMENT_ORDER_MIN <= self.moment_order <= MOMENT_ORDER_MAX:
            raise ValueError("moment_order is outside the registered search range")

    @property
    def logn(self) -> int:
        """Return ``log2(n)`` (exact because ``n`` is a power of two)."""

        return self.n.bit_length() - 1

    @property
    def sigma_fg(self) -> float:
        """Key-generation coefficient width ``gamma*sqrt(q/(2*n))``."""

        return self.gamma * sqrt(self.q / (2 * self.n))

    @property
    def sigma_fg_decimal(self) -> Decimal:
        """KeyGen width evaluated in the active :mod:`decimal` context."""

        return self.gamma_decimal * (
            Decimal(self.q) / Decimal(2 * self.n)
        ).sqrt()

    @property
    def s(self) -> float:
        """Gaussian parameter ``sqrt(2*pi)*sigma_sig`` used in the paper."""

        return sqrt(2 * pi) * self.sigma_sig

    @property
    def gaussian_s(self) -> float:
        """Descriptive alias for :attr:`s`."""

        return self.s

    @property
    def keygen_norm_cap(self) -> int:
        """Integer cap ``floor(gamma^2*q)`` for ``||(f,g)||_2^2``."""

        gamma = self.gamma_decimal
        # The expression is rational for every finite Decimal input.  Select
        # enough local precision to make the multiplication exact before the
        # floor; in particular, never round a near-integer profile through the
        # convenience binary64 ``gamma`` field.
        exact_digits = 2 * len(gamma.as_tuple().digits) + len(str(self.q)) + 4
        with localcontext() as context:
            context.prec = max(32, exact_digits)
            value = gamma * gamma * Decimal(self.q)
            return int(value.to_integral_value(rounding=ROUND_FLOOR))

    @property
    def signature_norm_bound(self) -> int:
        """Squared verification bound ``beta^2``."""

        return self.beta * self.beta

    @property
    def public_key_coefficient_bits(self) -> int:
        """Fixed coefficient width for a canonical public-key payload."""

        return (self.q - 1).bit_length()

    @property
    def public_key_payload_bytes(self) -> int:
        """Byte length of ``n`` fixed-width public-key coefficients."""

        bits = self.n * self.public_key_coefficient_bits
        return (bits + 7) // 8

    @property
    def gamma_hat(self) -> float:
        """Short alias for the fixed analytical correction multiplier."""

        return self.widehat_gamma

    @property
    def gamma_hat_decimal(self) -> Decimal:
        """Exact decimal source for the analytical correction multiplier."""

        return self.widehat_gamma_decimal

    @property
    def correction_multiplier(self) -> float:
        """Descriptive alias for :attr:`widehat_gamma`."""

        return self.widehat_gamma

    @property
    def correction_multiplier_decimal(self) -> Decimal:
        """Exact alias for :attr:`widehat_gamma_decimal`."""

        return self.widehat_gamma_decimal

    @property
    def Q_H(self) -> int:  # noqa: N802 - follows the paper's notation
        return HASH_QUERY_CAP

    @property
    def Q_s(self) -> int:  # noqa: N802 - follows the paper's notation
        return SIGN_QUERY_CAP

    @property
    def sampler_loss_bits(self) -> int:
        return SAMPLER_LOSS_BITS

    @property
    def salt_bits(self) -> int:
        return SALT_BITS

    @property
    def salt_bytes(self) -> int:
        return SALT_BYTES


FALCONPP_512_117: Final = FalconPPParameters(
    param_id=0x01,
    name="falconpp-512-953-gamma117",
    category="I",
    target_security=128,
    n=512,
    q=953,
    gamma="1.17",  # type: ignore[arg-type] - retain the paper's decimal source
    sigma_sig="19.1399",  # type: ignore[arg-type]
    beta=674,
    widehat_gamma="4.533254",  # type: ignore[arg-type]
    moment_order=142,
)

FALCONPP_512_125: Final = FalconPPParameters(
    param_id=0x02,
    name="falconpp-512-953-gamma125",
    category="I",
    target_security=128,
    n=512,
    q=953,
    gamma="1.25",  # type: ignore[arg-type]
    sigma_sig="20.9921",  # type: ignore[arg-type]
    beta=739,
    widehat_gamma="1.774456",  # type: ignore[arg-type]
    moment_order=272,
)

FALCONPP_1024_117: Final = FalconPPParameters(
    param_id=0x03,
    name="falconpp-1024-1949-gamma117",
    category="V",
    target_security=256,
    n=1024,
    q=1949,
    gamma="1.17",  # type: ignore[arg-type]
    sigma_sig="28.2544",  # type: ignore[arg-type]
    beta=1407,
    widehat_gamma="9.067426",  # type: ignore[arg-type]
    moment_order=145,
)

FALCONPP_1024_125: Final = FalconPPParameters(
    param_id=0x04,
    name="falconpp-1024-1949-gamma125",
    category="V",
    target_security=256,
    n=1024,
    q=1949,
    gamma="1.25",  # type: ignore[arg-type]
    sigma_sig="30.0203",  # type: ignore[arg-type]
    beta=1495,
    widehat_gamma="3.714950",  # type: ignore[arg-type]
    moment_order=184,
)


ORIGINAL_PARAMETER_SETS: Final[tuple[FalconPPParameters, ...]] = (
    FALCONPP_512_117,
    FALCONPP_512_125,
    FALCONPP_1024_117,
    FALCONPP_1024_125,
)

# Frozen outputs of experiments.candidate_analysis under
# determinant_normalized_inferred_missing_spec, with all orders 2..1024
# screened and the shortlisted/local neighbours evaluated at 100 dps.
# Winners and immediate neighbours were checked again at 120 dps.  Each
# multiplier is rounded DOWN to 30 significant decimal digits and the loss
# ledger is checked using that exact text.  These remain heuristic analytical
# estimates, not interval-certified global optima or generated-key fits.
# The attachment's candidate multipliers must not replace these constants.
FALCONPP_I_1245: Final = FalconPPParameters(
    param_id=0x05,
    name="Falcon++-I-1245",
    category="I",
    target_security=128,
    n=512,
    q=509,
    gamma="1.245",  # type: ignore[arg-type]
    sigma_sig="14.374684087488",  # type: ignore[arg-type]
    beta=_derived_signature_beta(512, "14.374684087488"),
    widehat_gamma="3.76212769429086303270862578382",  # type: ignore[arg-type]
    moment_order=130,
)

FALCONPP_I_1330: Final = FalconPPParameters(
    param_id=0x06,
    name="Falcon++-I-1330",
    category="I",
    target_security=128,
    n=512,
    q=953,
    gamma="1.330",  # type: ignore[arg-type]
    sigma_sig="20.976683230802",  # type: ignore[arg-type]
    beta=_derived_signature_beta(512, "20.976683230802"),
    widehat_gamma="3.01848774834217235401415845208",  # type: ignore[arg-type]
    moment_order=131,
)

FALCONPP_V_1205: Final = FalconPPParameters(
    param_id=0x07,
    name="Falcon++-V-1205",
    category="V",
    target_security=256,
    n=1024,
    q=1021,
    gamma="1.205",  # type: ignore[arg-type]
    sigma_sig="20.446387782892",  # type: ignore[arg-type]
    beta=_derived_signature_beta(1024, "20.446387782892"),
    widehat_gamma="10.6021853181894045935633196116",  # type: ignore[arg-type]
    moment_order=125,
)

FALCONPP_V_1330: Final = FalconPPParameters(
    param_id=0x08,
    name="Falcon++-V-1330",
    category="V",
    target_security=256,
    n=1024,
    q=1949,
    gamma="1.330",  # type: ignore[arg-type]
    sigma_sig="30.951242286169",  # type: ignore[arg-type]
    beta=_derived_signature_beta(1024, "30.951242286169"),
    widehat_gamma="5.65407764780565151144704534151",  # type: ignore[arg-type]
    moment_order=126,
)

ADDITIONAL_PARAMETER_SETS: Final[tuple[FalconPPParameters, ...]] = (
    FALCONPP_I_1245,
    FALCONPP_I_1330,
    FALCONPP_V_1205,
    FALCONPP_V_1330,
)

PARAMETER_SETS: Final[tuple[FalconPPParameters, ...]] = (
    ORIGINAL_PARAMETER_SETS + ADDITIONAL_PARAMETER_SETS
)

PARAMETERS_BY_ID = MappingProxyType({p.param_id: p for p in PARAMETER_SETS})
PARAMETERS_BY_NAME = MappingProxyType({p.name: p for p in PARAMETER_SETS})

_ALIASES = MappingProxyType(
    {
        "i-117": FALCONPP_512_117,
        "i-125": FALCONPP_512_125,
        "v-117": FALCONPP_1024_117,
        "v-125": FALCONPP_1024_125,
        "512-117": FALCONPP_512_117,
        "512-125": FALCONPP_512_125,
        "1024-117": FALCONPP_1024_117,
        "1024-125": FALCONPP_1024_125,
        "i-1245": FALCONPP_I_1245,
        "i-1330": FALCONPP_I_1330,
        "v-1205": FALCONPP_V_1205,
        "v-1330": FALCONPP_V_1330,
        "512-1245": FALCONPP_I_1245,
        "512-1330": FALCONPP_I_1330,
        "1024-1205": FALCONPP_V_1205,
        "1024-1330": FALCONPP_V_1330,
        "falconpp-i-1245": FALCONPP_I_1245,
        "falconpp-i-1330": FALCONPP_I_1330,
        "falconpp-v-1205": FALCONPP_V_1205,
        "falconpp-v-1330": FALCONPP_V_1330,
    }
)

# A conventional mapping name used by experiment scripts.
PARAMETERS = PARAMETERS_BY_NAME

# Compatibility alias: some modules use the shorter class name.
ParameterSet = FalconPPParameters


def get_parameters(
    identifier: int | str | FalconPPParameters,
) -> FalconPPParameters:
    """Resolve a registered parameter set by object, byte ID, or name.

    String lookup is case-insensitive and accepts short aliases such as
    ``I-117`` for an original set and ``I-1245`` for an additional candidate.
    """

    if isinstance(identifier, FalconPPParameters):
        return identifier
    if isinstance(identifier, bool):
        raise KeyError(f"unknown Falcon++ parameter set: {identifier!r}")
    if isinstance(identifier, int):
        try:
            return PARAMETERS_BY_ID[identifier]
        except KeyError as exc:
            raise KeyError(f"unknown Falcon++ parameter id: {identifier!r}") from exc
    if isinstance(identifier, str):
        key = identifier.strip().lower()
        by_name = {name.lower(): value for name, value in PARAMETERS_BY_NAME.items()}
        if key in by_name:
            return by_name[key]
        if key in _ALIASES:
            return _ALIASES[key]
        raise KeyError(f"unknown Falcon++ parameter name: {identifier!r}")
    raise TypeError("parameter identifier must be an id, name, or parameter object")


def iter_parameter_sets() -> Iterator[FalconPPParameters]:
    """Iterate over registered sets in stable wire-ID order."""

    return iter(PARAMETER_SETS)


def coefficient_entropy_bits(sigma: float) -> float:
    """Continuous-Gaussian entropy approximation used by size diagnostics."""

    if not isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    return 0.5 * log2(2 * pi * 2.718281828459045 * sigma * sigma)


__all__ = [
    "FalconPPParameters",
    "ParameterSet",
    "FALCONPP_512_117",
    "FALCONPP_512_125",
    "FALCONPP_1024_117",
    "FALCONPP_1024_125",
    "FALCONPP_I_1245",
    "FALCONPP_I_1330",
    "FALCONPP_V_1205",
    "FALCONPP_V_1330",
    "ORIGINAL_PARAMETER_SETS",
    "ADDITIONAL_PARAMETER_SETS",
    "PARAMETER_SETS",
    "PARAMETERS",
    "PARAMETERS_BY_ID",
    "PARAMETERS_BY_NAME",
    "HASH_QUERY_CAP",
    "SIGN_QUERY_CAP",
    "SAMPLER_LOSS_BITS",
    "SALT_BITS",
    "SALT_BYTES",
    "SALT_UNUSED_HIGH_BITS",
    "MOMENT_ORDER_MIN",
    "MOMENT_ORDER_MAX",
    "get_parameters",
    "iter_parameter_sets",
    "coefficient_entropy_bits",
]
