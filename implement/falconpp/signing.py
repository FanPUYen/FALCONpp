"""Falcon++ fresh-syndrome signing.

Each raw trial creates a new 325-bit salt, hashes ``h || salt || message``,
runs Falcon's recursive FFT/Klein sampler, evaluates the exact one-dimensional
KGPV correction in high precision, and only then applies the signature norm
test.  A rejection at either test discards the complete trial.

The provisional formal codec uses the conservative norm-certified rANS bound
as a fixed payload length so canonical wire tests can run before the final
size policy is frozen.  The variable-length raw codec remains available for
experiments.  Neither mode includes a parameter identifier in the signature.
Formal signing therefore requires an explicit
``allow_provisional_wire=True`` research acknowledgement; raw mode does not.

This module is a transparent research implementation, not constant-time
production cryptography.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Protocol, Sequence

from .correction import CorrectionDecision, evaluate_correction
from .ff_sampling import FFSampleTraceEntry, ff_sample, iter_leaves
from .fft import fft, fft_high_precision, ifft, ifft_high_precision, mul_fft
from .gaussian import DiscreteGaussianSampler
from .hash_to_point import hash_to_point
from .keys import SecretKey, Signature
from .parameters import FalconPPParameters, get_parameters
from .polynomial import add, negacyclic_mul, sub
from .randomness import generate_salt, make_prng
from .rans import (
    PaddedLengthAnalysis,
    RANSEncodeError,
    SignedRANSModel,
    analyze_padded_length,
    build_gaussian_model,
    decode_padded,
    decode_raw,
    encode_padded,
    encode_raw,
)


DEFAULT_MAX_SIGNING_TRIALS = 1_000_000
DEFAULT_CORRECTION_DPS = 100
# This is only the first chunk of the adaptive exact Bernoulli comparison; it
# is not a probability-quantisation or statistical-error parameter.  Starting
# at 64 bits avoids doing 256-bit theta-mass work for the overwhelmingly common
# case while the comparison automatically requests more bits when necessary.
DEFAULT_CORRECTION_BITS = 64


class SigningError(RuntimeError):
    """Raised when signing exhausts its explicit public trial limit."""

    def __init__(self, message: str, statistics: "SigningStatistics") -> None:
        super().__init__(message)
        self.statistics = statistics


class SigningEncodingError(RuntimeError):
    """A norm-valid signer output failed the supposedly lossless codec."""


class OneDimensionalSampler(Protocol):
    def sample(self, center: float, sigma: float, rng: Any = None) -> int:
        """Draw one conditional integer."""


@dataclass(frozen=True, slots=True)
class SignatureCodec:
    """Frozen rANS policy for one Falcon++ parameter set."""

    parameters: FalconPPParameters
    model: SignedRANSModel
    formal: bool = True
    padded_length: int | None = None
    length_analysis: PaddedLengthAnalysis | None = None

    def __post_init__(self) -> None:
        params = get_parameters(self.parameters)
        if self.model.coefficient_bound < params.beta:
            raise ValueError("rANS model does not cover every norm-permitted coefficient")
        expected_analysis = analyze_padded_length(params.n, params.beta, self.model)
        if (
            self.length_analysis is not None
            and self.length_analysis != expected_analysis
        ):
            raise ValueError(
                "rANS padded-length analysis does not match the current model, "
                "model fingerprint, or length-analysis algorithm"
            )
        # Never trust a caller-supplied certificate merely because its
        # `certified` flag is true.  The canonical object is deterministically
        # recomputed from the current model and parameter-set norm bound.
        analysis = expected_analysis
        if not analysis.certified:
            raise ValueError("formal signature sizing requires a certified bound")
        padded_length = self.padded_length
        if self.formal:
            if padded_length is None:
                padded_length = analysis.certified_upper_bound
            if isinstance(padded_length, bool) or not isinstance(padded_length, int):
                raise TypeError("padded_length must be an integer")
            if padded_length < analysis.certified_upper_bound:
                raise ValueError(
                    "formal padded_length is below the certified norm-constrained "
                    f"upper bound {analysis.certified_upper_bound}"
                )
        elif padded_length is not None:
            raise ValueError("raw signature mode does not use padded_length")

        object.__setattr__(self, "parameters", params)
        object.__setattr__(self, "padded_length", padded_length)
        object.__setattr__(self, "length_analysis", analysis)

    @property
    def payload_length(self) -> int | None:
        """Fixed encoded-s2 length, or ``None`` for raw mode."""

        return self.padded_length if self.formal else None

    @property
    def wire_policy_resolved(self) -> bool:
        """Whether the manuscript's final wire-size policy has been frozen."""

        return bool(
            self.formal
            and self.length_analysis is not None
            and self.length_analysis.wire_policy_resolved
        )

    @property
    def has_safe_fixed_length(self) -> bool:
        """Whether every norm-valid vector fits this experimental fixed length."""

        return bool(
            self.formal
            and self.padded_length is not None
            and self.length_analysis is not None
            and self.length_analysis.certified
            and self.padded_length >= self.length_analysis.certified_upper_bound
        )

    @property
    def fixed_length_policy(self) -> str:
        if not self.formal:
            return "raw-variable-length-experiment"
        assert self.length_analysis is not None
        if self.padded_length == self.length_analysis.certified_upper_bound:
            return "provisional-certified-norm-constrained-upper-bound"
        return "provisional-explicit-at-least-certified-length"

    def encode(self, coefficients: Sequence[int]) -> bytes:
        if len(coefficients) != self.parameters.n:
            raise RANSEncodeError(
                f"signature polynomial has {len(coefficients)} coefficients; "
                f"expected {self.parameters.n}"
            )
        if self.formal:
            assert self.padded_length is not None
            return encode_padded(
                coefficients,
                self.model,
                self.padded_length,
                norm_bound=self.parameters.beta,
            )
        return encode_raw(
            coefficients, self.model, norm_bound=self.parameters.beta
        )

    def decode(self, payload: bytes | bytearray | memoryview) -> tuple[int, ...]:
        """Strictly decode a canonical signature polynomial."""

        if self.formal:
            assert self.padded_length is not None
            return decode_padded(
                payload,
                self.model,
                self.parameters.n,
                self.padded_length,
                norm_bound=self.parameters.beta,
                require_canonical=True,
            )
        return decode_raw(
            payload,
            self.model,
            self.parameters.n,
            norm_bound=self.parameters.beta,
            require_canonical=True,
        )


@lru_cache(maxsize=16)
def _default_codec(
    parameters: FalconPPParameters,
    formal: bool,
    padded_length: int | None,
) -> SignatureCodec:
    model = build_gaussian_model(parameters.sigma_sig_decimal, parameters.beta)
    return SignatureCodec(parameters, model, formal, padded_length)


def make_signature_codec(
    parameters: int | str | FalconPPParameters,
    *,
    formal: bool = True,
    padded_length: int | None = None,
) -> SignatureCodec:
    """Return a cached canonical rANS codec for a parameter set."""

    params = get_parameters(parameters)
    return _default_codec(params, formal, padded_length)


def _require_signature_wire_acknowledgement(
    codec: SignatureCodec,
    allow_provisional_wire: bool,
) -> None:
    """Enforce the unresolved fixed-wire policy at every public I/O path."""

    if not isinstance(allow_provisional_wire, bool):
        raise TypeError("allow_provisional_wire must be a bool")
    if (
        codec.formal
        and not codec.wire_policy_resolved
        and not allow_provisional_wire
    ):
        raise ValueError(
            "the Falcon++ signature wire policy is provisional; pass "
            "allow_provisional_wire=True only for research experiments"
        )


@dataclass(frozen=True, slots=True)
class PreimageSample:
    """One raw Klein proposal before correction and norm rejection."""

    s1: tuple[int, ...]
    s2: tuple[int, ...]
    z0: tuple[int, ...]
    z1: tuple[int, ...]
    trace: tuple[FFSampleTraceEntry, ...]

    @property
    def squared_norm(self) -> int:
        return sum(value * value for value in (*self.s1, *self.s2))


@dataclass(frozen=True, slots=True)
class SigningTrialRecord:
    """Compact, non-secret diagnostic record for one discarded/accepted trial."""

    trial: int
    correction_accepted: bool
    norm_accepted: bool | None
    encoding_accepted: bool | None
    squared_norm: int | None
    log_delta: str | None
    correction_probability: str | None
    correction_backend: str = "primal"
    diagnostic_level: str = "full"


@dataclass(frozen=True, slots=True)
class SigningStatistics:
    """Trial history retained for reproducibility and rejection accounting."""

    trials: tuple[SigningTrialRecord, ...]
    correction_metrics: dict[str, int | float] = field(default_factory=dict)

    @property
    def total_trials(self) -> int:
        return len(self.trials)

    @property
    def correction_rejections(self) -> int:
        return sum(not trial.correction_accepted for trial in self.trials)

    @property
    def norm_rejections(self) -> int:
        return sum(trial.norm_accepted is False for trial in self.trials)

    @property
    def encoding_rejections(self) -> int:
        return sum(trial.encoding_accepted is False for trial in self.trials)


@dataclass(frozen=True, slots=True)
class SigningResult:
    """A returned signature plus the accepted preimage and retry diagnostics."""

    signature: Signature
    s1: tuple[int, ...]
    s2: tuple[int, ...]
    statistics: SigningStatistics

    @property
    def wire(self) -> bytes:
        return bytes(self.signature)


def _rounded_integer_polynomial(
    spectrum: Sequence[Any],
    *,
    backend: str = "float",
    dps: int = 100,
    tolerance: float = 2.0e-7,
) -> tuple[int, ...]:
    """Invert an FFT coordinate and reject any non-integral numerical residue."""

    if backend == "mpmath":
        import mpmath as mp

        values = ifft_high_precision(spectrum, dps=dps, real_if_close=False)
        with mp.workdps(dps):
            result: list[int] = []
            threshold = mp.power(10, -(dps - 20))
            for index, value in enumerate(values):
                real = mp.re(value)
                imaginary = abs(mp.im(value))
                nearest = int(mp.nint(real))
                if imaginary > threshold or abs(real - nearest) > threshold:
                    raise ArithmeticError(
                        "high-precision ffSampling output did not invert to an "
                        f"integer at coefficient {index}: {value!r}"
                    )
                result.append(nearest)
            return tuple(result)
    if backend != "float":
        raise ValueError("backend must be 'float' or 'mpmath'")

    values = ifft(spectrum, real_if_close=False)
    result: list[int] = []
    for index, value in enumerate(values):
        number = complex(value)
        nearest = round(number.real)
        if abs(number.imag) > tolerance or abs(number.real - nearest) > tolerance:
            raise ArithmeticError(
                "ffSampling output did not invert to an integer at coefficient "
                f"{index}: {number!r}"
            )
        result.append(int(nearest))
    return tuple(result)


def sample_preimage(
    secret_key: SecretKey,
    point: Sequence[int],
    sampler: OneDimensionalSampler,
    rng: Any,
) -> PreimageSample:
    """Sample one short preimage proposal for a fixed syndrome.

    The target coordinates use the closed form for the inverse of
    ``B=[[g,-f],[G,-F]]``.  Lattice multiplication is then recomputed with
    exact integer negacyclic products after the sampled FFT coordinates have
    been recovered, so floating-point error never enters the verification
    relation.
    """

    params = secret_key.parameters
    try:
        syndrome = tuple(point)
    except TypeError as exc:
        raise TypeError("hash point must be an iterable") from exc
    if len(syndrome) != params.n:
        raise ValueError("hash point has the wrong ring degree")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in syndrome):
        raise TypeError("hash point coefficients must be integers")
    if any(not 0 <= value < params.q for value in syndrome):
        raise ValueError("hash point coefficients must be canonical modulo q")

    backend = getattr(secret_key.sampler_tree, "backend", "float")
    dps = getattr(secret_key.sampler_tree, "dps", 100)
    b_fft = secret_key.basis_fft[0][1]
    d_fft = secret_key.basis_fft[1][1]

    if backend == "mpmath":
        import mpmath as mp

        with mp.workdps(dps):
            point_fft = fft_high_precision(syndrome, dps=dps)
            # With a=g, b=-f, c=G, d=-F, (point,0) B^{-1} is
            # (point*d/q, -point*b/q).
            target0_fft = [
                value / params.q for value in mul_fft(point_fft, d_fft)
            ]
            target1_fft = [
                -value / params.q for value in mul_fft(point_fft, b_fft)
            ]
            sampled = ff_sample(
                (target0_fft, target1_fft), secret_key.sampler_tree, sampler, rng
            )
    elif backend == "float":
        point_fft = fft(syndrome)
        target0_fft = [value / params.q for value in mul_fft(point_fft, d_fft)]
        target1_fft = [-value / params.q for value in mul_fft(point_fft, b_fft)]
        sampled = ff_sample(
            (target0_fft, target1_fft), secret_key.sampler_tree, sampler, rng
        )
    else:
        raise ValueError(f"unknown sampler-tree backend: {backend!r}")

    z0 = _rounded_integer_polynomial(
        sampled.z_fft[0], backend=backend, dps=dps
    )
    z1 = _rounded_integer_polynomial(
        sampled.z_fft[1], backend=backend, dps=dps
    )

    # v = z*B, and s=(point,0)-v.  Compute this exactly in Z[x]/(x^n+1).
    v0 = add(
        negacyclic_mul(z0, secret_key.g),
        negacyclic_mul(z1, secret_key.capital_g),
    )
    v1 = add(
        negacyclic_mul(z0, [-value for value in secret_key.f]),
        negacyclic_mul(z1, [-value for value in secret_key.capital_f]),
    )
    s1 = tuple(sub(syndrome, v0))
    s2 = tuple(-value for value in v1)

    # This is an internal invariant, stronger than merely relying on Verify.
    relation = add(
        list(s1),
        negacyclic_mul(secret_key.public_key.h, s2, params.q),
    )
    if any((left - right) % params.q for left, right in zip(relation, syndrome, strict=True)):
        raise ArithmeticError("sampled preimage failed the exact public-key relation")

    return PreimageSample(s1, s2, z0, z1, sampled.trace)


CorrectionEvaluator = Callable[..., CorrectionDecision]


def _precision_string(value: Any, digits: int) -> str | None:
    """Serialize an mpmath diagnostic without falling back to 15 display digits."""

    if value is None:
        return None
    import mpmath as mp

    return mp.nstr(value, n=max(17, digits))


class SigningCorrectionContext:
    """One-key, in-memory ThetaDiv cache; never a private-key wire format.

    Binding to object identity avoids accidentally retaining width caches
    across keys.  Only the current key and width-dependent kernel data are
    retained, not sampled centres, uniforms, or full traces.
    """

    def __init__(self, secret_key: SecretKey) -> None:
        if not isinstance(secret_key, SecretKey):
            raise TypeError("secret_key must be a SecretKey")
        from .theta_division import ThetaDivContext

        self.secret_key = secret_key
        self.kernel = ThetaDivContext()

    def require_key(self, secret_key: SecretKey) -> None:
        if self.secret_key is not secret_key:
            raise ValueError("correction context belongs to a different secret key")

    def prepare(self, *, bits: int = DEFAULT_CORRECTION_BITS + 32) -> dict[str, Any]:
        self.kernel.prepare_widths(
            (leaf.sigma for leaf in iter_leaves(self.secret_key.sampler_tree)),
            bits=bits,
            multiplier=self.secret_key.parameters.correction_multiplier_decimal,
        )
        return self.kernel.diagnostics()

    def diagnostics(self) -> dict[str, Any]:
        return self.kernel.diagnostics()


def sign_message(
    secret_key: SecretKey,
    message: bytes | bytearray | memoryview,
    *,
    rng: Any = None,
    sampler: OneDimensionalSampler | None = None,
    codec: SignatureCodec | None = None,
    max_trials: int = DEFAULT_MAX_SIGNING_TRIALS,
    correction_dps: int = DEFAULT_CORRECTION_DPS,
    correction_bits: int = DEFAULT_CORRECTION_BITS,
    correction_evaluator: CorrectionEvaluator = evaluate_correction,
    allow_provisional_wire: bool = False,
    correction_backend: str = "primal",
    diagnostic_level: str = "full",
    correction_context: SigningCorrectionContext | None = None,
) -> SigningResult:
    """Sign a message with fresh syndrome and clipped KGPV correction."""

    if not isinstance(secret_key, SecretKey):
        raise TypeError("secret_key must be a SecretKey")
    if not isinstance(message, (bytes, bytearray, memoryview)):
        raise TypeError("message must be bytes-like")
    message_bytes = bytes(message)
    if isinstance(max_trials, bool) or not isinstance(max_trials, int):
        raise TypeError("max_trials must be an integer")
    if max_trials <= 0:
        raise ValueError("max_trials must be positive")
    if correction_backend not in {"primal", "thetadiv"}:
        raise ValueError("correction_backend must be 'primal' or 'thetadiv'")
    if diagnostic_level not in {"full", "counts"}:
        raise ValueError("diagnostic_level must be 'full' or 'counts'")
    if correction_backend == "primal" and correction_context is not None:
        raise ValueError("correction_context is only used by thetadiv")
    context = correction_context
    if correction_backend == "thetadiv":
        context = SigningCorrectionContext(secret_key) if context is None else context
        if not isinstance(context, SigningCorrectionContext):
            raise TypeError("correction_context must be a SigningCorrectionContext")
        context.require_key(secret_key)

    params = secret_key.parameters
    random_source = make_prng() if rng is None else rng
    integer_sampler = sampler or DiscreteGaussianSampler(
        mode="high_precision", method="rejection"
    )
    if not callable(getattr(integer_sampler, "sample", None)):
        raise TypeError("sampler must provide sample(center, sigma, rng)")
    signature_codec = codec or make_signature_codec(params, formal=True)
    if signature_codec.parameters != params:
        raise ValueError("signature codec and secret key use different parameters")
    _require_signature_wire_acknowledgement(
        signature_codec, allow_provisional_wire
    )

    records: list[SigningTrialRecord] = []
    correction_metrics: dict[str, int | float] = {}
    correction_options: dict[str, Any] = {}
    # Keep calls to legacy injected evaluators unchanged under old defaults.
    if correction_backend != "primal" or diagnostic_level != "full":
        correction_options.update(
            correction_backend=correction_backend,
            diagnostic_level=diagnostic_level,
        )
    if context is not None:
        correction_options["correction_context"] = context.kernel
    for trial_index in range(1, max_trials + 1):
        # The salt is deliberately inside this loop.  Every correction, norm,
        # or encoding rejection discards the salt and its random-oracle point.
        salt = generate_salt(random_source)
        point = hash_to_point(
            secret_key.public_key.payload, salt, message_bytes, params
        )
        proposal = sample_preimage(
            secret_key, point, integer_sampler, random_source
        )
        decision = correction_evaluator(
            proposal.trace,
            params,
            random_source=random_source,
            bits=correction_bits,
            dps=correction_dps,
            **correction_options,
        )
        for name, value in getattr(decision, "metrics", {}).items():
            correction_metrics[name] = correction_metrics.get(name, 0) + value
        log_delta = _precision_string(decision.log_delta, correction_dps)
        probability = _precision_string(decision.probability, correction_dps)
        if not decision.accepted:
            records.append(
                SigningTrialRecord(
                    trial=trial_index,
                    correction_accepted=False,
                    norm_accepted=None,
                    encoding_accepted=None,
                    squared_norm=None,
                    log_delta=log_delta,
                    correction_probability=probability,
                    correction_backend=correction_backend,
                    diagnostic_level=diagnostic_level,
                )
            )
            continue

        squared_norm = proposal.squared_norm
        if squared_norm > params.signature_norm_bound:
            records.append(
                SigningTrialRecord(
                    trial=trial_index,
                    correction_accepted=True,
                    norm_accepted=False,
                    encoding_accepted=None,
                    squared_norm=squared_norm,
                    log_delta=log_delta,
                    correction_probability=probability,
                    correction_backend=correction_backend,
                    diagnostic_level=diagnostic_level,
                )
            )
            continue

        try:
            encoded_s2 = signature_codec.encode(proposal.s2)
        except RANSEncodeError as exc:
            # Encoding is required to be lossless on every norm-valid signer
            # output.  Treat a failure as an implementation/policy error, not
            # as another probabilistic retry that would alter the scheme law.
            raise SigningEncodingError(
                "norm-valid signer output did not fit the configured codec"
            ) from exc

        records.append(
            SigningTrialRecord(
                trial=trial_index,
                correction_accepted=True,
                norm_accepted=True,
                encoding_accepted=True,
                squared_norm=squared_norm,
                log_delta=log_delta,
                correction_probability=probability,
                correction_backend=correction_backend,
                diagnostic_level=diagnostic_level,
            )
        )
        signature = Signature(salt, encoded_s2)
        return SigningResult(
            signature=signature,
            s1=proposal.s1,
            s2=proposal.s2,
            statistics=SigningStatistics(tuple(records), correction_metrics),
        )

    statistics = SigningStatistics(tuple(records), correction_metrics)
    raise SigningError(
        f"signing exhausted {max_trials} fresh-syndrome trials "
        f"(correction rejections={statistics.correction_rejections}, "
        f"norm rejections={statistics.norm_rejections}, "
        f"encoding rejections={statistics.encoding_rejections})",
        statistics,
    )


# Concise functional alias.  It returns the detailed result intentionally;
# :class:`falconpp.scheme.FalconPlusPlus` offers a bytes-only convenience API.
sign = sign_message


__all__ = [
    "DEFAULT_CORRECTION_BITS",
    "DEFAULT_CORRECTION_DPS",
    "DEFAULT_MAX_SIGNING_TRIALS",
    "PreimageSample",
    "SignatureCodec",
    "SigningError",
    "SigningEncodingError",
    "SigningResult",
    "SigningStatistics",
    "SigningCorrectionContext",
    "SigningTrialRecord",
    "make_signature_codec",
    "sample_preimage",
    "sign",
    "sign_message",
]
