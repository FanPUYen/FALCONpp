"""Object-oriented facade for the single fast Falcon++ research runtime."""

from __future__ import annotations

from typing import Any

from .gaussian import DiscreteGaussianSampler
from .keygen import (
    DEFAULT_MAX_KEYGEN_ATTEMPTS,
    CoefficientSampler,
    generate_keypair,
)
from .keys import KeyPair, PublicKey, SecretKey, Signature
from .ntru import ReductionConfig
from .parameters import FalconPPParameters, get_parameters
from .randomness import make_prng
from .signing import (
    DEFAULT_CORRECTION_BITS,
    DEFAULT_CORRECTION_DPS,
    DEFAULT_MAX_SIGNING_TRIALS,
    SignatureCodec,
    SigningCorrectionContext,
    SigningResult,
    make_signature_codec,
    sign_message,
)
from .verification import (
    VerificationResult,
    verify_signature,
    verify_signature_detailed,
)


def _resolve_rng(seed: bytes | bytearray | memoryview | None, rng: Any) -> Any:
    if seed is not None and rng is not None:
        raise ValueError("supply either seed or rng, not both")
    if seed is not None:
        return make_prng(seed)
    return rng


class FalconPlusPlus:
    """Readable KeyGen/Sign/Verify interface for one fixed parameter profile.

    Normal calls use OS-seeded SHAKE256 streams.  A byte seed may be injected
    per operation to replay tests; deterministic mode is not a production
    randomness source.  Key generation is independent of the pending
    signature-wire decision.  Formal signing and verification require the
    explicit research acknowledgement ``allow_provisional_wire=True`` until
    the canonical rANS model and standardized lengths are frozen.

    The default runtime uses the exact Python general-center sampler, cached
    zero-center sampling, float FFT, and ThetaDiv correction. Component
    overrides are retained only for testing and arithmetic cross-validation;
    they do not change the high-precision NTRU and GSO defaults.
    """

    def __init__(
        self,
        parameters: int | str | FalconPPParameters,
        *,
        formal_signatures: bool = True,
        padded_signature_payload_length: int | None = None,
        allow_provisional_wire: bool = False,
        sampler: CoefficientSampler | None = None,
        fft_backend: str = "float",
        fft_dps: int = 100,
        correction_dps: int = DEFAULT_CORRECTION_DPS,
        correction_bits: int = DEFAULT_CORRECTION_BITS,
        correction_backend: str = "thetadiv",
        diagnostic_level: str = "counts",
    ) -> None:
        if not isinstance(allow_provisional_wire, bool):
            raise TypeError("allow_provisional_wire must be a bool")
        if correction_backend not in {"primal", "thetadiv"}:
            raise ValueError("correction_backend must be 'primal' or 'thetadiv'")
        if diagnostic_level not in {"full", "counts"}:
            raise ValueError("diagnostic_level must be 'full' or 'counts'")
        self.parameters = get_parameters(parameters)
        self.codec = make_signature_codec(
            self.parameters,
            formal=formal_signatures,
            padded_length=padded_signature_payload_length,
        )
        self.allow_provisional_wire = allow_provisional_wire
        self.sampler = sampler or DiscreteGaussianSampler(
            mode="high_precision", method="rejection",
            zero_center_backend="cached_probability",
            general_center_backend="exact_python_fast",
            general_center_initial_bits=64,
            general_center_refinement_bits=32,
        )
        self.fft_backend = fft_backend
        self.fft_dps = fft_dps
        self.correction_dps = correction_dps
        self.correction_bits = correction_bits
        self.correction_backend = correction_backend
        self.diagnostic_level = diagnostic_level
        self._correction_context: SigningCorrectionContext | None = None
        self._prepared_sampler_tree: Any = None

    def prepare_sampler(self, secret_key: SecretKey) -> dict[str, Any]:
        """Prepare exact widths for the Python sampler, without new key data.

        This is optional: signing prepares on first use if necessary. Only one
        tree's widths are retained, and switching keys clears the width cache.
        An explicitly injected validation sampler may need no preparation.
        """

        if not isinstance(secret_key, SecretKey):
            raise TypeError("secret_key must be a SecretKey")
        if secret_key.parameters != self.parameters:
            raise ValueError("secret key and scheme use different parameter sets")
        if getattr(self.sampler, "general_center_backend", "reference") != "exact_python_fast":
            return {}
        tree = secret_key.sampler_tree
        if self._prepared_sampler_tree is not tree:
            from .ff_sampling import iter_leaves

            leaves = iter_leaves(tree)
            if len(leaves) != 2 * self.parameters.n:
                raise ValueError("sampler tree must contain exactly 2*n widths")
            self.sampler.prepare_widths(
                (leaf.sigma for leaf in leaves), max_widths=2 * self.parameters.n,
            )
            self._prepared_sampler_tree = tree
        return self.sampler_diagnostics()

    def sampler_diagnostics(self) -> dict[str, Any]:
        """Nonsecret width-cache counts; no centers, widths, or sampled values."""

        if getattr(self.sampler, "general_center_backend", "reference") != "exact_python_fast":
            return {}
        return self.sampler.general_center_cache_info()

    def _context_for_key(self, secret_key: SecretKey) -> SigningCorrectionContext | None:
        if not isinstance(secret_key, SecretKey):
            raise TypeError("secret_key must be a SecretKey")
        if secret_key.parameters != self.parameters:
            raise ValueError("secret key and scheme use different parameter sets")
        if self.correction_backend != "thetadiv":
            return None
        if (self._correction_context is None or
                self._correction_context.secret_key is not secret_key):
            self._correction_context = SigningCorrectionContext(secret_key)
        return self._correction_context

    def prepare_correction(
        self, secret_key: SecretKey, *, bits: int | None = None
    ) -> dict[str, Any]:
        """Optionally precompute this key's width-only ThetaDiv coefficients.

        Callers measuring setup costs should time this method separately;
        without it, the same work is performed lazily on first use.
        """

        context = self._context_for_key(secret_key)
        if context is None:
            return {"correction_backend": "primal", "precomputation": "not_applicable"}
        return context.prepare(bits=self.correction_bits + 32 if bits is None else bits)

    def correction_diagnostics(self) -> dict[str, Any]:
        """Aggregate work counters for the current key, never its trace."""

        if self._correction_context is None:
            return {"correction_backend": self.correction_backend, "context_created": False}
        return dict(self._correction_context.diagnostics(), correction_backend="thetadiv")

    def _require_signature_wire_policy(self) -> None:
        """Refuse pending fixed-wire I/O without an explicit research opt-in."""

        if (
            self.codec.formal
            and not self.codec.wire_policy_resolved
            and not self.allow_provisional_wire
        ):
            raise ValueError(
                "the Falcon++ signature wire policy is provisional; pass "
                "allow_provisional_wire=True only for research experiments"
            )

    @property
    def public_key_payload_length(self) -> int:
        return self.parameters.public_key_payload_bytes

    @property
    def signature_payload_length(self) -> int | None:
        return self.codec.payload_length

    @property
    def signature_wire_length(self) -> int | None:
        if self.codec.payload_length is None:
            return None
        return self.parameters.salt_bytes + self.codec.payload_length

    def keygen(
        self,
        *,
        seed: bytes | bytearray | memoryview | None = None,
        rng: Any = None,
        max_attempts: int = DEFAULT_MAX_KEYGEN_ATTEMPTS,
        gram_schmidt_dps: int = 100,
        reduction_config: ReductionConfig | None = None,
        keygen_decimal_precision: int = 100,
    ) -> KeyPair:
        """Generate one exact NTRU trapdoor and its expanded signing cache."""

        return generate_keypair(
            self.parameters,
            rng=_resolve_rng(seed, rng),
            sampler=self.sampler,
            max_attempts=max_attempts,
            gram_schmidt_dps=gram_schmidt_dps,
            reduction_config=reduction_config,
            fft_backend=self.fft_backend,
            fft_dps=self.fft_dps,
            keygen_decimal_precision=keygen_decimal_precision,
        )

    def sign_detailed(
        self,
        secret_key: SecretKey,
        message: bytes | bytearray | memoryview,
        *,
        seed: bytes | bytearray | memoryview | None = None,
        rng: Any = None,
        max_trials: int = DEFAULT_MAX_SIGNING_TRIALS,
    ) -> SigningResult:
        self._require_signature_wire_policy()
        if secret_key.parameters != self.parameters:
            raise ValueError("secret key and scheme use different parameter sets")
        if getattr(self.sampler, "general_center_backend", "reference") == "exact_python_fast":
            self.prepare_sampler(secret_key)
        return sign_message(
            secret_key,
            message,
            rng=_resolve_rng(seed, rng),
            sampler=self.sampler,
            codec=self.codec,
            max_trials=max_trials,
            correction_dps=self.correction_dps,
            correction_bits=self.correction_bits,
            correction_backend=self.correction_backend,
            diagnostic_level=self.diagnostic_level,
            correction_context=self._context_for_key(secret_key),
            allow_provisional_wire=self.allow_provisional_wire,
        )

    def sign(
        self,
        secret_key: SecretKey,
        message: bytes | bytearray | memoryview,
        *,
        seed: bytes | bytearray | memoryview | None = None,
        rng: Any = None,
        max_trials: int = DEFAULT_MAX_SIGNING_TRIALS,
    ) -> bytes:
        """Return the canonical signature wire bytes."""

        return self.sign_detailed(
            secret_key,
            message,
            seed=seed,
            rng=rng,
            max_trials=max_trials,
        ).wire

    def verify_detailed(
        self,
        public_key: PublicKey | bytes | bytearray | memoryview,
        message: bytes | bytearray | memoryview,
        signature: Signature | bytes | bytearray | memoryview,
    ) -> VerificationResult:
        self._require_signature_wire_policy()
        return verify_signature_detailed(
            public_key,
            message,
            signature,
            parameters=self.parameters,
            codec=self.codec,
            allow_provisional_wire=self.allow_provisional_wire,
        )

    def verify(
        self,
        public_key: PublicKey | bytes | bytearray | memoryview,
        message: bytes | bytearray | memoryview,
        signature: Signature | bytes | bytearray | memoryview,
    ) -> bool:
        """Return whether a signature is canonical, valid, and short."""

        self._require_signature_wire_policy()
        return verify_signature(
            public_key,
            message,
            signature,
            parameters=self.parameters,
            codec=self.codec,
            allow_provisional_wire=self.allow_provisional_wire,
        )


# Alternative spelling used in a few experiment notebooks.
FalconPP = FalconPlusPlus


def keygen(
    parameters: int | str | FalconPPParameters,
    **kwargs: Any,
) -> KeyPair:
    """Functional KeyGen convenience wrapper."""

    return FalconPlusPlus(parameters).keygen(**kwargs)


def sign(
    secret_key: SecretKey,
    message: bytes | bytearray | memoryview,
    **kwargs: Any,
) -> bytes:
    """Functional signing convenience wrapper."""

    allow_provisional_wire = kwargs.pop("allow_provisional_wire", False)
    return FalconPlusPlus(
        secret_key.parameters,
        allow_provisional_wire=allow_provisional_wire,
    ).sign(secret_key, message, **kwargs)


def verify(
    public_key: PublicKey,
    message: bytes | bytearray | memoryview,
    signature: Signature | bytes | bytearray | memoryview,
    **kwargs: Any,
) -> bool:
    """Functional verification convenience wrapper."""

    allow_provisional_wire = kwargs.pop("allow_provisional_wire", False)
    return FalconPlusPlus(
        public_key.parameters,
        allow_provisional_wire=allow_provisional_wire,
    ).verify(
        public_key, message, signature, **kwargs
    )


__all__ = [
    "FalconPP",
    "FalconPlusPlus",
    "SignatureCodec",
    "keygen",
    "sign",
    "verify",
]
