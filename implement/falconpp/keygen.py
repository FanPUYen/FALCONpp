"""Falcon++ key generation and trapdoor expansion.

The accepted-key distribution is obtained by sampling ``f`` and ``g`` from
the configured coefficient Gaussian and applying the manuscript's norm,
invertibility, Gram--Schmidt, and NTRU-solvability filters.  The order of the
filters is an implementation optimization only; every accepted key satisfies
all of them.

This clarity-first Python code is not constant time.
"""

from __future__ import annotations

from decimal import Decimal, ROUND_FLOOR, localcontext
from typing import Any, Protocol, Sequence

from .ff_sampling import build_sampler_tree
from .fft import fft, fft_high_precision
from .gaussian import DiscreteGaussianSampler
from .keys import (
    FFTBasis,
    KeyGenerationStatistics,
    KeyPair,
    PublicKey,
    SecretKey,
)
from .ntru import (
    NTRUNoSolutionError,
    ReductionConfig,
    passes_gram_schmidt_bound,
    solve_ntru,
)
from .parameters import FalconPPParameters, get_parameters
from .polynomial import inverse_mod_q, negacyclic_mul
from .randomness import make_prng


DEFAULT_MAX_KEYGEN_ATTEMPTS = 100_000
DEFAULT_KEYGEN_DECIMAL_PRECISION = 100


class CoefficientSampler(Protocol):
    """Minimal interface needed by :func:`generate_keypair`."""

    def sample(self, center: float, sigma: Any, rng: Any = None) -> int:
        """Draw one integer coefficient."""


class KeyGenerationError(RuntimeError):
    """Raised when no acceptable key is found within the configured limit."""


class KeyExpansionError(RuntimeError):
    """A mathematically accepted key could not be expanded numerically."""


def exact_sigma_fg(
    parameters: int | str | FalconPPParameters,
    *,
    decimal_precision: int = DEFAULT_KEYGEN_DECIMAL_PRECISION,
) -> Decimal:
    """Compute ``gamma*sqrt(q/(2n))`` from printed decimal parameters.

    The registered dataclass also exposes a convenient binary64 property for
    displays.  Formal KeyGen instead starts from its retained paper-decimal
    source and performs the square root in :class:`decimal.Decimal`.
    """

    params = get_parameters(parameters)
    if isinstance(decimal_precision, bool) or not isinstance(decimal_precision, int):
        raise TypeError("decimal_precision must be an integer")
    if decimal_precision < 32:
        raise ValueError("decimal_precision must be at least 32")
    with localcontext() as context:
        context.prec = decimal_precision
        return +params.sigma_fg_decimal


def exact_keygen_norm_cap(
    parameters: int | str | FalconPPParameters,
) -> int:
    """Return ``floor(gamma^2*q)`` without binary floating-point rounding."""

    params = get_parameters(parameters)
    gamma = params.gamma_decimal
    # Squaring a finite decimal is exact when the context has at least twice
    # its coefficient length (plus room for q), so the final floor cannot be
    # changed by an ambient Decimal precision chosen by the caller.
    exact_digits = 2 * len(gamma.as_tuple().digits) + len(str(params.q)) + 4
    with localcontext() as context:
        context.prec = max(DEFAULT_KEYGEN_DECIMAL_PRECISION, exact_digits)
        value = gamma * gamma * Decimal(params.q)
        return int(value.to_integral_value(rounding=ROUND_FLOOR))


def _sample_polynomial(
    sampler: CoefficientSampler,
    degree: int,
    sigma: Any,
    rng: Any,
) -> list[int]:
    """Use a vector helper when available, otherwise use the scalar protocol."""

    vector_method = getattr(sampler, "sample_zero_centered", None)
    if callable(vector_method):
        values = vector_method(degree, sigma, rng)
    else:
        values = [sampler.sample(0.0, sigma, rng) for _ in range(degree)]
    try:
        polynomial = list(values)
    except TypeError as exc:
        raise TypeError("coefficient sampler did not return an iterable") from exc
    if len(polynomial) != degree:
        raise ValueError(
            f"coefficient sampler returned {len(polynomial)} values, "
            f"expected {degree}"
        )
    if any(isinstance(value, bool) or not isinstance(value, int) for value in polynomial):
        raise TypeError("coefficient sampler must return Python integers")
    return polynomial


def expand_ntru_basis(
    f: Sequence[int],
    g: Sequence[int],
    capital_f: Sequence[int],
    capital_g: Sequence[int],
    *,
    backend: str = "mpmath",
    dps: int = 100,
) -> FFTBasis:
    """Return the FFT expansion of Falcon's row basis ``[[g,-f],[G,-F]]``.

    ``backend="mpmath"`` is the formal reference path.  ``"float"`` is kept
    for fast diagnostics and toy cross-checks only.
    """

    n = len(f)
    if n == 0 or len(g) != n or len(capital_f) != n or len(capital_g) != n:
        raise ValueError("all NTRU polynomials must have one common nonzero degree")
    if backend == "mpmath":
        transform = lambda values: fft_high_precision(values, dps=dps)
    elif backend == "float":
        transform = fft
    else:
        raise ValueError("backend must be 'mpmath' or 'float'")
    basis = (
        (tuple(transform(g)), tuple(transform([-value for value in f]))),
        (
            tuple(transform(capital_g)),
            tuple(transform([-value for value in capital_f])),
        ),
    )
    return basis


def generate_keypair(
    parameters: int | str | FalconPPParameters,
    *,
    rng: Any = None,
    sampler: CoefficientSampler | None = None,
    max_attempts: int = DEFAULT_MAX_KEYGEN_ATTEMPTS,
    gram_schmidt_dps: int = 100,
    reduction_config: ReductionConfig | None = None,
    fft_backend: str = "mpmath",
    fft_dps: int = 100,
    keygen_decimal_precision: int = DEFAULT_KEYGEN_DECIMAL_PRECISION,
) -> KeyPair:
    """Generate and fully expand one Falcon++ key pair.

    Args:
        parameters: One registered profile (or a compatible parameter object).
        rng: Random byte source.  ``None`` creates an OS-seeded SHAKE stream.
        sampler: Injectable one-dimensional sampler.  The authoritative
            high-precision sampler is used by default.
        max_attempts: Public safety limit on sampled ``(f,g)`` pairs.
        gram_schmidt_dps: Decimal precision for the pre-solver GS filter.
        reduction_config: Optional deterministic NTRU Babai controls.

    Returns:
        A :class:`~falconpp.keys.KeyPair` containing exact polynomials, the
        public payload, cached FFT basis/tree, and rejection counters.
    """

    params = get_parameters(parameters)
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int):
        raise TypeError("max_attempts must be an integer")
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    if isinstance(gram_schmidt_dps, bool) or not isinstance(gram_schmidt_dps, int):
        raise TypeError("gram_schmidt_dps must be an integer")
    if gram_schmidt_dps < 30:
        raise ValueError("gram_schmidt_dps must be at least 30")

    random_source = make_prng() if rng is None else rng
    coefficient_sampler = sampler or DiscreteGaussianSampler(
        mode="high_precision", method="rejection"
    )
    if not callable(getattr(coefficient_sampler, "sample", None)):
        raise TypeError("sampler must provide sample(center, sigma, rng)")
    sigma_fg = exact_sigma_fg(
        params, decimal_precision=keygen_decimal_precision
    )
    norm_cap = exact_keygen_norm_cap(params)

    norm_rejections = 0
    invertibility_rejections = 0
    gram_schmidt_rejections = 0
    ntru_rejections = 0

    for attempt in range(1, max_attempts + 1):
        f = _sample_polynomial(
            coefficient_sampler, params.n, sigma_fg, random_source
        )
        g = _sample_polynomial(
            coefficient_sampler, params.n, sigma_fg, random_source
        )

        squared_norm = sum(value * value for value in (*f, *g))
        if squared_norm > norm_cap:
            norm_rejections += 1
            continue

        try:
            inverse_f = inverse_mod_q(f, params.q)
        except ValueError:
            invertibility_rejections += 1
            continue

        if not passes_gram_schmidt_bound(
            f, g, params.q, params.gamma_decimal, dps=gram_schmidt_dps
        ):
            gram_schmidt_rejections += 1
            continue

        try:
            solution = solve_ntru(
                f, g, params.q, reduction_config=reduction_config
            )
        except NTRUNoSolutionError:
            ntru_rejections += 1
            continue

        capital_f = solution.capital_f
        capital_g = solution.capital_g
        h = negacyclic_mul(g, inverse_f, params.q)
        public_key = PublicKey.from_coefficients(h, params)

        # Numerical expansion is deterministic working storage, not a scheme
        # rejection filter.  If it fails, stop loudly: silently resampling
        # would change the accepted-key distribution defined by the paper.
        try:
            if fft_backend == "mpmath":
                import mpmath as mp

                with mp.workdps(fft_dps):
                    tree_sigma: Any = mp.mpf(str(params.sigma_sig_decimal))
            else:
                tree_sigma = params.sigma_sig
            basis_fft = expand_ntru_basis(
                f,
                g,
                capital_f,
                capital_g,
                backend=fft_backend,
                dps=fft_dps,
            )
            sampler_tree = build_sampler_tree(
                basis_fft,
                tree_sigma,
                backend=fft_backend,
                dps=fft_dps,
            )
        except (ArithmeticError, ZeroDivisionError):
            raise KeyExpansionError(
                "accepted NTRU key could not be expanded into a positive "
                "high-precision ffLDL tree"
            ) from None

        secret_key = SecretKey(
            parameters=params,
            f=tuple(f),
            g=tuple(g),
            capital_f=tuple(capital_f),
            capital_g=tuple(capital_g),
            public_key=public_key,
            basis_fft=basis_fft,
            sampler_tree=sampler_tree,
            _inverse_f_hint=inverse_f,
        )
        statistics = KeyGenerationStatistics(
            sampled_pairs=attempt,
            norm_rejections=norm_rejections,
            invertibility_rejections=invertibility_rejections,
            gram_schmidt_rejections=gram_schmidt_rejections,
            ntru_rejections=ntru_rejections,
        )
        return KeyPair(secret_key, public_key, statistics)

    raise KeyGenerationError(
        f"no acceptable {params.name} key after {max_attempts} sampled pairs; "
        f"rejections: norm={norm_rejections}, "
        f"invertibility={invertibility_rejections}, "
        f"Gram-Schmidt={gram_schmidt_rejections}, NTRU={ntru_rejections}"
    )


# Conventional short spelling used by scripts and examples.
keygen = generate_keypair


__all__ = [
    "CoefficientSampler",
    "DEFAULT_KEYGEN_DECIMAL_PRECISION",
    "DEFAULT_MAX_KEYGEN_ATTEMPTS",
    "KeyGenerationError",
    "KeyExpansionError",
    "expand_ntru_basis",
    "exact_keygen_norm_cap",
    "exact_sigma_fg",
    "generate_keypair",
    "keygen",
]
