"""Falcon++ Section 6 parameter profiles and GSO heuristic."""

from __future__ import annotations

from dataclasses import dataclass

import mpmath as mp


@dataclass(frozen=True)
class SecurityProfile:
    """An analytical profile, with optional published comparison targets.

    Additional candidate profiles deliberately leave the ``paper_*`` fields
    empty: externally supplied candidate numbers are not paper constants and
    must never silently become operational inputs.
    """

    name: str
    param_id: int
    n: int
    q: int
    gamma: str
    sigma_sig: str
    beta: int
    security_bits: int
    paper_moment_order: int | None = None
    paper_gamma_hat: str | None = None
    paper_p_corr: str | None = None
    paper_norm_exponent: str | None = None
    paper_falconws_trials: str | None = None
    paper_key_recovery: tuple[int, int] | None = None
    paper_forgery: tuple[int, int] | None = None
    paper_chi_bdd: tuple[int, int] | None = None
    paper_chi_challenges_log2: int | None = None

    @property
    def gamma_mpf(self) -> mp.mpf:
        return mp.mpf(self.gamma)

    @property
    def sigma_sig_mpf(self) -> mp.mpf:
        return mp.mpf(self.sigma_sig)

    @property
    def s(self) -> mp.mpf:
        return mp.sqrt(2 * mp.pi) * self.sigma_sig_mpf

    @property
    def sigma_fg(self) -> mp.mpf:
        return self.gamma_mpf * mp.sqrt(mp.mpf(self.q) / (2 * self.n))

    @property
    def candidate_cap(self) -> int:
        return (1 << 64) + (1 << (50 if self.n == 512 else 36))

    @property
    def raw_trial_cap(self) -> int:
        if self.n == 512 and self.gamma == "1.25":
            return (1 << 64) + (1 << 63)
        return 1 << 65


PAPER_PROFILES: tuple[SecurityProfile, ...] = (
    SecurityProfile(
        "falconpp-512-953-gamma117", 0x01, 512, 953, "1.17", "19.1399", 674, 128,
        142, "4.533254", "0.562718", "10.0115", "9.95",
        (133, 121), (171, 155), (133, 121), 35,
    ),
    SecurityProfile(
        "falconpp-512-953-gamma125", 0x02, 512, 953, "1.25", "20.9921", 739, 128,
        272, "1.774456", "0.737699", "9.9455", "5.14",
        (136, 124), (164, 149), (136, 124), 46,
    ),
    SecurityProfile(
        "falconpp-1024-1949-gamma117", 0x03, 1024, 1949, "1.17", "28.2544", 1407, 256,
        145, "9.067426", "0.575062", "19.9942", "20.94",
        (273, 248), (372, 338), (273, 248), 36,
    ),
    SecurityProfile(
        "falconpp-1024-1949-gamma125", 0x04, 1024, 1949, "1.25", "30.0203", 1495, 256,
        184, "3.714950", "0.644956", "20.0122", "26.39",
        (278, 252), (364, 330), (278, 252), 46,
    ),
)


def get_profile(name: str) -> SecurityProfile:
    for profile in PAPER_PROFILES:
        if profile.name == name:
            return profile
    raise KeyError(f"unknown Falcon++ security profile: {name}")


PRINTED_ASYMPTOTIC = "printed_asymptotic"
DETERMINANT_NORMALIZED = "determinant_normalized_inferred_missing_spec"


def finite_dimension_pair_factor(n: int) -> mp.mpf:
    """Return the inferred second-block factor that makes the volume exact.

    ``main.tex`` says that finite-dimensional geometric normalization is used
    (lines 101--103 and 238--241), but it never defines it.  If the first block
    is kept fixed, the determinant constraint uniquely gives

        a_n = (prod_{i=0}^{n/2-1} ((n-i)/n))**(-2/n).

    This function is intentionally labelled *inferred_missing_spec*.  It is
    useful for reproducing the paper table, but is not presented as a formula
    that currently appears in the paper.
    """

    if n < 2 or n % 2:
        raise ValueError("n must be a positive even integer")
    log_product = mp.fsum(mp.log(mp.mpf(n - i) / n) for i in range(n // 2))
    return mp.exp(-2 * log_product / n)


def heuristic_gso_lengths(
    profile: SecurityProfile,
    *,
    normalization: str = PRINTED_ASYMPTOTIC,
) -> tuple[mp.mpf, ...]:
    """Return the paired GSO profile under an explicitly selected convention.

    ``printed_asymptotic`` uses Eq. ``prest-gso-model`` literally, including
    its ``e/2`` second-block factor.  It is not exactly volume compatible.
    ``determinant_normalized_inferred_missing_spec`` preserves the first block
    and replaces ``e/2`` by :func:`finite_dimension_pair_factor`; this is the
    missing normalization apparently used for the Section 6 baseline.
    """

    n = profile.n
    q_root = mp.sqrt(profile.q)
    first = profile.gamma_mpf * q_root
    if normalization == PRINTED_ASYMPTOTIC:
        pair_factor = mp.e / 2
    elif normalization == DETERMINANT_NORMALIZED:
        pair_factor = finite_dimension_pair_factor(n)
    else:
        raise ValueError(f"unknown GSO normalization: {normalization}")
    second = pair_factor * q_root / profile.gamma_mpf
    values: list[mp.mpf] = []
    for scale in (first, second):
        for i in range(n // 2):
            length = scale * mp.sqrt(mp.mpf(n - i) / n)
            values.extend((length, length))
    return tuple(values)


def profile_log_volume(
    profile: SecurityProfile,
    *,
    normalization: str = PRINTED_ASYMPTOTIC,
) -> mp.mpf:
    return mp.fsum(
        mp.log(length)
        for length in heuristic_gso_lengths(profile, normalization=normalization)
    )


def profile_volume_ratio(
    profile: SecurityProfile,
    *,
    normalization: str = PRINTED_ASYMPTOTIC,
) -> mp.mpf:
    """Return product(model lengths) / q**n; one would mean exact volume."""

    return mp.exp(
        profile_log_volume(profile, normalization=normalization)
        - profile.n * mp.log(profile.q)
    )
