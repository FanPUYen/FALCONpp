"""Paper-model attack work factors from Falcon Section 2.5.1."""

from __future__ import annotations

from dataclasses import dataclass

import mpmath as mp

from .profiles import SecurityProfile


@dataclass(frozen=True)
class CoreSvpCost:
    block_size: int
    classical_raw: mp.mpf
    quantum_raw: mp.mpf
    classical_bits: int
    quantum_bits: int


@dataclass(frozen=True)
class AttackCostEstimate:
    profile_name: str
    key_recovery: CoreSvpCost
    forgery: CoreSvpCost
    chi_bdd_composed: CoreSvpCost
    chi_bdd_challenge_log2: int
    chi_bdd_minimum_challenges: int
    chi_bdd_character_gap: mp.mpf
    chi_bdd_error_at_power_of_two: mp.mpf
    heuristic_work_factors: bool = True
    chi_bdd_formula_scope: str = "reconstructed_missing_from_main_tex"
    endpoint_advantage_model_available: bool = False
    warnings: tuple[str, ...] = (
        "uses_main_tex_quantum_core_svp_slope_0.265",
        "chi_bdd_character_hoeffding_formula_is_missing_from_main_tex",
        "work_factors_are_not_endpoint_advantage_bounds",
    )


def core_svp_cost(block_size: int) -> CoreSvpCost:
    if block_size <= 0:
        raise ValueError("block size must be positive")
    classical = mp.mpf("0.292") * block_size
    quantum = mp.mpf("0.265") * block_size
    return CoreSvpCost(
        block_size,
        +classical,
        +quantum,
        int(mp.floor(classical)),
        int(mp.floor(quantum)),
    )


def key_recovery_block_size(
    n: int,
    q: object,
    sigma_fg: object,
    *,
    minimum: int = 100,
    maximum: int = 10000,
) -> int:
    """First integer b satisfying Falcon's full-dimensional KR condition."""

    modulus = mp.mpf(q)
    sigma = mp.mpf(sigma_fg)
    if n <= 0 or modulus <= 0 or sigma <= 0:
        raise ValueError("n, q, and sigma_fg must be positive")
    for block in range(minimum, maximum + 1):
        b = mp.mpf(block)
        lhs = (b / (2 * mp.pi * mp.e)) ** (1 - mp.mpf(n) / b) * mp.sqrt(modulus)
        rhs = mp.sqrt(3 * b / 4) * sigma
        if lhs > rhs:
            return block
    raise ArithmeticError("key-recovery block-size search exceeded maximum")


def forgery_block_size(
    n: int,
    q: object,
    beta: object,
    *,
    minimum: int = 100,
    maximum: int = 10000,
) -> int:
    """First integer b satisfying Falcon's full-dimensional forgery condition."""

    modulus = mp.mpf(q)
    radius = mp.mpf(beta)
    if n <= 0 or modulus <= 0 or radius <= 0:
        raise ValueError("n, q, and beta must be positive")
    for block in range(minimum, maximum + 1):
        b = mp.mpf(block)
        lhs = (b / (2 * mp.pi * mp.e)) ** (mp.mpf(n) / b) * mp.sqrt(modulus)
        if lhs <= radius:
            return block
    raise ArithmeticError("forgery block-size search exceeded maximum")


def chi_bdd_challenge_count(
    profile: SecurityProfile,
) -> tuple[int, int, mp.mpf, mp.mpf]:
    """Reconstruct the character/Hoeffding challenge calculation.

    This derivation exactly reproduces Section 6's challenge counts, but the
    formula itself is absent from ``main.tex`` and the checked Falconws code.
    The return tuple is ``(ceil_log2, exact_ceiling, gap, error_at_2**ell)``.
    """

    norm_cap = int(mp.floor(profile.gamma_mpf**2 * profile.q))
    sigma = profile.sigma_sig_mpf
    gap = mp.exp(-2 * mp.pi**2 * sigma**2 * norm_cap / profile.q**2)
    minimum_real = 8 * mp.log(8) / gap**2
    minimum_integer = int(mp.ceil(minimum_real))
    log2_power = int(mp.ceil(mp.log(minimum_real, 2)))
    power_count = 1 << log2_power
    total_error_bound = 2 * mp.exp(-mp.mpf(power_count) * gap**2 / 8)
    return log2_power, minimum_integer, +gap, +total_error_bound


def estimate_attack_costs(
    profile: SecurityProfile, *, dps: int = 80
) -> AttackCostEstimate:
    with mp.workdps(dps):
        key_block = key_recovery_block_size(profile.n, profile.q, profile.sigma_fg)
        forgery_block = forgery_block_size(profile.n, profile.q, profile.beta)
        challenge_log, challenge_min, gap, error = chi_bdd_challenge_count(profile)
        key_cost = core_svp_cost(key_block)
        return AttackCostEstimate(
            profile_name=profile.name,
            key_recovery=key_cost,
            forgery=core_svp_cost(forgery_block),
            # Section 6 composes relation recovery with a cheap character test and
            # therefore prints the key-recovery exponent again.
            chi_bdd_composed=key_cost,
            chi_bdd_challenge_log2=challenge_log,
            chi_bdd_minimum_challenges=challenge_min,
            chi_bdd_character_gap=+gap,
            chi_bdd_error_at_power_of_two=+error,
        )
