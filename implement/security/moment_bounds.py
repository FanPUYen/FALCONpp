"""Section 6 heuristic moment and clipped-acceptance calculations.

All reported values retain their scope: they are evaluations of a heuristic
Gram--Schmidt profile, not support-wide certificates over KeyGen.  The order
search is deliberately two-stage.  Every order from 2 through 1024 is screened
with a vectorized evaluation, then the best candidates are recomputed with
``mpmath`` and the paper's finite root-lattice theta upper bound.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import mpmath as mp

from .profiles import (
    DETERMINANT_NORMALIZED,
    PRINTED_ASYMPTOTIC,
    SecurityProfile,
    heuristic_gso_lengths,
    profile_volume_ratio,
)
from .root_lattice import log_root_theta_product_upper, screen_root_theta_logs
from .theta import log_theta3


@dataclass(frozen=True)
class MomentRecord:
    order: int
    log_moment_upper: mp.mpf
    log_tau: mp.mpf
    tau: mp.mpf
    method: str
    dps: int | None
    truncation: int
    roots: int
    high_precision: bool
    machine_outward_rounded: bool


@dataclass(frozen=True)
class OrderSearch:
    records: tuple[MomentRecord, ...]
    screened_orders: tuple[int, ...]
    refined_orders: tuple[int, ...]
    selected: MomentRecord
    screening_method: str
    certification_method: str
    heuristic_profile: bool = True
    proves_global_optimum: bool = False


@dataclass(frozen=True)
class ProfileEstimate:
    profile: SecurityProfile
    normalization: str
    normalization_scope: str
    volume_ratio: mp.mpf
    candidate_cap: int
    sampler_budget_bits: mp.mpf
    target_deficit: mp.mpf
    log_delta_hat: mp.mpf
    delta_hat: mp.mpf
    order_search: OrderSearch
    active_moment_order: int
    active_log_moment_upper: mp.mpf
    gamma_hat_computed: mp.mpf
    operational_gamma_hat: mp.mpf
    operational_tau: mp.mpf
    deficit_upper: mp.mpf
    r_infinity_upper: mp.mpf
    p_corr_lower: mp.mpf
    norm_tail_exponent: mp.mpf
    p_norm_lower: mp.mpf
    total_mean_trials_upper: mp.mpf
    sampler_loss_bits: mp.mpf
    warnings: tuple[str, ...]
    heuristic_profile: bool = True
    conditional_bound: bool = True


def target_deficit(candidate_cap: int, sampler_budget_bits: object = 2) -> mp.mpf:
    """Return ``1 - 2**(-b/C)`` without catastrophic cancellation."""

    if candidate_cap <= 0:
        raise ValueError("candidate_cap must be positive")
    bits = mp.mpf(sampler_budget_bits)
    if bits <= 0:
        raise ValueError("sampler budget must be positive")
    return -mp.expm1(-bits * mp.log(2) / candidate_cap)


def _log_tau(order: int, log_moment: mp.mpf, log_deficit: mp.mpf) -> mp.mpf:
    k = mp.mpf(order)
    candidate = (
        k * mp.log(k)
        - (k - 1) * mp.log(k - 1)
        + log_deficit
        - log_moment
    ) / (k - 1)
    return min(mp.mpf("0"), candidate)


class HeuristicMomentModel:
    """Evaluate one Falcon++ Section 6 parameter profile."""

    def __init__(
        self,
        profile: SecurityProfile,
        *,
        normalization: str = DETERMINANT_NORMALIZED,
        dps: int = 100,
        sampler_budget_bits: object = 2,
    ) -> None:
        if dps < 40:
            raise ValueError("at least 40 decimal digits are required")
        self.profile = profile
        self.normalization = normalization
        self.dps = dps
        # Parse caller-supplied strings, Decimal objects, or existing mpf
        # values only after entering the requested precision context.  Calling
        # ``str(mpf_value)`` here would first round it at the ambient mp.dps.
        with mp.workdps(dps):
            parsed_budget = mp.mpf(sampler_budget_bits)
            if not mp.isfinite(parsed_budget) or parsed_budget <= 0:
                raise ValueError("sampler budget must be finite and positive")
            self.sampler_budget_bits = +parsed_budget

    @property
    def normalization_scope(self) -> str:
        if self.normalization == DETERMINANT_NORMALIZED:
            return "inferred_missing_spec"
        if self.normalization == PRINTED_ASYMPTOTIC:
            return "printed_equation"
        return "unknown"

    def _theta_arguments(self) -> tuple[mp.mpf, ...]:
        lengths = heuristic_gso_lengths(
            self.profile, normalization=self.normalization
        )
        # Every modeled GSO length is repeated exactly twice.  Returning one
        # argument per pair cuts both scan time and accidental counting errors.
        return tuple((self.profile.s / lengths[index]) ** 2 for index in range(0, 2 * self.profile.n, 2))

    def log_delta_hat(self) -> mp.mpf:
        """Evaluate delta from Eq. ``fresh-normalization`` in log domain.

        The hidden theta-only formula assumes exact modeled volume.  For the
        literally printed asymptotic profile we retain the otherwise omitted
        ``prod(lengths)/q**n`` factor, so this method remains consistent with
        the general definition even when that profile's volume is not exact.
        """

        with mp.workdps(self.dps):
            args = self._theta_arguments()
            d = 2 * self.profile.n
            log_volume_ratio = mp.log(
                profile_volume_ratio(
                    self.profile, normalization=self.normalization
                )
            )
            result = (
                log_volume_ratio
                + d * log_theta3(self.profile.s**2, dps=self.dps)
                - 2 * mp.fsum(log_theta3(value, dps=self.dps) for value in args)
            )
            return +result

    def log_moment_upper(
        self,
        order: int,
        *,
        truncation: int = 6,
        roots: int = 256,
    ) -> mp.mpf:
        """High-precision finite upper bound for ``log(M_k)``."""

        if order < 2:
            raise ValueError("moment order must be at least two")
        with mp.workdps(self.dps):
            s2 = self.profile.s**2
            d = 2 * self.profile.n
            prefactor = (
                d * log_theta3(s2 / order, dps=self.dps)
                - d * order * log_theta3(s2, dps=self.dps)
            )
            aggregate = log_root_theta_product_upper(
                self._theta_arguments(),
                order,
                multiplicity=2,
                dps=self.dps,
                truncation=truncation,
                roots=roots,
            )
            return +(prefactor + aggregate.log_product)

    def moment_record(
        self,
        order: int,
        *,
        truncation: int = 6,
        roots: int = 256,
    ) -> MomentRecord:
        with mp.workdps(self.dps):
            log_moment = self.log_moment_upper(
                order, truncation=truncation, roots=roots
            )
            deficit = target_deficit(
                self.profile.candidate_cap, self.sampler_budget_bits
            )
            log_tau = _log_tau(order, log_moment, mp.log(deficit))
            return MomentRecord(
                order=order,
                log_moment_upper=+log_moment,
                log_tau=+log_tau,
                tau=+mp.exp(log_tau),
                method="mpmath_finite_root_theta_upper",
                dps=self.dps,
                truncation=truncation,
                roots=roots,
                high_precision=True,
                machine_outward_rounded=False,
            )

    def screen_orders(
        self,
        orders: Iterable[int] = range(2, 1025),
        *,
        truncation: int = 3,
        roots: int = 128,
    ) -> tuple[MomentRecord, ...]:
        order_tuple = tuple(int(order) for order in orders)
        if not order_tuple:
            raise ValueError("order set must be non-empty")
        arguments = self._theta_arguments()
        batch = screen_root_theta_logs(
            arguments, order_tuple, truncation=truncation, roots=roots
        )
        d = 2 * self.profile.n
        s2 = self.profile.s**2
        base_log_theta = float(log_theta3(s2, dps=50))
        deficit = target_deficit(
            self.profile.candidate_cap, self.sampler_budget_bits
        )
        log_deficit = float(mp.log(deficit))

        records: list[MomentRecord] = []
        for order, root_logs in zip(order_tuple, batch.log_values, strict=True):
            log_moment_float = (
                d * float(log_theta3(s2 / order, dps=50))
                - d * order * base_log_theta
                + 2 * sum(root_logs)
            )
            log_tau_float = min(
                0.0,
                (
                    order * mp.log(order)
                    - (order - 1) * mp.log(order - 1)
                    + log_deficit
                    - log_moment_float
                )
                / (order - 1),
            )
            records.append(
                MomentRecord(
                    order=order,
                    log_moment_upper=mp.mpf(str(log_moment_float)),
                    log_tau=mp.mpf(str(log_tau_float)),
                    tau=mp.exp(mp.mpf(str(log_tau_float))),
                    method=batch.method,
                    dps=None,
                    truncation=truncation,
                    roots=roots,
                    high_precision=False,
                    machine_outward_rounded=False,
                )
            )
        return tuple(records)

    def select_order(
        self,
        orders: Iterable[int] = range(2, 1025),
        *,
        shortlist_size: int = 9,
        certification_truncation: int = 6,
        certification_roots: int = 256,
    ) -> OrderSearch:
        """Screen all orders and refine the strongest candidates with mpmath."""

        screened = self.screen_orders(orders)
        if shortlist_size < 1:
            raise ValueError("shortlist_size must be positive")
        ranked = sorted(screened, key=lambda item: item.log_tau, reverse=True)
        shortlist = ranked[: min(shortlist_size, len(ranked))]
        refined = tuple(
            self.moment_record(
                item.order,
                truncation=certification_truncation,
                roots=certification_roots,
            )
            for item in shortlist
        )
        selected = max(refined, key=lambda item: item.log_tau)
        return OrderSearch(
            records=screened,
            screened_orders=tuple(item.order for item in screened),
            refined_orders=tuple(item.order for item in refined),
            selected=selected,
            screening_method="numpy_finite_root_upper_M3_R128_binary64",
            certification_method=(
                f"mpmath_finite_root_upper_M{certification_truncation}_"
                f"R{certification_roots}_dps{self.dps}"
            ),
            # Binary64 screening does not provide an outward-rounded ordering
            # proof, even though selected moment values are recomputed at high
            # precision.  The artifact must not overclaim this distinction.
            proves_global_optimum=False,
        )

    def estimate(
        self,
        *,
        use_paper_operational_gamma: bool = False,
        order_search: OrderSearch | None = None,
        operational_moment_record: MomentRecord | None = None,
        operational_gamma: object | None = None,
        operational_order: int | None = None,
    ) -> ProfileEstimate:
        """Build the complete heuristic acceptance profile.

        When ``use_paper_operational_gamma`` is true, signing uses the six-digit
        multiplier printed in Section 6 and the deficit is re-evaluated at the
        paper's listed moment order.  Otherwise the freshly selected multiplier
        is used.  An explicit ``operational_gamma`` and ``operational_order``
        instead evaluate a caller's frozen, freshly optimized multiplier;
        they cannot be combined with the paper mode.  All modes preserve the
        b=2 target separately from the actual deficit after multiplier rounding.
        """

        with mp.workdps(self.dps):
            if use_paper_operational_gamma and (
                operational_gamma is not None or operational_order is not None
            ):
                raise ValueError("explicit operational values cannot use paper mode")
            if use_paper_operational_gamma and (
                self.profile.paper_gamma_hat is None
                or self.profile.paper_moment_order is None
            ):
                raise ValueError("this profile has no paper operational constants")
            if operational_order is not None and operational_order < 2:
                raise ValueError("operational order must be at least two")
            search = self.select_order() if order_search is None else order_search
            log_delta = self.log_delta_hat()
            delta = mp.exp(log_delta)
            computed_gamma = search.selected.tau / delta

            if use_paper_operational_gamma:
                gamma = mp.mpf(self.profile.paper_gamma_hat)
            elif operational_gamma is not None:
                gamma = mp.mpf(operational_gamma)
            else:
                gamma = computed_gamma
            if not mp.isfinite(gamma) or gamma <= 0:
                raise ValueError("operational gamma must be finite and positive")
            tau = gamma * delta
            if not 0 < tau <= 1:
                raise ArithmeticError(
                    "operational normalized multiplier exceeds one: "
                    f"tau={mp.nstr(tau, 20)} > 1"
                )

            if use_paper_operational_gamma:
                if operational_moment_record is None:
                    active_record = self.moment_record(
                        self.profile.paper_moment_order
                    )
                else:
                    if (
                        operational_moment_record.order
                        != self.profile.paper_moment_order
                    ):
                        raise ValueError(
                            "operational moment record does not match paper order"
                        )
                    active_record = operational_moment_record
            else:
                active_order = (
                    search.selected.order
                    if operational_order is None
                    else operational_order
                )
                if operational_moment_record is not None:
                    if operational_moment_record.order != active_order:
                        raise ValueError("operational moment record does not match order")
                    active_record = operational_moment_record
                elif active_order == search.selected.order:
                    active_record = search.selected
                else:
                    active_record = self.moment_record(active_order)

            k = active_record.order
            log_deficit = (
                (k - 1) * mp.log(k - 1)
                - k * mp.log(k)
                + (k - 1) * mp.log(tau)
                + active_record.log_moment_upper
            )
            deficit = mp.exp(log_deficit)
            if not 0 <= deficit < 1:
                raise ArithmeticError("clipping deficit must lie in [0, 1)")
            r_infinity = 1 / (1 - deficit)
            p_corr = tau * (1 - deficit)

            sigma = self.profile.sigma_sig_mpf
            u = mp.mpf(self.profile.beta**2 + 1) / (
                2 * self.profile.n * sigma**2
            )
            if u <= 1:
                raise ArithmeticError("the paper's norm-tail lemma requires u > 1")
            norm_exponent = self.profile.n * (u - 1 - mp.log(u))
            norm_failure = r_infinity * mp.exp(-norm_exponent)
            p_norm = max(mp.mpf("0"), 1 - norm_failure)
            total_trials = 1 / (p_corr * p_norm)
            sampler_loss = self.profile.candidate_cap * mp.log(r_infinity, 2)

            target = target_deficit(
                self.profile.candidate_cap, self.sampler_budget_bits
            )
            warnings = [
                "heuristic_profile_not_support_wide_certificate",
                "conditional_security_bound_open_proof_obligations",
            ]
            if self.normalization == DETERMINANT_NORMALIZED:
                warnings.append(
                    "finite_dimension_normalization_inferred_but_missing_from_main_tex"
                )
            if use_paper_operational_gamma:
                warnings.append(
                    "paper_multiplier_and_displayed_1.98_bit_ledger_differ_from_exact_b2_target"
                )
            return ProfileEstimate(
                profile=self.profile,
                normalization=self.normalization,
                normalization_scope=self.normalization_scope,
                volume_ratio=+profile_volume_ratio(
                    self.profile, normalization=self.normalization
                ),
                candidate_cap=self.profile.candidate_cap,
                sampler_budget_bits=+self.sampler_budget_bits,
                target_deficit=+target,
                log_delta_hat=+log_delta,
                delta_hat=+delta,
                order_search=search,
                active_moment_order=active_record.order,
                active_log_moment_upper=+active_record.log_moment_upper,
                gamma_hat_computed=+computed_gamma,
                operational_gamma_hat=+gamma,
                operational_tau=+tau,
                deficit_upper=+deficit,
                r_infinity_upper=+r_infinity,
                p_corr_lower=+p_corr,
                norm_tail_exponent=+norm_exponent,
                p_norm_lower=+p_norm,
                total_mean_trials_upper=+total_trials,
                sampler_loss_bits=+sampler_loss,
                warnings=tuple(warnings),
            )
