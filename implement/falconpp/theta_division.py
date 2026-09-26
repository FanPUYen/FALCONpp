"""Certified, refinable ThetaDiv products for frozen finite trace scalars.

For standard-deviation widths, ``r = exp(-2*pi**2*sigma**2)`` and
``rho_sigma(c)/rho_sigma(0) = prod_i(1-a_i*sin(pi*c)**2)``, where
``a_i=4*r**(2*i-1)/(1+r**(2*i-1))**2``.  For paper rho widths the
exponent is ``-pi*s**2``; no rounded width conversion is performed.

This module uses integer fixed-point intervals, not guessed floating-point
ulps.  Machin bounds enclose pi; alternating Taylor bounds enclose sine;
Decimal's correctly rounded exponential is enclosed by the existing helper.
The omitted product is enclosed with the independently valid union bound
``S_K = 4*r**(2*K+1)/(1-r*r)``.  In particular no use is made of the
incorrect log inequality in the local weak-smoothing Lemma 17 proof.

Only width-dependent data are cached, boundedly and on a context belonging
to one key.  Centers, traces and samples are never retained.  These bounds
certify correction for the *stored* scalars, not upstream FFT error.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Iterable

from .gaussian import exact_real_fraction


def _ceil_div(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


def _floor_scaled(value: Fraction, scale: int) -> int:
    return value.numerator * scale // value.denominator


def _ceil_scaled(value: Fraction, scale: int) -> int:
    return _ceil_div(value.numerator * scale, value.denominator)


def _bits(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError("bits must be a positive integer")
    return value


def _reduced_center(center: Fraction) -> Fraction:
    # Reduction is exact even for huge integer parts or dyadic denominators.
    residue = center.numerator % center.denominator
    return Fraction(min(residue, center.denominator - residue), center.denominator)


@dataclass(frozen=True, slots=True)
class _WidthTable:
    coefficients: tuple[tuple[int, int], ...]
    tail_upper: int
    primal_coefficient: tuple[Fraction, Fraction] | None = None
    primal_radius: int = 0
    zero_mass: tuple[int, int] | None = None


class ThetaDivContext:
    """Per-key bounded cache and certified probability evaluator.

    ``bits`` requests an absolute interval width at most ``2**(-bits)``.
    Refinement is unbounded: increased requests never hit a precision cap.
    The finite ThetaDiv term cap selects the independent positive-series
    backend, rather than weakening an error bound.
    """

    def __init__(self, *, max_cached_widths: int = 4096, max_terms: int = 256):
        for name, value in (("max_cached_widths", max_cached_widths), ("max_terms", max_terms)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.max_cached_widths = max_cached_widths
        self.max_terms = max_terms
        self._widths: OrderedDict[tuple[str, Fraction, int], _WidthTable] = OrderedDict()
        self._pi: OrderedDict[int, tuple[int, int]] = OrderedDict()
        self._stats = dict(ratio_calls=0, trace_calls=0, precomputations=0,
                           cache_hits=0, terms_evaluated=0, fallback_calls=0,
                           max_terms_used=0, precision_refinements=0)

    def stats_snapshot(self) -> dict[str, int]:
        """Nonsecret aggregate counters only, without cache keys/contents."""
        return {**self._stats, "cached_widths": len(self._widths)}

    diagnostics = stats_snapshot

    def clear_cache(self) -> None:
        self._widths.clear()
        self._pi.clear()

    def _pi_fixed(self, work_bits: int) -> tuple[int, int]:
        from .correction import _pi_bounds

        if work_bits not in self._pi:
            scale = 1 << work_bits
            lower, upper = _pi_bounds(work_bits + 12)
            self._pi[work_bits] = (_floor_scaled(lower, scale), _ceil_scaled(upper, scale))
            if len(self._pi) > 16:
                self._pi.popitem(last=False)
        self._pi.move_to_end(work_bits)
        return self._pi[work_bits]

    @staticmethod
    def _exp_negative_fixed(lower: Fraction, upper: Fraction, work_bits: int) -> tuple[int, int]:
        """Enclose exp(-a), including enormous a without huge denominators."""
        from .correction import _exp_fraction_bounds

        if lower < 0 or lower > upper:
            raise ArithmeticError("invalid nonnegative exponential argument")
        scale = 1 << work_bits
        if upper == 0:
            return scale, scale
        # e >= 2 gives exp(-a) <= 2**(-a).  These cutoffs are inequalities,
        # not Decimal underflow.  They remain refinable for any fixed a.
        cutoff = work_bits + 4
        if lower >= cutoff:
            return 0, 1
        if upper >= cutoff:
            _, exp_upper = _exp_fraction_bounds(-lower, -lower, bits=work_bits + 16)
            return 0, min(scale, _ceil_scaled(exp_upper, scale))
        exp_lower, exp_upper = _exp_fraction_bounds(-upper, -lower, bits=work_bits + 16)
        return max(0, _floor_scaled(exp_lower, scale)), min(scale, _ceil_scaled(exp_upper, scale))

    def _sin_squared_fixed(self, center: Fraction, work_bits: int) -> tuple[int, int]:
        """Enclose sin(pi*c)^2 using exact modulo and alternating Taylor."""
        scale = 1 << work_bits
        if center == 0:
            return 0, 0
        if center == Fraction(1, 2):
            return scale, scale
        pi_lower, pi_upper = self._pi_fixed(work_bits)
        x_lower = pi_lower * center.numerator // center.denominator
        x_upper = _ceil_div(pi_upper * center.numerator, center.denominator)
        square_lower = x_lower * x_lower // scale
        square_upper = _ceil_div(x_upper * x_upper, scale)
        # True x <= pi/2; therefore the exact Taylor term magnitudes decrease
        # from the first term, since x^2 < 6.  Endpoint dependency widening is
        # handled by interval operations, with no sine monotonicity assumption.
        term_lower, term_upper = x_lower, x_upper
        total_lower, total_upper = term_lower, term_upper
        index = 0
        while True:
            divisor = (2 * index + 2) * (2 * index + 3) * scale
            next_lower = term_lower * square_lower // divisor
            next_upper = _ceil_div(term_upper * square_upper, divisor)
            if next_upper <= 1:
                if index % 2 == 0:  # next term has negative sign
                    total_lower -= next_upper
                else:
                    total_upper += next_upper
                break
            index += 1
            if index % 2:
                total_lower -= next_upper
                total_upper -= next_lower
            else:
                total_lower += next_lower
                total_upper += next_upper
            term_lower, term_upper = next_lower, next_upper
        total_lower = max(0, total_lower)
        total_upper = min(scale, total_upper)
        return total_lower * total_lower // scale, min(scale, _ceil_div(total_upper * total_upper, scale))

    def _primal_mass(self, center: Fraction, coefficient: tuple[Fraction, Fraction],
                     radius: int, work_bits: int) -> tuple[int, int]:
        """Positive primal sum with a certified <=1 fixed-point ulp tail."""
        lower = upper = 0
        for integer in range(-radius, radius + 1):
            square = (integer - center) ** 2
            term_lower, term_upper = self._exp_negative_fixed(
                coefficient[0] * square, coefficient[1] * square, work_bits)
            lower += term_lower
            upper += term_upper
        # Radius construction proves the omitted two-tail mass <= 2^-work_bits.
        return lower, upper + 1

    def _primal_table(self, coefficient: tuple[Fraction, Fraction], work_bits: int) -> _WidthTable:
        radius = 1
        coefficient_lower = coefficient[0]
        while (coefficient_lower * Fraction(2 * radius + 1, 2) ** 2 < work_bits + 8
               or coefficient_lower * (2 * radius + 2) < 1):
            radius *= 2
        # Two tails <=2 exp(-a(R+1/2)^2)/(1-exp(-a(2R+2)))
        # <=4*2^(-work_bits-8) <=2^(-work_bits), using e>=2.
        zero_lower, zero_upper = self._primal_mass(Fraction(0), coefficient, radius, work_bits)
        zero_lower = max(1 << work_bits, zero_lower)  # the z=0 term is exactly 1
        return _WidthTable((), 0, coefficient, radius, (zero_lower, zero_upper))

    def _width_table(self, width: Fraction, convention: str, work_bits: int) -> _WidthTable:
        key = convention, width, work_bits
        existing = self._widths.get(key)
        if existing is not None:
            self._widths.move_to_end(key)
            self._stats["cache_hits"] += 1
            return existing
        scale = 1 << work_bits
        pi_lower_i, pi_upper_i = self._pi_fixed(work_bits)
        pi_lower, pi_upper = Fraction(pi_lower_i, scale), Fraction(pi_upper_i, scale)
        width_squared = width * width
        if convention == "sigma":
            exponent = 2 * pi_lower * pi_lower * width_squared, 2 * pi_upper * pi_upper * width_squared
            coefficient = Fraction(1, 2) / width_squared
            primal_coefficient = coefficient, coefficient
        else:
            exponent = pi_lower * width_squared, pi_upper * width_squared
            primal_coefficient = pi_lower / width_squared, pi_upper / width_squared
        r_lower, r_upper = self._exp_negative_fixed(*exponent, work_bits)
        coefficients: list[tuple[int, int]] = []
        if r_upper >= scale or r_upper * 4 > scale * 3:
            table = self._primal_table(primal_coefficient, work_bits)
        else:
            step_lower = r_lower * r_lower // scale
            step_upper = _ceil_div(r_upper * r_upper, scale)
            t_lower, t_upper = r_lower, r_upper
            denominator = scale * scale - r_upper * r_upper
            tail_upper = _ceil_div(4 * t_upper * scale * scale, denominator)
            while tail_upper > 16 and len(coefficients) < self.max_terms:
                # a(t) is increasing for t in [0,1], so endpoint evaluation
                # is tighter than generic interval division and still exact.
                a_lower = 4 * t_lower * scale * scale // (scale + t_lower) ** 2
                a_upper = _ceil_div(4 * t_upper * scale * scale, (scale + t_upper) ** 2)
                coefficients.append((max(0, a_lower), min(scale, a_upper)))
                t_lower = t_lower * step_lower // scale
                t_upper = _ceil_div(t_upper * step_upper, scale)
                tail_upper = _ceil_div(4 * t_upper * scale * scale, denominator)
            if tail_upper > 16:
                table = self._primal_table(primal_coefficient, work_bits)
            else:
                table = _WidthTable(tuple(coefficients), tail_upper)
        self._widths[key] = table
        self._stats["precomputations"] += 1
        self._stats["max_terms_used"] = max(self._stats["max_terms_used"], len(table.coefficients))
        if len(self._widths) > self.max_cached_widths:
            self._widths.popitem(last=False)
        return table

    def _ratio_fixed(self, width: Fraction, center: Fraction, convention: str,
                     work_bits: int) -> tuple[int, int]:
        self._stats["ratio_calls"] += 1
        scale = 1 << work_bits
        center = _reduced_center(center)
        if center == 0:
            return scale, scale
        table = self._width_table(width, convention, work_bits)
        if table.primal_coefficient is not None:
            self._stats["fallback_calls"] += 1
            mass_lower, mass_upper = self._primal_mass(center, table.primal_coefficient,
                                                       table.primal_radius, work_bits)
            zero_lower, zero_upper = table.zero_mass  # type: ignore[misc]
            return max(0, mass_lower * scale // zero_upper), min(scale, _ceil_div(mass_upper * scale, zero_lower))
        if not table.coefficients:
            return max(0, scale - table.tail_upper), scale
        sin_lower, sin_upper = self._sin_squared_fixed(center, work_bits)
        product_lower = product_upper = scale
        for a_lower, a_upper in table.coefficients:
            factor_lower = max(0, scale - _ceil_div(a_upper * sin_upper, scale))
            factor_upper = min(scale, scale - a_lower * sin_lower // scale)
            product_lower = product_lower * factor_lower // scale
            product_upper = _ceil_div(product_upper * factor_upper, scale)
        self._stats["terms_evaluated"] += len(table.coefficients)
        product_lower = product_lower * max(0, scale - table.tail_upper) // scale
        return product_lower, min(scale, product_upper)

    def ratio_bounds(self, sigma: Any, center: Any, *, bits: int = 80) -> tuple[Fraction, Fraction]:
        """Enclose one standard-deviation mass ratio with exact input semantics."""
        bits = _bits(bits)
        width = exact_real_fraction(sigma, name="sigma")
        center_q = exact_real_fraction(center, name="center")
        if width <= 0:
            raise ValueError("sigma must be positive")
        work_bits = bits + 32
        while True:
            lower, upper = self._ratio_fixed(width, center_q, "sigma", work_bits)
            if upper - lower <= 1 << (work_bits - bits):
                return Fraction(lower, 1 << work_bits), Fraction(upper, 1 << work_bits)
            work_bits += 32
            self._stats["precision_refinements"] += 1

    def prepare_widths(self, widths: Iterable[Any], *, bits: int = 80,
                       multiplier: Any | None = None) -> None:
        """Precompute width tables, optionally matching a full trace budget.

        With ``multiplier`` supplied, provide all coordinate widths (including
        repeats) so their count and multiplier match trace_probability_bounds.
        Without it this primes individual ratio_bounds calls.
        """
        frozen_widths = tuple(widths)
        work_bits = _bits(bits) + 32
        if multiplier is not None:
            multiplier_q = exact_real_fraction(multiplier, name="multiplier")
            if multiplier_q < 0:
                raise ValueError("multiplier must be nonnegative")
            amplification = max(0, multiplier_q.numerator.bit_length() - multiplier_q.denominator.bit_length() + 1)
            work_bits += len(frozen_widths).bit_length() + amplification
        for sigma in frozen_widths:
            width = exact_real_fraction(sigma, name="sigma")
            if width <= 0:
                raise ValueError("sigma must be positive")
            self._width_table(width, "sigma", work_bits)

    def trace_probability_bounds(self, trace: Iterable[Any], multiplier: Any,
                                 *, bits: int = 80, global_rho_s: Any | None = None
                                 ) -> tuple[Fraction, Fraction]:
        """Enclose min(1, multiplier * product of all trace ratios).

        This call retains no trace.  The work precision explicitly budgets
        coordinate accumulation and multiplier amplification, and a final
        interval-width check triggers more precision if necessary.
        """
        from .correction import _prepare_certified_trace

        bits = _bits(bits)
        multiplier_q = exact_real_fraction(multiplier, name="multiplier")
        if multiplier_q < 0:
            raise ValueError("multiplier must be nonnegative")
        prepared = _prepare_certified_trace(trace, global_rho_s=global_rho_s)
        self._stats["trace_calls"] += 1
        if multiplier_q == 0:
            return Fraction(0), Fraction(0)
        amplification = max(0, multiplier_q.numerator.bit_length() - multiplier_q.denominator.bit_length() + 1)
        work_bits = bits + 32 + len(prepared).bit_length() + amplification
        while True:
            scale = 1 << work_bits
            lower = upper = scale
            for convention, center, width in prepared:
                ratio_lower, ratio_upper = self._ratio_fixed(width, center, convention, work_bits)
                lower = lower * ratio_lower // scale
                upper = _ceil_div(upper * ratio_upper, scale)
            lower = min(scale, lower * multiplier_q.numerator // multiplier_q.denominator)
            upper = min(scale, _ceil_div(upper * multiplier_q.numerator, multiplier_q.denominator))
            if upper - lower <= 1 << (work_bits - bits):
                return Fraction(lower, scale), Fraction(upper, scale)
            work_bits += 32
            self._stats["precision_refinements"] += 1
