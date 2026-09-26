"""Falcon-style fast Fourier LDL decomposition and Klein sampling.

This module implements the ring-efficient path (``path B`` in the project
plan).  It deliberately keeps the one-dimensional calls visible: every leaf
sample produces a :class:`FFSampleTraceEntry`.  Those entries are sufficient
to evaluate the KGPV correction factor used by Falcon++ because the
normalising mass of a shifted integer Gaussian is periodic modulo integers.

Conventions
-----------

* Polynomial vectors and matrices use Falcon's *row-basis* convention.
* ``sigma`` always means ordinary standard deviation.  The paper's
  cryptographic Gaussian parameter is ``s = sqrt(2*pi) * sigma``.
* A Gram leaf stores ``gs_norm_squared``.  Its sampler width is therefore
  ``sigma / sqrt(gs_norm_squared)``.
* FFT values are represented by the sequence type used by :mod:`falconpp.fft`.

The implementation is a readable reference implementation, not a
constant-time implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from typing import Any, Callable, Protocol, Sequence, TypeAlias

from . import fft as _fft


FFTVector: TypeAlias = Sequence[Any]
FFTMatrix2: TypeAlias = Sequence[Sequence[FFTVector]]


class IntegerGaussianSampler(Protocol):
    """Protocol implemented by :class:`DiscreteGaussianSampler`.

    ``rng`` is intentionally opaque here.  The Gaussian module owns the
    random-source convention; this module merely passes it through.
    """

    def sample(self, center: float, sigma: float, rng: Any = None) -> int:
        """Draw from ``D_{Z, sigma, center}`` (standard-deviation form)."""


@dataclass(frozen=True, slots=True)
class FFTLDLLeaf:
    """A scalar diagonal leaf of a fast-Fourier LDL tree."""

    gs_norm_squared: Any
    gs_norm: Any
    sigma: Any


FFTLDLChild: TypeAlias = "FFTLDLNode | FFTLDLLeaf"


@dataclass(frozen=True, slots=True)
class FFTLDLNode:
    """One node of Falcon's recursive ffLDL tree.

    ``left`` decomposes the first diagonal polynomial and ``right`` the
    second.  Sampling consequently visits ``right`` before ``left``, exactly
    as backward Klein sampling visits the last coordinate first.
    """

    l10: tuple[Any, ...]
    left: FFTLDLChild
    right: FFTLDLChild
    degree: int
    backend: str = "float"
    dps: int = 100


@dataclass(frozen=True, slots=True)
class FFSampleTraceEntry:
    """One one-dimensional conditional Gaussian used by ffSampling.

    The correction factor uses ``center`` modulo one and ``sigma``.  ``path``
    is a stable description of the recursion leaf (``1`` means the right
    subtree and ``0`` the left subtree); ``sequence_index`` records actual
    sampling order.
    """

    sequence_index: int
    path: tuple[int, ...]
    center: Any
    sigma: Any
    value: int
    gs_norm: Any

    @property
    def paper_width(self) -> Any:
        """Return ``s_i = sqrt(2*pi)*sigma_i`` used in the paper."""

        if type(self.sigma).__module__.startswith("mpmath"):
            import mpmath as mp

            return mp.sqrt(2 * mp.pi) * self.sigma
        return math.sqrt(2.0 * math.pi) * self.sigma


@dataclass(frozen=True, slots=True)
class FFSamplingResult:
    """Output coefficients in FFT form together with the complete trace."""

    z_fft: tuple[tuple[Any, ...], tuple[Any, ...]]
    trace: tuple[FFSampleTraceEntry, ...]

    @property
    def sample_count(self) -> int:
        return len(self.trace)


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _as_tuple(values: FFTVector) -> tuple[Any, ...]:
    return tuple(values)


def _add_fft(left: FFTVector, right: FFTVector) -> list[Any]:
    if len(left) != len(right):
        raise ValueError("FFT operands have different degrees")
    return [a + b for a, b in zip(left, right, strict=True)]


def _sub_fft(left: FFTVector, right: FFTVector) -> list[Any]:
    if len(left) != len(right):
        raise ValueError("FFT operands have different degrees")
    return [a - b for a, b in zip(left, right, strict=True)]


def _mul_fft(left: FFTVector, right: FFTVector) -> list[Any]:
    if len(left) != len(right):
        raise ValueError("FFT operands have different degrees")
    return [a * b for a, b in zip(left, right, strict=True)]


def _div_fft(left: FFTVector, right: FFTVector) -> list[Any]:
    if len(left) != len(right):
        raise ValueError("FFT operands have different degrees")
    return [a / b for a, b in zip(left, right, strict=True)]


def _adj_fft(values: FFTVector) -> list[Any]:
    return [value.conjugate() if hasattr(value, "conjugate") else value for value in values]


def _split_fft(values: FFTVector, *, backend: str, dps: int) -> tuple[list[Any], list[Any]]:
    if backend == "mpmath":
        return _fft.split_fft_high_precision(values, dps=dps)
    if backend != "float":
        raise ValueError("backend must be 'float' or 'mpmath'")
    return _fft.split_fft(values)


def _merge_fft(
    parts: Sequence[FFTVector], *, backend: str, dps: int
) -> list[Any]:
    if backend == "mpmath":
        return _fft.merge_fft_high_precision(parts, dps=dps)
    if backend != "float":
        raise ValueError("backend must be 'float' or 'mpmath'")
    return _fft.merge_fft(parts)


def _validate_gram(gram: FFTMatrix2) -> int:
    if len(gram) != 2 or any(len(row) != 2 for row in gram):
        raise ValueError("ffLDL requires a 2 by 2 Gram matrix")
    degree = len(gram[0][0])
    if not _is_power_of_two(degree):
        raise ValueError("FFT polynomial degree must be a positive power of two")
    if any(len(gram[i][j]) != degree for i in range(2) for j in range(2)):
        raise ValueError("all Gram-matrix entries must have the same degree")
    return degree


def gram_fft(
    basis_fft: FFTMatrix2,
    *,
    backend: str = "float",
    dps: int = 100,
) -> list[list[list[Any]]]:
    """Return ``B * B^*`` for a 2-by-2 row basis in FFT form.

    Selecting ``backend="mpmath"`` keeps all pointwise products inside one
    ``mp.workdps(dps)`` context.  This is necessary because mpmath arithmetic
    uses the active context even when its operands were originally created at
    a higher precision.
    """

    if len(basis_fft) != 2 or any(len(row) != 2 for row in basis_fft):
        raise ValueError("the Falcon NTRU basis must be 2 by 2")
    degree = len(basis_fft[0][0])
    if not _is_power_of_two(degree):
        raise ValueError("FFT polynomial degree must be a positive power of two")
    if any(len(basis_fft[i][j]) != degree for i in range(2) for j in range(2)):
        raise ValueError("all basis entries must have the same FFT degree")

    if backend not in {"float", "mpmath"}:
        raise ValueError("backend must be 'float' or 'mpmath'")
    if isinstance(dps, bool) or not isinstance(dps, int) or dps < 30:
        raise ValueError("dps must be an integer of at least 30")

    def compute() -> list[list[list[Any]]]:
        result: list[list[list[Any]]] = [[[], []], [[], []]]
        for i in range(2):
            for j in range(2):
                term0 = _mul_fft(basis_fft[i][0], _adj_fft(basis_fft[j][0]))
                term1 = _mul_fft(basis_fft[i][1], _adj_fft(basis_fft[j][1]))
                result[i][j] = _add_fft(term0, term1)
        return result

    if backend == "mpmath":
        import mpmath as mp

        with mp.workdps(dps):
            return compute()
    return compute()


def ldl_fft(
    gram: FFTMatrix2,
    *,
    backend: str = "float",
    dps: int = 100,
) -> tuple[list[Any], list[Any], list[Any]]:
    """Return ``(l10, d00, d11)`` for a Hermitian 2-by-2 Gram matrix."""

    _validate_gram(gram)
    if backend not in {"float", "mpmath"}:
        raise ValueError("backend must be 'float' or 'mpmath'")
    if isinstance(dps, bool) or not isinstance(dps, int) or dps < 30:
        raise ValueError("dps must be an integer of at least 30")

    def compute() -> tuple[list[Any], list[Any], list[Any]]:
        d00 = list(gram[0][0])
        l10 = _div_fft(gram[1][0], d00)
        correction = _mul_fft(_mul_fft(l10, _adj_fft(l10)), d00)
        d11 = _sub_fft(gram[1][1], correction)
        return l10, d00, d11

    if backend == "mpmath":
        import mpmath as mp

        with mp.workdps(dps):
            return compute()
    return compute()


def _real_and_imag(value: Any) -> tuple[Any, Any]:
    if hasattr(value, "real"):
        return value.real, value.imag
    return value, 0


def _uses_mpmath(value: Any) -> bool:
    """Return whether ``value`` is one of mpmath's scalar types."""

    return type(value).__module__.startswith("mpmath")


def _is_finite(value: Any) -> bool:
    if _uses_mpmath(value):
        import mpmath as mp

        return bool(mp.isfinite(value))
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _imaginary_residue_is_small(real: Any, imag: Any, tolerance: float) -> bool:
    """Compare a relative imaginary residue in its native arithmetic."""

    if _uses_mpmath(real) or _uses_mpmath(imag):
        import mpmath as mp

        if not mp.isfinite(real) or not mp.isfinite(imag):
            return False
        scale = max(mp.mpf(1), abs(real))
        return bool(abs(imag) <= mp.mpf(repr(tolerance)) * scale)
    try:
        if not math.isfinite(float(real)) or not math.isfinite(float(imag)):
            return False
        scale = max(1.0, abs(float(real)))
        return abs(float(imag)) <= tolerance * scale
    except (TypeError, ValueError, OverflowError):
        return False


def _positive_real(value: Any, *, tolerance: float) -> Any:
    real, imag = _real_and_imag(value)
    if not _imaginary_residue_is_small(real, imag, tolerance):
        raise ArithmeticError(
            f"ffLDL leaf is not real (imaginary residue {imag!r})"
        )
    if not _is_finite(real) or real <= 0:
        raise ArithmeticError(f"ffLDL Gram leaf is not positive: {real!r}")
    return real


def _sqrt(value: Any, *, backend: str) -> Any:
    if backend == "mpmath":
        import mpmath as mp

        return mp.sqrt(value)
    return math.sqrt(float(value))


def build_ffldl_tree(
    gram: FFTMatrix2,
    sigma: Any,
    *,
    backend: str = "float",
    dps: int = 100,
    reality_tolerance: float = 2.0**-35,
) -> FFTLDLNode:
    """Build and normalise a Falcon ffLDL tree.

    Args:
        gram: Hermitian 2-by-2 Gram matrix in FFT representation.
        sigma: Global signing standard deviation (not the paper's ``s``).
        reality_tolerance: Relative tolerance for imaginary roundoff at leaves.

    Returns:
        An immutable tree whose scalar leaves contain the local standard
        deviations used by the one-dimensional sampler.
    """

    degree = _validate_gram(gram)
    if backend not in {"float", "mpmath"}:
        raise ValueError("backend must be 'float' or 'mpmath'")
    if isinstance(dps, bool) or not isinstance(dps, int) or dps < 30:
        raise ValueError("dps must be an integer of at least 30")
    if not math.isfinite(reality_tolerance) or reality_tolerance < 0.0:
        raise ValueError("reality_tolerance must be finite and non-negative")

    # This variable is deliberately populated inside the selected arithmetic
    # context below.  Calling mp.mpf(existing_mpf) at the process-global
    # default precision can otherwise round a caller's high-precision width
    # before the ffLDL recursion even starts.
    sigma_value: Any

    def make_leaf(value: Any) -> FFTLDLLeaf:
        squared = _positive_real(value, tolerance=reality_tolerance)
        norm = _sqrt(squared, backend=backend)
        return FFTLDLLeaf(
            gs_norm_squared=squared,
            gs_norm=norm,
            sigma=sigma_value / norm,
        )

    def recurse(local_gram: FFTMatrix2) -> FFTLDLNode:
        local_degree = _validate_gram(local_gram)
        l10, d00, d11 = ldl_fft(local_gram, backend=backend, dps=dps)
        if local_degree == 1:
            return FFTLDLNode(
                l10=_as_tuple(l10),
                left=make_leaf(d00[0]),
                right=make_leaf(d11[0]),
                degree=1,
                backend=backend,
                dps=dps,
            )

        d00_lo, d00_hi = _split_fft(d00, backend=backend, dps=dps)
        d11_lo, d11_hi = _split_fft(d11, backend=backend, dps=dps)
        left_gram = (
            (d00_lo, d00_hi),
            (_adj_fft(d00_hi), d00_lo),
        )
        right_gram = (
            (d11_lo, d11_hi),
            (_adj_fft(d11_hi), d11_lo),
        )
        return FFTLDLNode(
            l10=_as_tuple(l10),
            left=recurse(left_gram),
            right=recurse(right_gram),
            degree=local_degree,
            backend=backend,
            dps=dps,
        )

    if backend == "mpmath":
        import mpmath as mp

        with mp.workdps(dps):
            try:
                sigma_value = mp.mpf(sigma)
            except (TypeError, ValueError) as exc:
                raise ValueError("sigma must be finite and positive") from exc
            if not mp.isfinite(sigma_value) or sigma_value <= 0:
                raise ValueError("sigma must be finite and positive")
            tree = recurse(gram)
    else:
        try:
            sigma_value = float(sigma)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("sigma must be finite and positive") from exc
        if not math.isfinite(sigma_value) or sigma_value <= 0:
            raise ValueError("sigma must be finite and positive")
        tree = recurse(gram)
    if tree.degree != degree:  # Defensive guard against an FFT API mismatch.
        raise AssertionError("internal ffLDL degree mismatch")
    return tree


def build_sampler_tree(
    basis_fft: FFTMatrix2,
    sigma: Any,
    *,
    backend: str = "float",
    dps: int = 100,
    reality_tolerance: float = 2.0**-35,
) -> FFTLDLNode:
    """Convenience wrapper: construct ``Gram(B)`` and its sampler tree."""

    return build_ffldl_tree(
        gram_fft(basis_fft, backend=backend, dps=dps),
        sigma,
        backend=backend,
        dps=dps,
        reality_tolerance=reality_tolerance,
    )


def _real_center(value: Any, *, tolerance: float) -> Any:
    real, imag = _real_and_imag(value)
    if not _imaginary_residue_is_small(real, imag, tolerance):
        raise ArithmeticError(
            f"scalar conditional center is not real (imaginary residue {imag!r})"
        )
    center = real
    if not _is_finite(center):
        raise ArithmeticError("conditional center is not finite")
    return center


def _draw_integer(
    sampler: IntegerGaussianSampler | Callable[[float, float, Any], int],
    center: Any,
    sigma: Any,
    rng: Any,
) -> int:
    if hasattr(sampler, "sample"):
        value = sampler.sample(center, sigma, rng)  # type: ignore[union-attr]
    else:
        value = sampler(center, sigma, rng)  # type: ignore[operator]
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("the one-dimensional sampler must return an integer")
    return int(value)


def ff_sample(
    target_fft: Sequence[FFTVector],
    tree: FFTLDLNode,
    sampler: IntegerGaussianSampler | Callable[[float, float, Any], int],
    rng: Any = None,
    *,
    reality_tolerance: float = 2.0**-32,
) -> FFSamplingResult:
    """Run Falcon's recursive FFT/Klein sampler and retain every leaf call.

    ``target_fft`` contains the two ring coordinates of the target in the
    basis-coordinate representation.  The returned pair has the same shape.
    The caller obtains the lattice vector by multiplying ``z_fft`` by its
    expanded NTRU basis, as in Falcon.
    """

    if len(target_fft) != 2:
        raise ValueError("ffSampling target must have two ring coordinates")
    if len(target_fft[0]) != tree.degree or len(target_fft[1]) != tree.degree:
        raise ValueError("target FFT degree and ffLDL tree degree differ")
    if not math.isfinite(reality_tolerance) or reality_tolerance < 0.0:
        raise ValueError("reality_tolerance must be finite and non-negative")

    trace: list[FFSampleTraceEntry] = []

    def recurse(
        t0: FFTVector,
        t1: FFTVector,
        node: FFTLDLNode,
        path: tuple[int, ...],
    ) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        if len(t0) != node.degree or len(t1) != node.degree:
            raise ValueError("malformed ffLDL tree or target split")

        if node.degree == 1:
            if not isinstance(node.left, FFTLDLLeaf) or not isinstance(
                node.right, FFTLDLLeaf
            ):
                raise ValueError("degree-one ffLDL node must have scalar leaves")

            center1 = _real_center(t1[0], tolerance=reality_tolerance)
            z1 = _draw_integer(sampler, center1, node.right.sigma, rng)
            trace.append(
                FFSampleTraceEntry(
                    sequence_index=len(trace),
                    path=path + (1,),
                    center=center1,
                    sigma=node.right.sigma,
                    value=z1,
                    gs_norm=node.right.gs_norm,
                )
            )

            residual1 = [t1[0] - z1]
            adjusted0 = _add_fft(t0, _mul_fft(residual1, node.l10))
            center0 = _real_center(adjusted0[0], tolerance=reality_tolerance)
            z0 = _draw_integer(sampler, center0, node.left.sigma, rng)
            trace.append(
                FFSampleTraceEntry(
                    sequence_index=len(trace),
                    path=path + (0,),
                    center=center0,
                    sigma=node.left.sigma,
                    value=z0,
                    gs_norm=node.left.gs_norm,
                )
            )
            return (z0,), (z1,)

        if isinstance(node.left, FFTLDLLeaf) or isinstance(node.right, FFTLDLLeaf):
            raise ValueError("non-scalar ffLDL node must have recursive children")

        t1_lo, t1_hi = _split_fft(t1, backend=node.backend, dps=node.dps)
        z1_lo, z1_hi = recurse(t1_lo, t1_hi, node.right, path + (1,))
        z1 = _as_tuple(
            _merge_fft((z1_lo, z1_hi), backend=node.backend, dps=node.dps)
        )

        adjusted0 = _add_fft(t0, _mul_fft(_sub_fft(t1, z1), node.l10))
        t0_lo, t0_hi = _split_fft(
            adjusted0, backend=node.backend, dps=node.dps
        )
        z0_lo, z0_hi = recurse(t0_lo, t0_hi, node.left, path + (0,))
        z0 = _as_tuple(
            _merge_fft((z0_lo, z0_hi), backend=node.backend, dps=node.dps)
        )
        return z0, z1

    if tree.backend == "mpmath":
        import mpmath as mp

        with mp.workdps(tree.dps):
            z0, z1 = recurse(target_fft[0], target_fft[1], tree, ())
    else:
        z0, z1 = recurse(target_fft[0], target_fft[1], tree, ())
    expected = 2 * tree.degree
    if len(trace) != expected:
        raise AssertionError(
            f"ffSampling produced {len(trace)} leaf calls, expected {expected}"
        )
    return FFSamplingResult(z_fft=(z0, z1), trace=tuple(trace))


# Descriptive alias used by callers that want to emphasize trace retention.
ff_sample_with_trace = ff_sample


def iter_leaves(tree: FFTLDLNode) -> tuple[FFTLDLLeaf, ...]:
    """Return scalar leaves in structural left-to-right order."""

    leaves: list[FFTLDLLeaf] = []

    def visit(child: FFTLDLChild) -> None:
        if isinstance(child, FFTLDLLeaf):
            leaves.append(child)
            return
        visit(child.left)
        visit(child.right)

    visit(tree)
    return tuple(leaves)


__all__ = [
    "FFTLDLLeaf",
    "FFLDLNode",
    "FFSampleTraceEntry",
    "FFSamplingResult",
    "IntegerGaussianSampler",
    "build_ffldl_tree",
    "build_sampler_tree",
    "ff_sample",
    "ff_sample_with_trace",
    "gram_fft",
    "iter_leaves",
    "ldl_fft",
]
