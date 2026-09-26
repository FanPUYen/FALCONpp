"""Numerical entropy enclosures, conditional on the manuscript's profile bounds.

Run from any directory with python.
Outputs are written under results/. No signatures or keys are sampled.
"""
from pathlib import Path
import hashlib
import json
import sys
import mpmath as mp

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
SOURCE = ROOT / 'implement/falconpp/parameters.py'
sys.path.insert(0, str(ROOT / 'implement'))
from falconpp.parameters import FALCONPP_I_1245, FALCONPP_1024_117


def endpoints(value):
    return mp.mpf(value.a), mp.mpf(value.b)


def serialize(value):
    lo, hi = endpoints(value)
    return {'lower': mp.nstr(lo, 35), 'upper': mp.nstr(hi, 35)}


def calculate(n, sigma_text, epsilon_text, exponent_text, dps, cutoff):
    mp.mp.dps = dps + 20
    mp.iv.dps = dps
    iv = mp.iv
    sigma = iv.mpf(sigma_text)
    limit = int(mp.ceil(cutoff * mp.mpf(sigma_text)))
    sums = {0: iv.mpf(1), 2: iv.mpf(0), 4: iv.mpf(0)}
    for k in range(1, limit + 1):
        weight = 2 * iv.exp(-iv.mpf(k * k) / (2 * sigma**2))
        for order in sums:
            sums[order] += k**order * weight

    # For k >= M+1, the ratio of successive k^m exp(-k^2/(2 sigma^2))
    # is <= ((M+2)/(M+1))^m exp(-(2M+3)/(2 sigma^2)).
    # This geometric majorant bounds both infinite tails, for m = 0,2,4.
    tails = {}
    for order in sums:
        first = (2 * iv.mpf(limit + 1)**order
                 * iv.exp(-iv.mpf((limit + 1)**2) / (2 * sigma**2)))
        ratio = ((iv.mpf(limit + 2) / (limit + 1))**order
                 * iv.exp(-iv.mpf(2 * limit + 3) / (2 * sigma**2)))
        assert endpoints(ratio)[1] < 1
        upper_tail = endpoints(first / (1 - ratio))[1]
        tails[str(order)] = mp.nstr(upper_tail, 25)
        sums[order] += iv.mpf([0, upper_tail])

    ln2 = iv.ln(2)
    z = sums[0]
    second = sums[2] / z
    fourth = sums[4] / z
    h = n * (iv.ln(z) / ln2 + second / (2 * sigma**2 * ln2))
    v = n * (fourth - second**2) / (4 * sigma**4 * ln2**2)
    pnorm_lower = 1 - iv.exp(-iv.mpf(exponent_text))
    # Stable expression for c-1, including the tiny r_infinity-1 term.
    c_minus_one = ((iv.mpf(epsilon_text) + iv.exp(-iv.mpf(exponent_text)))
                   / pnorm_lower)
    correction = iv.sqrt(c_minus_one * v)
    kl_upper = iv.ln(1 + c_minus_one) / ln2
    lower_expression = h - correction - kl_upper
    upper_expression = h + correction
    lower = endpoints(lower_expression)[0]
    upper = endpoints(upper_expression)[1]
    gaussian_approx = n * iv.ln(sigma * iv.sqrt(2 * iv.pi * iv.e)) / ln2
    nominal_interval = (iv.mpf([lower, upper]) / 8) + 41
    ceil_lower = int(mp.ceil(endpoints(nominal_interval)[0]))
    ceil_upper = int(mp.ceil(endpoints(nominal_interval)[1]))
    assert ceil_lower == ceil_upper
    return {
        'n': n, 'sigma': sigma_text, 'precision_digits': dps,
        'cutoff_sigma': cutoff, 'summation_limit': limit,
        'raw_moment_tail_upper_bounds': tails,
        'gaussian_entropy_bits': serialize(h),
        'gaussian_entropy_bytes': serialize(h / 8),
        'information_variance_bits_squared': serialize(v),
        'pnorm_lower_bound_evaluation': serialize(pnorm_lower),
        'c_minus_one': serialize(c_minus_one),
        'upper_correction_bytes': serialize(correction / 8),
        'kl_upper_bits': serialize(kl_upper),
        'conditional_P_entropy_bits': {
            'lower': mp.nstr(lower, 35), 'upper': mp.nstr(upper, 35)},
        'conditional_P_entropy_bytes': {
            'lower': mp.nstr(lower / 8, 35), 'upper': mp.nstr(upper / 8, 35)},
        'discrete_minus_continuous_entropy_bits': serialize(h - gaussian_approx),
        'entropy_bytes_plus_41_byte_storage': serialize(nominal_interval),
        'rounded_entropy_based_estimate_bytes': ceil_upper,
    }


def main():
    OUT.mkdir(exist_ok=True)
    available = {p.name: {'n': p.n, 'sigma_sig_decimal': str(p.sigma_sig_decimal)}
                 for p in (FALCONPP_I_1245, FALCONPP_1024_117)}
    manuscript = (ROOT / 'data/paper_tables.tex').read_text(encoding='utf-8')
    settings = [
        ('FALCON++-512', 'Falcon++-I-1245', '7.52e-20', '9.9275', 419),
        ('FALCON++-1024', 'falconpp-1024-1949-gamma117', '7.43e-20', '19.9940', 921),
    ]
    result = {
        'scope': 'Ideal sampling interface, conditional on Section 6 heuristic-profile bounds. Not empirical entropy or a certificate for actual generated keys.',
        'entropy_target': 'Norm-accepted s2 marginal, not joint entropy of salt and s2.',
        'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'source': str(SOURCE.relative_to(ROOT)), 'runs': [],
        'identity': 'H(P)=E_P[-log2 G]-KL_2(P||G)',
        'bound': 'H(G)-sqrt((c-1)V)-log2(c) <= H(P) <= H(G)+sqrt((c-1)V), c=r_inf_upper/p_norm_lower',
    }
    lines = ['# Signature entropy verification', '', result['scope'], '',
             'All infinite Gaussian sums have explicit geometric tail bounds. '
             'Interval arithmetic also encloses rounding error. '
             'Runs use 60/90 decimal digits and 12/16 sigma cutoffs.', '',
             '| Set | Gaussian entropy (B) | Conditional H(P) range (B) | Upward correction (B) | 41 + ceil(H(P)/8) |',
             '|---|---:|---:|---:|---:|']
    for display, name, epsilon, exponent, expected in settings:
        assert exponent in manuscript
        assert epsilon.split('e')[0] + r'\times10^{-20}' in manuscript
        param = available[name]
        runs = [calculate(param['n'], param['sigma_sig_decimal'], epsilon, exponent, dps, cutoff)
                for dps, cutoff in [(60, 12), (90, 16)]]
        for run in runs:
            run['set'] = display
            run['input_r_infinity_minus_one'] = epsilon
            run['input_norm_tail_exponent'] = exponent
            assert run['rounded_entropy_based_estimate_bytes'] == expected
        # Independent precision/cutoff runs must agree on the entropy range
        # to much better than any precision reported in the manuscript.
        for endpoint in ('lower', 'upper'):
            assert abs(mp.mpf(runs[0]['conditional_P_entropy_bytes'][endpoint])
                       - mp.mpf(runs[1]['conditional_P_entropy_bytes'][endpoint])) < mp.mpf('1e-20')
        result['runs'].extend(runs)
        r = runs[-1]
        h = float(r['gaussian_entropy_bytes']['lower'])
        lo = float(r['conditional_P_entropy_bytes']['lower'])
        hi = float(r['conditional_P_entropy_bytes']['upper'])
        inc = float(r['upper_correction_bytes']['upper'])
        lines.append(f'| {display} | {h:.9f} | {lo:.9f} to {hi:.9f} | {inc:.9f} | {expected} |')
        print(display, 'conditional entropy bytes:', mp.nstr(mp.mpf(str(lo)), 12),
              'to', mp.nstr(mp.mpf(str(hi)), 12), 'rounded estimate:', expected)
    lines += ['', 'The final column is an entropy-based accounting calculation, '
              'not a fixed-length rANS guarantee. The 41-byte salt is stored separately. '
              'Its accepted joint entropy is not evaluated.', '',
              'The input widths are read from the exact-decimal parameter registry. '
              'The divergence and norm-tail bounds are the displayed Section 6 values. '
              'The moment-model implementation explicitly labels them as heuristic-profile, '
              'not support-wide KeyGen certificates. No manuscript files were changed.', '',
              'Reproduce from the repository root:', '',
              '```powershell',
              '& python verify_entropy.py',
              '```']
    (OUT / 'entropy.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    (OUT / 'entropy.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
