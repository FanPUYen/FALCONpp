"""Portable entry point for the two parameter sets in the current manuscript.

Numerical models only; no lattice attack is executed. The chi-BDD diagnostic
reconstructs the recorded single-character/GSA calculation, not FALCONws Fig. 1.
"""
from pathlib import Path
import argparse
import json
import math
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'implement'))
import mpmath as mp
import numpy as np
from falconpp.parameters import FALCONPP_I_1245, FALCONPP_1024_117
from security.profiles import SecurityProfile, DETERMINANT_NORMALIZED
from security.moment_bounds import HeuristicMomentModel
from security.falconws import falconws_profile_trials
from security.attack_costs import (
    key_recovery_block_size, forgery_block_size, core_svp_cost,
)
from security.trial_caps import chernoff_lower_tail_bits


def generic_character_diagnostic(p):
    """Scan the recorded GSA model; minimize max(log2 reduction, log2 samples).

    delta_b = (b*(pi*b)**(1/b)/(2*pi*e))**(1/(2*(b-1)))
    ell = delta_b**(m-1)*q**(n/m)
    N_suff = ceil(8*ln(8)*exp(4*pi**2*sigma**2*ell**2/q**2)).
    The Gaussian leading-term bias is a model, not a hardness lower bound.
    """
    best = None
    for b in range(40, 2*p.n + 1):
        m = np.arange(max(p.n + 1, b), 2*p.n + 1, dtype=float)
        ld = (math.log(b) + math.log(math.pi*b)/b
              - math.log(2*math.pi*math.e)) / (2*(b-1))
        ell2 = np.exp(2*((m-1)*ld + p.n/m*math.log(p.q)))
        sample_log = (math.log2(8*math.log(8))
                      + 4*math.pi**2*float(p.sigma_sig_mpf)**2*ell2
                      / (p.q**2*math.log(2)))
        score = np.maximum(0.292*b, sample_log)
        i = int(np.argmin(score))
        point = (float(score[i]), b, int(m[i]))
        if best is None or point < best:
            best = point
    _, b, m = best
    # Independently evaluate the winning point at 100 decimal digits.
    bb = mp.mpf(b)
    delta = (bb*(mp.pi*bb)**(1/bb)/(2*mp.pi*mp.e))**(1/(2*(bb-1)))
    ell = delta**(m-1)*mp.mpf(p.q)**(mp.mpf(p.n)/m)
    sample_log = (mp.log(8*mp.log(8), 2)
                  + 4*mp.pi**2*p.sigma_sig_mpf**2*ell**2/(p.q**2*mp.log(2)))
    return {'b': b, 'm': m, 'predicted_vector_length': ell,
            'classical_reduction_exponent': mp.mpf('0.292')*b,
            'log2_sufficient_samples_leading_term': sample_log,
            'table_integer_samples': int(mp.floor(sample_log)),
            'scope': 'Generic single-character GSA diagnostic; no preimage oracle or planted NTRU relation. Sample exponent is NOT security bits.'}


def numerical():
    mp.mp.dps = 100
    rows = []
    for reg in (FALCONPP_I_1245, FALCONPP_1024_117):
        print('Evaluating', reg.name, flush=True)
        p = SecurityProfile(reg.name, reg.param_id, reg.n, reg.q,
                            str(reg.gamma_decimal), str(reg.sigma_sig_decimal),
                            reg.beta, reg.target_security)
        model = HeuristicMomentModel(p, normalization=DETERMINANT_NORMALIZED, dps=100)
        delta = mp.exp(model.log_delta_hat())
        rec = model.moment_record(reg.moment_order)
        k = rec.order
        gamma = mp.mpf(str(reg.widehat_gamma_decimal))
        tau = gamma*delta
        deficit = mp.exp((k-1)*mp.log(k-1)-k*mp.log(k)
                         +(k-1)*mp.log(tau)+rec.log_moment_upper)
        r = 1/(1-deficit)
        u = mp.mpf(p.beta**2+1)/(2*p.n*p.sigma_sig_mpf**2)
        exponent = p.n*(u-1-mp.log(u))-mp.log(r)
        pc, pn = tau*(1-deficit), 1-mp.exp(-exponent)
        kr = core_svp_cost(key_recovery_block_size(p.n, p.q, p.sigma_fg))
        fg = core_svp_cost(forgery_block_size(p.n, p.q, p.beta))
        dual = generic_character_diagnostic(p)
        row = {
            'name': reg.name, 'n': p.n, 'q': p.q, 'gamma': p.gamma,
            'sigma_sig': p.sigma_sig, 'beta': p.beta, 'moment_order': k,
            'gamma_hat': gamma, 'delta': delta, 'tau': tau,
            'r_infinity_minus_one_upper': deficit/(1-deficit),
            'p_corr_lower': pc, 'norm_tail_exponent': exponent,
            'p_norm_lower': pn, 'trials_upper': 1/(pc*pn),
            'falconws_same_point_trials': falconws_profile_trials(p).trials,
            'sampler_loss_bits': -p.candidate_cap*mp.log1p(-deficit)/mp.log(2),
            'candidate_cap_tail_bits': chernoff_lower_tail_bits(p.candidate_cap, pn, 1<<64),
            'raw_cap_tail_bits': chernoff_lower_tail_bits(1<<65, pc, p.candidate_cap),
            'public_key_payload_bytes': (p.n*(p.q-1).bit_length()+7)//8,
            'key_recovery': {'b': kr.block_size, 'classical': kr.classical_bits, 'quantum': kr.quantum_bits},
            'forgery': {'b': fg.block_size, 'classical': fg.classical_bits, 'quantum': fg.quantum_bits},
            'chi_bdd_generic_diagnostic': dual,
        }
        expected = ((468,136,124,581,169,153,1024,1401) if p.n==512
                    else (936,273,248,1277,372,338,2048,2813))
        actual = (kr.block_size,kr.classical_bits,kr.quantum_bits,
                  fg.block_size,fg.classical_bits,fg.quantum_bits,
                  dual['b'],dual['table_integer_samples'])
        assert actual == expected, (actual, expected)
        # Displayed lower bounds are truncated downward, not rounded to nearest.
        assert 0 <= pc-mp.mpf('0.526341' if p.n==512 else '0.575072') < mp.mpf('0.000001')
        assert 0 <= exponent-mp.mpf('9.9275' if p.n==512 else '19.9940') < mp.mpf('0.0001')
        print('  trials:', mp.nstr(row['trials_upper'],8), 'KR:', row['key_recovery'], flush=True)
        rows.append(row)
    out = ROOT/'results'; out.mkdir(exist_ok=True)
    document = {'scope': 'Heuristic-profile numerical evaluation at registered moment orders; no order search or KeyGen campaign in this command.', 'rows': rows}
    (out/'selected_parameters.json').write_text(
        json.dumps(document, indent=2, default=lambda x: mp.nstr(x,60))+'\n', encoding='utf-8')


def run_script(relative):
    subprocess.run([sys.executable, str(ROOT/relative)], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    numerical()
    run_script('verify_entropy.py')
    run_script('export_tables.py')
    run_script('check_paper_tables.py')
    print('Requested numerical checks completed.', flush=True)


if __name__ == '__main__':
    main()
