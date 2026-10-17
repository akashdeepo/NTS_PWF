"""
Every number quoted in the manuscript, recomputed from the model code.

Writes outputs/summary_numbers.json and prints a readable version. Use it to
check the text after any change to the parameters or the library.
"""

import json
import os

import numpy as np
import pandas as pd
from scipy import stats

from nts_utils import (STYLIZED_BENCHMARK, channel_specs, nts_moments, probability_grid,
                       all_indices, aggregate_indices)
from temStaPy.distNTS import pnts, fitnts

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'outputs', 'summary_numbers.json')


def describe(p, prior, post):
    r = all_indices(p, prior, post)
    d = r['w'] - p
    i = int(np.argmax(np.abs(d)))
    at = lambda q, key: float(r[key][np.argmin(np.abs(p - q))])
    crossings = int(np.sum(np.diff(np.sign(d[(p > 0.002) & (p < 0.998)])) != 0))
    m = nts_moments(post)
    return {
        'params': [round(float(v), 6) for v in post],
        'sd': round(m['variance'] ** 0.5, 4), 'skewness': round(m['skewness'], 3),
        'excess_kurtosis': round(m['excess kurtosis'], 3),
        'max_abs_dev_pct': round(100 * abs(d[i]), 2), 'at_p': round(float(p[i]), 3),
        'sign_changes_of_dev': crossings,
        'gamma_at_0.05': round(at(0.05, 'gamma'), 2), 'gamma_at_0.95': round(at(0.95, 'gamma'), 2),
        'GFI_at_0.01': round(at(0.01, 'G_FI'), 1), 'GFI_at_0.99': round(at(0.99, 'G_FI'), 1),
        'SJS_max_abs': round(float(np.max(np.abs(r['S_JS']))), 3),
        'E_max_abs_on_[0.05,0.95]': round(float(np.max(np.abs(r['E'][(p >= 0.05) & (p <= 0.95)]))), 2),
        **{k: round(v, 4) for k, v in aggregate_indices(p, r).items()},
    }


out = {'stylized': {}, 'empirical': {}}
p = probability_grid(2000)

b = STYLIZED_BENCHMARK
out['stylized']['benchmark'] = {'params': b, **{k: round(v, 4) for k, v in nts_moments(b).items()}}
for ch, sc in channel_specs(b).items():
    for s in ('fearful', 'greedy'):
        out['stylized'][f'{ch}_{s}'] = describe(p, b, sc[s])

df = pd.read_csv(os.path.join(ROOT, 'data', 'spy_prices.csv'), parse_dates=['Date']).sort_values('Date')
r = df['log_ret'].dropna().values
fit = [float(x) for x in fitnts(r, maxeval=500)]
out['empirical']['sample'] = {
    'N': int(len(r)), 'first_date': str(df['Date'].min().date()), 'last_date': str(df['Date'].max().date()),
    'mean': float(r.mean()), 'sd': float(r.std()), 'annualized_vol': float(r.std() * np.sqrt(252)),
    'skewness': float(stats.skew(r)), 'excess_kurtosis': float(stats.kurtosis(r)),
    'min': float(r.min()), 'max': float(r.max()),
}
out['empirical']['fit'] = {'params': fit, **{k: round(v, 6) for k, v in nts_moments(fit).items()}}
cdf = lambda x: pnts(np.atleast_1d(x), fit)
ks, cvm = stats.kstest(r, cdf), stats.cramervonmises(r, cdf)
cdf_norm = stats.norm(r.mean(), r.std()).cdf
ksn = stats.kstest(r, cdf_norm)
cvmn = stats.cramervonmises(r, cdf_norm)
out['empirical']['gof'] = {
    'NTS': {'KS': float(ks.statistic), 'KS_p': float(ks.pvalue), 'CvM': float(cvm.statistic), 'CvM_p': float(cvm.pvalue)},
    'Gaussian': {'KS': float(ksn.statistic), 'KS_p': float(ksn.pvalue), 'CvM': float(cvmn.statistic), 'CvM_p': float(cvmn.pvalue)},
}
for ch, sc in channel_specs(fit).items():
    for s in ('fearful', 'greedy'):
        out['empirical'][f'{ch}_{s}'] = describe(p, fit, sc[s])

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(out, open(OUT, 'w'), indent=1)
print(json.dumps(out, indent=1))
print(f'\nwrote {OUT}')
