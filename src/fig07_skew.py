"""
Figure 7: PWF deviation delta(p) = w(p) - p for the skew channel.

Around the stylized benchmark (skewness 0, unit variance), beta is chosen so
that the posterior has skewness -0.5 (fearful, left skew) or +0.5 (greedy,
right skew), and gamma is adjusted so the variance stays at 1.

Output: outputs/figures/fig07_skew_deviation
"""

import numpy as np

from nts_utils import STYLIZED_BENCHMARK, channel_specs, probability_grid, compute_pwf, nts_moments
import paper_style as ps
import matplotlib.pyplot as plt

specs = channel_specs(STYLIZED_BENCHMARK)['skew']
p = probability_grid(2000)

print('Figure 7: skew channel')
fig, ax = plt.subplots(figsize=(6.6, 4.2))
ax.axhline(0, color='gray', linewidth=1)
for s in ('fearful', 'greedy'):
    params = specs[s]
    d = 100 * (compute_pwf(p, STYLIZED_BENCHMARK, params) - p)
    i = np.argmax(np.abs(d))
    mo = nts_moments(params)
    print(f'  {s:8s} beta={params[2]:+.4f} gamma={params[3]:.4f} skew={mo["skewness"]:+.3f} '
          f'var={mo["variance"]:.3f}; max|w-p| = {abs(d[i]):.2f}% at p={p[i]:.3f}')
    ax.plot(p, d, **ps.line_kw(s, label=f'{ps.LABEL[s]} (skewness ${mo["skewness"]:+.1f}$)'))
    ax.plot(p[i], d[i], 'o', color=ps.COLOR[s], markersize=8, markeredgecolor='black', zorder=5)
ax.set_xlabel('$p$')
ax.set_ylabel(r'$100\,[w(p)-p]$ (percentage points)')
ax.set_xlim(0, 1)
ax.legend(loc='upper left')
fig.tight_layout()
ps.save(fig, 'fig07_skew_deviation')
