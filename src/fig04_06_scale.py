"""
Figures 4-6: the scale (volatility) channel around the stylized benchmark.

    fearful  gamma = 1.4  (variance x1.96)
    greedy   gamma = 0.7  (variance x0.49)

Fig 4  PWFs w(p)
Fig 5  classical index gamma(p) = (w(p) - p) / (p (1 - p)) on [0.01, 0.99]
Fig 6  Fisher-normalized logit shift G_FI(p)

Output: outputs/figures/fig04_scale_pwf, fig05_scale_gamma, fig06_scale_gfi
"""

import numpy as np

from nts_utils import STYLIZED_BENCHMARK, channel_specs, probability_grid, all_indices, aggregate_indices
import paper_style as ps
import matplotlib.pyplot as plt

specs = channel_specs(STYLIZED_BENCHMARK)['scale']
p = probability_grid(2000)
res = {s: all_indices(p, STYLIZED_BENCHMARK, specs[s]) for s in ('fearful', 'greedy')}

print('Figures 4-6: scale channel')
for s, r in res.items():
    d = r['w'] - p
    i = np.argmax(np.abs(d))
    agg = aggregate_indices(p, r)
    print(f'  {s:8s} max|w-p| = {100*abs(d[i]):.2f}% at p={p[i]:.3f}; '
          f'G_FI(0.01)={r["G_FI"][np.argmin(abs(p-0.01))]:+.1f}; '
          f'int|G_FI|={agg["int_abs_GFI"]:.3f} int|S_JS|={agg["int_abs_SJS"]:.4f} int|E|={agg["int_abs_E"]:.3f}')

# Fig 4: PWFs
fig, ax = plt.subplots(figsize=(5.2, 5.2))
ax.plot([0, 1], [0, 1], color='gray', linestyle=':', linewidth=1.5, label='Benchmark ($w(p)=p$)')
for s in ('fearful', 'greedy'):
    ax.plot(p, res[s]['w'], **ps.line_kw(s, label=f'{ps.LABEL[s]} ($\\gamma={specs[s][3]:.1f}$)'))
ax.set_xlabel('$p$')
ax.set_ylabel('$w(p)$')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')
ax.legend(loc='upper left')
fig.tight_layout()
ps.save(fig, 'fig04_scale_pwf')

# Fig 5: classical index on [0.01, 0.99]
m = (p >= 0.01) & (p <= 0.99)
fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.axhline(0, color='gray', linewidth=1)
for s in ('fearful', 'greedy'):
    ax.plot(p[m], res[s]['gamma'][m], **ps.line_kw(s, label=f'{ps.LABEL[s]} ($\\gamma={specs[s][3]:.1f}$)'))
ax.set_xlabel('$p$')
ax.set_ylabel(r'$g(p) = (w(p)-p)/[p(1-p)]$')
ax.set_xlim(0, 1)
ax.legend(loc='upper right')
fig.tight_layout()
ps.save(fig, 'fig05_scale_gamma')

# Fig 6: Fisher-normalized logit shift (unbounded at the endpoints; axis cut at +-40)
fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.axhline(0, color='gray', linewidth=1)
for s in ('fearful', 'greedy'):
    ax.plot(p, res[s]['G_FI'], **ps.line_kw(s, label=f'{ps.LABEL[s]} ($\\gamma={specs[s][3]:.1f}$)'))
ax.set_xlabel('$p$')
ax.set_ylabel(r'$G_{\mathrm{FI}}(p)$')
ax.set_xlim(0, 1)
ax.set_ylim(-40, 40)
ax.legend(loc='upper right')
fig.tight_layout()
ps.save(fig, 'fig06_scale_gfi')
