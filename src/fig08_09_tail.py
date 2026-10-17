"""
Figures 8-9: the tail-thickness channel around the stylized benchmark.

    fearful  alpha = 0.9  (heavier tails, excess kurtosis 4.1)
    greedy   alpha = 1.5  (thinner tails, excess kurtosis 1.9)
Variance is unchanged because beta = 0.

Fig 8  PWFs w(p)
Fig 9  the three indices G_FI, S_JS and E

Output: outputs/figures/fig08_tail_pwf, fig09_tail_indices
"""

import numpy as np

from nts_utils import STYLIZED_BENCHMARK, channel_specs, probability_grid, all_indices, aggregate_indices, nts_moments
import paper_style as ps
import matplotlib.pyplot as plt

specs = channel_specs(STYLIZED_BENCHMARK)['tail']
p = probability_grid(2000)
res = {s: all_indices(p, STYLIZED_BENCHMARK, specs[s]) for s in ('fearful', 'greedy')}

print('Figures 8-9: tail channel')
for s, r in res.items():
    d = r['w'] - p
    i = np.argmax(np.abs(d))
    mo = nts_moments(specs[s])
    agg = aggregate_indices(p, r)
    print(f'  {s:8s} alpha={specs[s][0]:.1f} exkurt={mo["excess kurtosis"]:.3f} var={mo["variance"]:.3f}; '
          f'max|w-p| = {100*abs(d[i]):.2f}% at p={p[i]:.3f}; int|G_FI|={agg["int_abs_GFI"]:.3f} '
          f'int|S_JS|={agg["int_abs_SJS"]:.4f} int|E|={agg["int_abs_E"]:.3f}')

# Fig 8: PWFs
fig, ax = plt.subplots(figsize=(5.2, 5.2))
ax.plot([0, 1], [0, 1], color='gray', linestyle=':', linewidth=1.5, label='Benchmark ($w(p)=p$)')
for s in ('fearful', 'greedy'):
    ax.plot(p, res[s]['w'], **ps.line_kw(s, label=f'{ps.LABEL[s]} ($\\alpha={specs[s][0]:.1f}$)'))
ax.set_xlabel('$p$')
ax.set_ylabel('$w(p)$')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')
ax.legend(loc='upper left')
fig.tight_layout()
ps.save(fig, 'fig08_tail_pwf')

# Fig 9: three indices
fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.4))
panels = [('G_FI', r'$G_{\mathrm{FI}}(p)$'), ('S_JS', r'$S_{\mathrm{JS}}(p)$'), ('E', r'$E(p)$')]
for ax, (key, ylabel) in zip(axes, panels):
    ax.axhline(0, color='gray', linewidth=1)
    for s in ('fearful', 'greedy'):
        ax.plot(p, res[s][key], **ps.line_kw(s, label=f'{ps.LABEL[s]} ($\\alpha={specs[s][0]:.1f}$)'))
    ax.set_xlabel('$p$')
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    if key != 'S_JS':
        ps.interior_ylim(ax, p, [res[s][key] for s in res], lo=0.01, hi=0.99)
h, l = axes[0].get_legend_handles_labels()
ps.legend_below(fig, h, l, ncol=2)
fig.tight_layout(rect=[0, 0.1, 1, 1])
ps.save(fig, 'fig09_tail_indices')
