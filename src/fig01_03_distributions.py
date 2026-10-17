"""
Figures 1-3: densities, CDFs and quantile functions of the stylized benchmark
and one perturbation per channel.

Benchmark NTS(alpha=1.2, theta=0.4, beta=0, gamma=1, mu=0) with
    high volatility   gamma = 1.4
    low volatility    gamma = 0.7
    right skew        skewness +0.5 at unit variance
    thick tails       alpha = 0.9
and the standard normal as reference.

Output: outputs/figures/fig01_densities, fig02_cdfs, fig03_quantiles
"""

import numpy as np
from scipy.stats import norm

from nts_utils import STYLIZED_BENCHMARK, channel_specs, nts_moments
from temStaPy.distNTS import dnts, pnts, qnts
import paper_style as ps
import matplotlib.pyplot as plt

specs = channel_specs(STYLIZED_BENCHMARK)
curves = [
    ('Benchmark', STYLIZED_BENCHMARK, dict(color='black', linestyle='-', linewidth=2.4)),
    (r'High volatility ($\gamma=1.4$)', specs['scale']['fearful'], dict(color='tab:red', linestyle='--')),
    (r'Low volatility ($\gamma=0.7$)', specs['scale']['greedy'], dict(color='tab:blue', linestyle='-.')),
    (r'Right skew (skewness $+0.5$)', specs['skew']['greedy'], dict(color='tab:orange', linestyle='-')),
    (r'Thick tails ($\alpha=0.9$)', specs['tail']['fearful'], dict(color='tab:purple', linestyle=':', linewidth=2.6)),
]
normal_kw = dict(color='gray', linestyle=(0, (5, 3)), linewidth=2.0, label='Standard normal')

x = np.linspace(-5, 5, 1001)
p = np.linspace(0.001, 0.999, 999)

print('Figures 1-3: distributions')
for label, params, _ in curves:
    m = nts_moments(params)
    print(f'  {label:32s} params={np.round(params, 4).tolist()} '
          f'var={m["variance"]:.3f} skew={m["skewness"]:+.3f} exkurt={m["excess kurtosis"]:.3f}')


def panel(fn, ref, ylabel, name, ylim=None, xvals=x):
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for label, params, kw in curves:
        ax.plot(xvals, fn(xvals, params), label=label, **kw)
    ax.plot(xvals, ref(xvals), **normal_kw)
    ax.set_xlabel('$x$' if xvals is x else '$p$')
    ax.set_ylabel(ylabel)
    ax.set_xlim(xvals[0], xvals[-1])
    if ylim:
        ax.set_ylim(*ylim)
    h, l = ax.get_legend_handles_labels()
    ps.legend_below(fig, h, l, ncol=2, y=-0.04)
    fig.tight_layout(rect=[0, 0.1, 1, 1])
    ps.save(fig, name)


panel(dnts, norm.pdf, '$f_X(x)$', 'fig01_densities', ylim=(0, None))
panel(pnts, norm.cdf, '$F_X(x)$', 'fig02_cdfs', ylim=(0, 1))
panel(qnts, norm.ppf, '$Q_X(p)$', 'fig03_quantiles', ylim=(-5, 5), xvals=p)
