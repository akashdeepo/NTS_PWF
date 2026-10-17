"""
Figures 10-14 and Tables 1-3: NTS calibrated to SPY daily log-returns.

1. Fit NTS(alpha, theta, beta, gamma, mu) to SPY log-returns 2015-2025 with
   temStaPy.fitnts (location and scale from the sample mean and standard
   deviation; shape parameters by least squares between the NTS CDF and a
   kernel-smoothed empirical CDF).
2. Goodness of fit: Kolmogorov-Smirnov and Cramer-von Mises statistics for the
   fitted NTS and for a Gaussian with the same mean and variance.
3. Apply the three channels of Section 4 around the fitted benchmark
   (scale x1.4 / x0.7, skewness -/+0.5 at constant variance, alpha -/+0.3).
4. PWFs, deviations and indices for each channel.

Output: outputs/figures/fig10_spy_fit ... fig14_spy_indices
        outputs/tables/table_sample_statistics.csv, table_fitted_parameters.csv,
        table_goodness_of_fit.csv, table_channel_specs.csv, table_aggregate_indices.{csv,tex}
"""

import os

import numpy as np
import pandas as pd
from scipy import stats

from nts_utils import channel_specs, probability_grid, all_indices, aggregate_indices, nts_moments
from temStaPy.distNTS import dnts, pnts, fitnts
import paper_style as ps
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data', 'spy_prices.csv')

# ---------------------------------------------------------------- 1. data and fit
df = pd.read_csv(DATA, parse_dates=['Date']).sort_values('Date')
r = df['log_ret'].dropna().values
print(f'SPY: {len(r)} daily log-returns, {df["Date"].min().date()} to {df["Date"].max().date()}')

fit = [float(x) for x in fitnts(r, maxeval=500)]
alpha, theta, beta, gamma, mu = fit
mo = nts_moments(fit)
print(f'fit: alpha={alpha:.4f} theta={theta:.4f} beta={beta:.6f} gamma={gamma:.6f} mu={mu:.6f}')
print(f'     implied skewness {mo["skewness"]:+.3f}, excess kurtosis {mo["excess kurtosis"]:.2f}')

sample = {
    'Observations': len(r), 'Mean': r.mean(), 'Standard deviation': r.std(),
    'Annualized volatility': r.std() * np.sqrt(252),
    'Skewness': stats.skew(r), 'Excess kurtosis': stats.kurtosis(r),
    'Minimum': r.min(), 'Maximum': r.max(),
}
pd.DataFrame(sample.items(), columns=['Statistic', 'Value']).to_csv(
    os.path.join(ps.TAB_DIR, 'table_sample_statistics.csv'), index=False)
pd.DataFrame({'Parameter': ['alpha', 'theta', 'beta', 'gamma', 'mu'], 'Estimate': fit,
              }).to_csv(os.path.join(ps.TAB_DIR, 'table_fitted_parameters.csv'), index=False)

# ---------------------------------------------------------------- 2. goodness of fit
cdf_nts = lambda x: pnts(np.atleast_1d(x), fit)
ks = stats.kstest(r, cdf_nts)
cvm = stats.cramervonmises(r, cdf_nts)
cdf_norm = stats.norm(r.mean(), r.std()).cdf
ks_n = stats.kstest(r, cdf_norm)
cvm_n = stats.cramervonmises(r, cdf_norm)
gof = pd.DataFrame([
    ['NTS', ks.statistic, ks.pvalue, cvm.statistic, cvm.pvalue],
    ['Gaussian', ks_n.statistic, ks_n.pvalue, cvm_n.statistic, cvm_n.pvalue],
], columns=['Model', 'KS statistic', 'KS p-value', 'CvM statistic', 'CvM p-value'])
gof.to_csv(os.path.join(ps.TAB_DIR, 'table_goodness_of_fit.csv'), index=False)
print(gof.to_string(index=False))

# ---------------------------------------------------------------- 3. channels
specs = channel_specs(fit)
rows = []
for ch, sc in specs.items():
    for s in ('fearful', 'benchmark', 'greedy'):
        m = nts_moments(sc[s])
        rows.append([ch, s] + [round(v, 6) for v in sc[s]] +
                    [m['variance'] ** 0.5, m['skewness'], m['excess kurtosis']])
pd.DataFrame(rows, columns=['channel', 'scenario', 'alpha', 'theta', 'beta', 'gamma', 'mu',
                            'sd', 'skewness', 'excess_kurtosis']).to_csv(
    os.path.join(ps.TAB_DIR, 'table_channel_specs.csv'), index=False)

p = probability_grid(2000)
res = {ch: {s: all_indices(p, fit, sc[s]) for s in ('fearful', 'greedy')} for ch, sc in specs.items()}

agg_rows = []
for ch in specs:
    for s in ('fearful', 'greedy'):
        d = res[ch][s]['w'] - p
        i = np.argmax(np.abs(d))
        a = aggregate_indices(p, res[ch][s])
        agg_rows.append([ch, s, 100 * abs(d[i]), p[i], a['int_GFI'], a['int_abs_GFI'],
                         a['int_SJS'], a['int_abs_SJS'], a['int_abs_E']])
        print(f'  {ch:5s} {s:8s} max|w-p|={100*abs(d[i]):.2f}% at p={p[i]:.3f}  '
              f'int G_FI={a["int_GFI"]:+.3f} int|G_FI|={a["int_abs_GFI"]:.3f} '
              f'int S_JS={a["int_SJS"]:+.4f} int|S_JS|={a["int_abs_SJS"]:.4f} int|E|={a["int_abs_E"]:.3f}')
agg = pd.DataFrame(agg_rows, columns=['channel', 'scenario', 'max_dev_pct', 'at_p', 'int_GFI',
                                      'int_abs_GFI', 'int_SJS', 'int_abs_SJS', 'int_abs_E'])
agg.to_csv(os.path.join(ps.TAB_DIR, 'table_aggregate_indices.csv'), index=False)
with open(os.path.join(ps.TAB_DIR, 'table_aggregate_indices.tex'), 'w') as f:
    f.write('% channel & scenario & max |w-p| (pp) & int G_FI & int |G_FI| & int S_JS & int |S_JS| & int |E|\n')
    for _, x in agg.iterrows():
        f.write(f'{x.channel.capitalize()} & {x.scenario.capitalize()} & {x.max_dev_pct:.1f} & '
                f'${x.int_GFI:+.2f}$ & {x.int_abs_GFI:.2f} & ${x.int_SJS:+.3f}$ & {x.int_abs_SJS:.3f} & {x.int_abs_E:.2f} \\\\\n')

# ---------------------------------------------------------------- 4. figures
CH_TITLE = {'scale': 'Scale channel', 'skew': 'Skew channel', 'tail': 'Tail channel'}

# Fig 10: histogram vs fitted density
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
xr = np.linspace(r.min(), r.max(), 600)
pdf = dnts(xr, fit)
for ax, log in zip(axes, (False, True)):
    ax.hist(r, bins=120, density=True, alpha=0.6, color='steelblue', edgecolor='white', label='SPY returns')
    ax.plot(xr, pdf, color='tab:red', linewidth=2.2, label='Fitted NTS')
    ax.set_xlabel('Daily log-return')
    if log:
        ax.set_yscale('log')
        ax.set_ylim(1e-2, 1e2)
        ax.set_ylabel('Density (log scale)')
        ax.set_title('Tails')
    else:
        ax.set_xlim(-0.06, 0.06)
        ax.set_ylabel('Density')
        ax.set_title('Centre')
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
h, l = axes[0].get_legend_handles_labels()
ps.legend_below(fig, h, l, ncol=2)
fig.tight_layout(rect=[0, 0.1, 1, 1])
ps.save(fig, 'fig10_spy_fit')


def three_panel(key, ylabel, name, transform=lambda y, p: y, square=False, clip=None):
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.4))
    for ax, ch in zip(axes, specs):
        if square:
            ax.plot([0, 1], [0, 1], color='gray', linestyle=':', linewidth=1.5, label='Benchmark')
        else:
            ax.axhline(0, color='gray', linewidth=1, label='Benchmark')
        for s in ('fearful', 'greedy'):
            ax.plot(p, transform(res[ch][s][key], p), **ps.line_kw(s))
        ax.set_title(CH_TITLE[ch])
        ax.set_xlabel('$p$')
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 1)
        if square:
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')
        elif clip:
            ps.interior_ylim(ax, p, [transform(res[ch][s][key], p) for s in res[ch]], lo=clip, hi=1 - clip)
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    h, l = axes[0].get_legend_handles_labels()
    ps.legend_below(fig, h, l, ncol=3)
    fig.tight_layout(rect=[0, 0.1, 1, 1])
    ps.save(fig, name)


three_panel('w', '$w(p)$', 'fig11_spy_pwf', square=True)
three_panel('w', r'$100\,[w(p)-p]$ (pp)', 'fig12_spy_deviation', transform=lambda w, p: 100 * (w - p))
three_panel('G_FI', r'$G_{\mathrm{FI}}(p)$', 'fig13_spy_gfi', clip=0.01)

# Fig 14: the three indices, scale channel
fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.4))
for ax, (key, ylabel) in zip(axes, [('G_FI', r'$G_{\mathrm{FI}}(p)$'), ('S_JS', r'$S_{\mathrm{JS}}(p)$'), ('E', r'$E(p)$')]):
    ax.axhline(0, color='gray', linewidth=1, label='Benchmark')
    for s in ('fearful', 'greedy'):
        ax.plot(p, res['scale'][s][key], **ps.line_kw(s))
    ax.set_xlabel('$p$')
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    if key != 'S_JS':
        ps.interior_ylim(ax, p, [res['scale'][s][key] for s in res['scale']], lo=0.01, hi=0.99)
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
h, l = axes[0].get_legend_handles_labels()
ps.legend_below(fig, h, l, ncol=3)
fig.tight_layout(rect=[0, 0.1, 1, 1])
ps.save(fig, 'fig14_spy_indices')
