"""
Animated channel signatures for the README (docs/channel_signatures.gif).

Each panel sweeps one channel from its greedy extreme to its fearful extreme
and back, around the stylized benchmark of Section 4, and plots the PWF
deviation 100 [w(p) - p]. The number of sign changes (one, two, three) is
the paper's identifying signature for the scale, skew and tail channels.

    python src/make_animation.py        # ~1 minute

Not part of run_all.py; the GIF is committed under docs/ so the README can
show it without running anything.
"""

import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

from nts_utils import (STYLIZED_BENCHMARK, beta_for_skewness, gamma_for_variance,  # noqa: E402
                       nts_moments, probability_grid, compute_pwf)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'docs', 'channel_signatures.gif')
os.makedirs(os.path.dirname(OUT), exist_ok=True)

alpha0, theta0, beta0, gamma0, mu0 = STYLIZED_BENCHMARK
var0 = nts_moments(STYLIZED_BENCHMARK)['variance']
p = probability_grid(600)

N_HALF = 28                                   # frames from one extreme to the other
s_path = np.concatenate([np.linspace(-1, 1, N_HALF), np.linspace(1, -1, N_HALF)[1:-1]])


def spec(channel, s):
    """Posterior for channel at strength s in [-1, 1]: -1 greedy, +1 fearful."""
    if channel == 'scale':
        g = 1.4 ** s if s >= 0 else 0.7 ** (-s)
        return [alpha0, theta0, beta0, gamma0 * g, mu0], f'$\\gamma = {gamma0 * g:.2f}$'
    if channel == 'skew':
        sk = -0.5 * s
        if abs(sk) < 1e-9:
            return list(STYLIZED_BENCHMARK), 'skewness $0.00$'
        return beta_for_skewness(alpha0, theta0, sk, var0, mu0), f'skewness ${sk:+.2f}$'
    a = alpha0 - 0.3 * s
    return [a, theta0, beta0, gamma_for_variance(a, theta0, beta0, var0), mu0], f'$\\alpha = {a:.2f}$'


print('precomputing frames ...')
channels = ['scale', 'skew', 'tail']
frames = []
for s in s_path:
    row = []
    for ch in channels:
        params, label = spec(ch, s)
        row.append((100 * (compute_pwf(p, STYLIZED_BENCHMARK, params) - p), label))
    frames.append(row)
print(f'  {len(frames)} frames')

plt.rcParams.update({'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11})
fig, axes = plt.subplots(1, 3, figsize=(10, 3.9), dpi=90)
titles = {'scale': 'Scale channel: one sign change',
          'skew': 'Skew channel: two sign changes',
          'tail': 'Tail channel: three sign changes'}
title_artists = []
ylims = {'scale': 8.5, 'skew': 2.6, 'tail': 2.3}
lines, labels = [], []
for ax, ch in zip(axes, channels):
    ax.axhline(0, color='gray', linewidth=1)
    (ln,) = ax.plot([], [], linewidth=2.4)
    lines.append(ln)
    ax.set_xlim(0, 1)
    ax.set_ylim(-ylims[ch], ylims[ch])
    title_artists.append(ax.set_title(titles[ch] + '\n'))
    ax.set_xlabel('$p$')
    ax.grid(True, alpha=0.3)
axes[0].set_ylabel(r'$100\,[w(p) - p]$')
fig.text(0.5, 0.005, 'Posterior swept from greedy (blue) to fearful (red) around a unit-variance NTS benchmark, '
         'variance held fixed', ha='center', fontsize=9.5, color='dimgray')
fig.tight_layout(rect=[0, 0.05, 1, 1])

cmap = plt.get_cmap('coolwarm')


def draw(i):
    s = s_path[i]
    colour = cmap(0.5 + 0.5 * s)
    for ln, ta, ch, (dev, text) in zip(lines, title_artists, channels, frames[i]):
        ln.set_data(p, dev)
        ln.set_color(colour)
        ta.set_text(titles[ch] + '\n' + text)
    return lines + title_artists


anim = FuncAnimation(fig, draw, frames=len(frames), blit=False)
anim.save(OUT, writer=PillowWriter(fps=12))
print('wrote', OUT, f'({os.path.getsize(OUT) / 1e6:.2f} MB)')

# a still of the fearful extreme for quick inspection
draw(N_HALF - 1)
fig.savefig(OUT.replace('.gif', '_still.png'))
