"""Shared matplotlib style, colours and output helpers for the paper's figures.

Every figure is printed at 0.7\\textwidth in the manuscript, so fonts are set
large relative to the figure size. Fearful specifications are red dashed,
greedy ones blue dash-dotted, and the benchmark is solid black.
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(_ROOT, 'outputs', 'figures')
TAB_DIR = os.path.join(_ROOT, 'outputs', 'tables')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 13,
    'axes.titlesize': 13,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'lines.linewidth': 2.0,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

COLOR = {'benchmark': 'black', 'fearful': 'tab:red', 'greedy': 'tab:blue'}
STYLE = {'benchmark': '-', 'fearful': '--', 'greedy': '-.'}
LABEL = {'benchmark': 'Benchmark', 'fearful': 'Fearful', 'greedy': 'Greedy'}


def line_kw(scenario, **extra):
    kw = dict(color=COLOR[scenario], linestyle=STYLE[scenario], label=LABEL[scenario])
    kw.update(extra)
    return kw


def legend_below(fig, handles, labels, ncol, y=-0.02):
    fig.legend(handles, labels, loc='lower center', ncol=ncol,
               bbox_to_anchor=(0.5, y), frameon=True)


def interior_ylim(ax, p, curves, lo=0.02, hi=0.98, pad=1.1):
    """Symmetric y-limits from the curves' range on p in [lo, hi]."""
    m = (p >= lo) & (p <= hi)
    top = max(np.max(np.abs(c[m])) for c in curves) * pad
    ax.set_ylim(-top, top)


def save(fig, name):
    """Write outputs/figures/<name>.pdf and .png and close the figure."""
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIG_DIR, f'{name}.{ext}'))
    plt.close(fig)
    print(f'  wrote {name}.pdf')
