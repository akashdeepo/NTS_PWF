"""
Generate Figure 2(a) — Overview of NTS Parameter Effects (fig_2a.pdf)
======================================================================

Single-panel overview density plot showing the benchmark NTS alongside four
single-parameter perturbations and a standard-normal reference, matching the
manuscript caption:

    Benchmark NTS (blue solid):     mu=0, sigma=1.0, C=0.6, G=M=5,    Y=1.2
    High volatility (green):        sigma=1.4   (scale channel)
    Low volatility (red):           sigma=0.7   (scale channel)
    Right-skew (orange):            G=15, M=1   (skew channel)
    Thick tails (purple):           Y=0.9       (tail channel)
    Standard normal (black dashed): reference

The original generator for this figure could not be located in the repo
(possibly deleted); this script reconstructs it from the caption specification.

Output: outputs/figure_overview/Figure_2a_Overview_PDFs.{png,pdf}

Copy the PDF to PWF_for_NTS_Laws.../figures/fig_2a.pdf to use in the manuscript.

Author: rebuilt 2026-05 to address Lindquist's "reduce size, larger text" review.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from scipy.stats import norm as _scipy_norm

# Path setup: works from any cwd (script at NTS_PWF/src/)
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, 'lib', 'temStaPy_v0.5'))
from temStaPy.distNTS import dnts

# Publication style: larger fonts relative to plot, smaller physical figsize
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 15,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
})

OUTPUT_DIR = os.path.join(_ROOT, "outputs", "figure_overview")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def cgmy_to_nts(C, G, M, Y, mu=0.0, sigma=1.0):
    """Convert CGMY-parameterized NTS to the (alpha, theta, beta, gamma, mu)
    parameterization used by temStaPy.distNTS."""
    alpha = Y
    theta = C * (G ** Y + M ** Y) / Y
    beta_raw = (M ** Y - G ** Y) / (M ** Y + G ** Y)
    gamma = sigma
    beta = beta_raw * sigma
    return [alpha, theta, beta, gamma, mu]


# Benchmark + perturbations
mu, C = 0.0, 0.6
Y_bench = 1.2
G_bench, M_bench = 5.0, 5.0
sigma_bench = 1.0

specs = [
    # (label, nts-params, color, linestyle, linewidth)
    ("Benchmark ($\\sigma=1.0$, $G=M=5$, $Y=1.2$)",
     cgmy_to_nts(C, G_bench, M_bench, Y_bench, mu, sigma_bench),
     "tab:blue", "-", 2.2),
    ("High volatility ($\\sigma=1.4$)",
     cgmy_to_nts(C, G_bench, M_bench, Y_bench, mu, 1.4),
     "tab:green", "-", 1.8),
    ("Low volatility ($\\sigma=0.7$)",
     cgmy_to_nts(C, G_bench, M_bench, Y_bench, mu, 0.7),
     "tab:red", "-", 1.8),
    ("Right-skew ($G=15$, $M=1$)",
     cgmy_to_nts(C, 15.0, 1.0, Y_bench, mu, sigma_bench),
     "tab:orange", "-", 1.8),
    ("Thick tails ($Y=0.9$)",
     cgmy_to_nts(C, G_bench, M_bench, 0.9, mu, sigma_bench),
     "tab:purple", "-", 1.8),
]

# Grid: wide enough to show tails for sigma=1.4 and Y=0.9
x_grid = np.linspace(-6.0, 6.0, 1201)

print("=" * 70)
print("Generating Figure 2(a): NTS overview densities")
print("=" * 70)

fig, ax = plt.subplots(figsize=(7, 4.8))
for label, params, color, ls, lw in specs:
    print(f"  Computing density for: {label}")
    pdf_vals = dnts(x_grid, params)
    ax.plot(x_grid, pdf_vals, color=color, linestyle=ls, linewidth=lw,
            label=label, alpha=0.9)

# Standard-normal reference (dashed) — drawn on top with heavier dashed line
ax.plot(x_grid, _scipy_norm.pdf(x_grid),
        color="black", linestyle=(0, (6, 4)), linewidth=2.0,
        label="Standard normal (reference)", zorder=10)

ax.set_xlabel("x")
ax.set_ylabel(r"$f_X(x)$")
ax.set_title("Densities under single-channel NTS perturbations")
ax.set_xlim(-6, 6)
ax.set_ylim(0, None)
ax.grid(True, alpha=0.3)
ax.legend(loc="upper right", framealpha=0.95)
plt.tight_layout()

save_path = os.path.join(OUTPUT_DIR, "Figure_2a_Overview_PDFs")
plt.savefig(save_path + ".png", dpi=300, bbox_inches="tight")
plt.savefig(save_path + ".pdf", bbox_inches="tight")
plt.close()
print(f"\nSaved: {save_path}.png and {save_path}.pdf")
print("\nTo use in the manuscript, copy the PDF over the existing overview:")
print("  copy outputs\\figure_overview\\Figure_2a_Overview_PDFs.pdf "
      "..\\..\\PWFs for NTS\\PWF_for_NTS_Laws__Channels__Indices__"
      "and_Behavioral_Rational_Integration\\figures\\fig_2a.pdf")
