"""
Generate Figures per Dr. Rachev's Manuscript (October 11, 2025)
Clean implementation with exact specifications
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Path setup: works from any cwd (script at NTS_PWF/src/)
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, 'lib', 'temStaPy_v0.5'))
from temStaPy.distNTS import qnts, pnts, dnts

# Publication style: larger fonts relative to plot, smaller physical figsize
# (per Lindquist review: "reduce the size and make the text on it larger")
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 15,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
})
from scipy.stats import norm as _scipy_norm  # for the standard-normal reference curve

# Create output directory
OUTPUT_DIR = os.path.join(_ROOT, "outputs", "case1_overview")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*80)
print("GENERATING FIGURES PER DR. RACHEV'S MANUSCRIPT SPECIFICATIONS")
print("="*80)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def cgmy_to_nts(C, G, M, Y, mu=0, sigma=1):
    """Convert CGMY parameters to NTS parameters"""
    alpha = Y
    theta = C * (G**Y + M**Y) / Y
    beta_raw = (M**Y - G**Y) / (M**Y + G**Y)
    gamma = sigma
    beta = beta_raw * sigma
    return [alpha, theta, beta, gamma, mu]

def compute_pwf(u_grid, prior_params, post_params):
    """Compute Probability Weighting Function: w(u) = F_post(Q_prior(u))"""
    q_prior = qnts(u_grid, prior_params)
    w = pnts(q_prior, post_params)
    return w

# ============================================================================
# COMMON PARAMETERS (Fixed across all cases per manuscript)
# ============================================================================
mu = 0.0
C = 0.6
Y = 1.2

# ============================================================================
# FIGURES 2(a), 2(b), 2(c): NTS FUNDAMENTALS
# ============================================================================

print("\n" + "-"*80)
print("Generating NTS Fundamental Figures (2a, 2b, 2c)")
print("-"*80)

# Parameters for fundamental figures (showing NTS properties)
G_base = 5.0
M_base = 5.0
sigma_base = 1.0

# Different specifications to show variety
nts_base = cgmy_to_nts(C, G_base, M_base, Y, mu, sigma_base)
nts_high_vol = cgmy_to_nts(C, G_base, M_base, Y, mu, 1.4)
nts_low_vol = cgmy_to_nts(C, G_base, M_base, Y, mu, 0.7)

x_grid = np.linspace(-6, 6, 1201)
u_grid = np.linspace(0.001, 0.999, 999)

# ----------------------------------------
# Figure 2(a): PDFs
# ----------------------------------------
print("Generating Figure 2(a): NTS PDFs...")

pdf_base = dnts(x_grid, nts_base)
pdf_high_vol = dnts(x_grid, nts_high_vol)
pdf_low_vol = dnts(x_grid, nts_low_vol)

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.plot(x_grid, pdf_base, 'k-', linewidth=2, label='Benchmark (sigma=1.0)', alpha=0.8)
ax.plot(x_grid, pdf_high_vol, 'b-', linewidth=2, label='High volatility (sigma=1.4)', alpha=0.8)
ax.plot(x_grid, pdf_low_vol, 'r-', linewidth=2, label='Low volatility (sigma=0.7)', alpha=0.8)

ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('f_X(x)', fontsize=12)
ax.set_title('Figure 2(a): PDFs of Normal Tempered-Stable distributions', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(-6, 6)
ax.set_ylim(0, None)
plt.tight_layout()

save_path = os.path.join(OUTPUT_DIR, 'Figure_2a_NTS_PDFs')
plt.savefig(save_path + '.png', dpi=300, bbox_inches='tight')
plt.savefig(save_path + '.pdf', bbox_inches='tight')
plt.close()
print(f"  Saved: {save_path}.png and .pdf")

# ----------------------------------------
# Figure 2(b): CDFs
# ----------------------------------------
print("Generating Figure 2(b): NTS CDFs...")

cdf_base = pnts(x_grid, nts_base)
cdf_high_vol = pnts(x_grid, nts_high_vol)
cdf_low_vol = pnts(x_grid, nts_low_vol)

fig, ax = plt.subplots(figsize=(6, 4.5))
# Plot the NTS variants
ax.plot(x_grid, cdf_base, 'k-', linewidth=2, label='Benchmark (sigma=1.0)', alpha=0.8, zorder=2)
ax.plot(x_grid, cdf_high_vol, 'b-', linewidth=2, label='High volatility (sigma=1.4)', alpha=0.8, zorder=2)
ax.plot(x_grid, cdf_low_vol, 'r-', linewidth=2, label='Low volatility (sigma=0.7)', alpha=0.8, zorder=2)
# Standard normal reference (dashed) — drawn LAST with a heavier dashed line so it isn't hidden
# behind the benchmark NTS curve (per Lindquist review).
ax.plot(x_grid, _scipy_norm.cdf(x_grid), color='dimgray', linestyle=(0, (6, 4)),
        linewidth=2.2, label='Standard normal (reference)', zorder=3)

ax.set_xlabel('x')
ax.set_ylabel(r'$F_X(x)$')
ax.set_title('CDFs of Normal Tempered-Stable distributions')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(-6, 6)
ax.set_ylim(0, 1)
plt.tight_layout()

save_path = os.path.join(OUTPUT_DIR, 'Figure_2b_NTS_CDFs')
plt.savefig(save_path + '.png', dpi=300, bbox_inches='tight')
plt.savefig(save_path + '.pdf', bbox_inches='tight')
plt.close()
print(f"  Saved: {save_path}.png and .pdf")

# ----------------------------------------
# Figure 2(c): Quantile Functions
# ----------------------------------------
print("Generating Figure 2(c): NTS Quantile Functions...")

q_base = qnts(u_grid, nts_base)
q_high_vol = qnts(u_grid, nts_high_vol)
q_low_vol = qnts(u_grid, nts_low_vol)

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.plot(u_grid, q_base, 'k-', linewidth=2, label='Benchmark (sigma=1.0)', alpha=0.8)
ax.plot(u_grid, q_high_vol, 'b-', linewidth=2, label='High volatility (sigma=1.4)', alpha=0.8)
ax.plot(u_grid, q_low_vol, 'r-', linewidth=2, label='Low volatility (sigma=0.7)', alpha=0.8)

ax.set_xlabel('u', fontsize=12)
ax.set_ylabel('Q(u)', fontsize=12)
ax.set_title('Figure 2(c): Quantile functions of Normal Tempered-Stable distributions', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1)
ax.set_ylim(-6, 6)
plt.tight_layout()

save_path = os.path.join(OUTPUT_DIR, 'Figure_2c_NTS_Quantiles')
plt.savefig(save_path + '.png', dpi=300, bbox_inches='tight')
plt.savefig(save_path + '.pdf', bbox_inches='tight')
plt.close()
print(f"  Saved: {save_path}.png and .pdf")

print("  NTS Fundamentals Complete: All 3 figures generated")

# ============================================================================
# CASE 1: SCALE/VOLATILITY CHANNEL
# ============================================================================
print("\n" + "-"*80)
print("CASE 1: Scale/Volatility Channel")
print("-"*80)

# Case 1 Parameters (from manuscript)
G1 = 5.0
M1 = 5.0
sigma_bench = 1.0  # Benchmark
sigma_fear = 1.4   # Fearful (higher volatility) - from manuscript
sigma_greed = 0.7  # Greedy (lower volatility) - from manuscript

# Convert to NTS parameters
nts_bench = cgmy_to_nts(C, G1, M1, Y, mu, sigma_bench)
nts_fear = cgmy_to_nts(C, G1, M1, Y, mu, sigma_fear)
nts_greed = cgmy_to_nts(C, G1, M1, Y, mu, sigma_greed)

# ----------------------------------------
# Figure 2(d): PWF for Case 1 (Already done, just regenerate for consistency)
# ----------------------------------------
print("Generating Figure 2(d): PWF for Case 1...")

u_grid = np.linspace(0.001, 0.999, 999)
w_fear = compute_pwf(u_grid, nts_bench, nts_fear)
w_greed = compute_pwf(u_grid, nts_bench, nts_greed)

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='45° line', alpha=0.7)
ax.plot(u_grid, w_fear, 'b-', linewidth=2, label='Fearful')
ax.plot(u_grid, w_greed, 'r-', linewidth=2, label='Greedy')

ax.set_xlabel('u', fontsize=12)
ax.set_ylabel('w(u)', fontsize=12)
ax.set_title('Figure 2(d): PWF for Case 1 - Scale/Volatility Channel', fontsize=13)
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')
plt.tight_layout()

save_path = os.path.join(OUTPUT_DIR, 'Figure_2d_Case1_PWF')
plt.savefig(save_path + '.png', dpi=300, bbox_inches='tight')
plt.savefig(save_path + '.pdf', bbox_inches='tight')
plt.close()
print(f"  Saved: {save_path}.png and .pdf")

# ----------------------------------------
# Figure 2(e): Symmetric Greed-Fear Index
# ----------------------------------------
print("Generating Figure 2(e): Symmetric greed-fear index...")

# Exclude endpoints to avoid division by zero
mask = (u_grid > 0.01) & (u_grid < 0.99)
p = u_grid[mask]
gamma_fear = w_fear[mask] / (p * (1 - p))
gamma_greed = w_greed[mask] / (p * (1 - p))

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.plot(p, gamma_fear, 'b-', linewidth=2, label='Fearful (σ = 1.4)')
ax.plot(p, gamma_greed, 'r-', linewidth=2, label='Greedy (σ = 0.7)')

ax.set_xlabel('p', fontsize=12)
ax.set_ylabel('Γ(p)', fontsize=12)
ax.set_title('Figure 2(e): Γ(p) = w(p)/[p(1-p)] for Case 1', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1)
plt.tight_layout()

save_path = os.path.join(OUTPUT_DIR, 'Figure_2e_Case1_GreedFearIndex')
plt.savefig(save_path + '.png', dpi=300, bbox_inches='tight')
plt.savefig(save_path + '.pdf', bbox_inches='tight')
plt.close()
print(f"  Saved: {save_path}.png and .pdf")

# ----------------------------------------
# Figure 2(f): PDFs for Case 1
# ----------------------------------------
print("Generating Figure 2(f): PDFs for Case 1...")

x_grid = np.linspace(-6, 6, 1201)
pdf_bench = dnts(x_grid, nts_bench)
pdf_fear = dnts(x_grid, nts_fear)
pdf_greed = dnts(x_grid, nts_greed)

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.plot(x_grid, pdf_bench, 'k-', linewidth=2, label='Benchmark (σ = 1.0)', alpha=0.8)
ax.plot(x_grid, pdf_fear, 'b-', linewidth=2, label='Fearful (σ = 1.4)', alpha=0.8)
ax.plot(x_grid, pdf_greed, 'r-', linewidth=2, label='Greedy (σ = 0.7)', alpha=0.8)

ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('f_X(x)', fontsize=12)
ax.set_title('Figure 2(f): PDFs for Case 1 (benchmark, greedy, fearful)', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(-6, 6)
ax.set_ylim(0, None)
plt.tight_layout()

save_path = os.path.join(OUTPUT_DIR, 'Figure_2f_Case1_PDFs')
plt.savefig(save_path + '.png', dpi=300, bbox_inches='tight')
plt.savefig(save_path + '.pdf', bbox_inches='tight')
plt.close()
print(f"  Saved: {save_path}.png and .pdf")

print("  Case 1 Complete: All 3 figures generated")

# ============================================================================
# CASE 2: SKEW/ASYMMETRY CHANNEL
# ============================================================================
print("\n" + "-"*80)
print("CASE 2: Skew/Asymmetry Channel")
print("-"*80)

# Case 2 Parameters (from manuscript)
# Benchmark (symmetric)
G0 = 5.0
M0 = 5.0
sigma0 = 1.0

# Greedy (right-skew): G > M
Gg = 9.0
Mg = 3.0
sigma_g = 0.98  # Slight adjustment to preserve variance

# Fearful (left-skew): M > G
Gf = 3.0
Mf = 9.0
sigma_f = 1.02  # Slight adjustment to preserve variance

# Convert to NTS parameters
nts_bench_c2 = cgmy_to_nts(C, G0, M0, Y, mu, sigma0)
nts_greed_c2 = cgmy_to_nts(C, Gg, Mg, Y, mu, sigma_g)
nts_fear_c2 = cgmy_to_nts(C, Gf, Mf, Y, mu, sigma_f)

# ----------------------------------------
# Figure 2(g): PDFs for Case 2
# ----------------------------------------
print("Generating Figure 2(g): PDFs for Case 2...")

pdf_bench_c2 = dnts(x_grid, nts_bench_c2)
pdf_greed_c2 = dnts(x_grid, nts_greed_c2)
pdf_fear_c2 = dnts(x_grid, nts_fear_c2)

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.plot(x_grid, pdf_bench_c2, 'k-', linewidth=2, label='Benchmark (G=M=5)', alpha=0.8)
ax.plot(x_grid, pdf_greed_c2, 'r-', linewidth=2, label='Greedy (G=9, M=3)', alpha=0.8)
ax.plot(x_grid, pdf_fear_c2, 'b-', linewidth=2, label='Fearful (G=3, M=9)', alpha=0.8)

ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('f_X(x)', fontsize=12)
ax.set_title('Figure 2(g): PDFs for Case 2', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(-6, 6)
ax.set_ylim(0, None)
plt.tight_layout()

save_path = os.path.join(OUTPUT_DIR, 'Figure_2g_Case2_PDFs')
plt.savefig(save_path + '.png', dpi=300, bbox_inches='tight')
plt.savefig(save_path + '.pdf', bbox_inches='tight')
plt.close()
print(f"  Saved: {save_path}.png and .pdf")

# ----------------------------------------
# Figure 2(h): CDFs for Case 2
# ----------------------------------------
print("Generating Figure 2(h): CDFs for Case 2...")

cdf_bench_c2 = pnts(x_grid, nts_bench_c2)
cdf_greed_c2 = pnts(x_grid, nts_greed_c2)
cdf_fear_c2 = pnts(x_grid, nts_fear_c2)

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.plot(x_grid, cdf_bench_c2, 'k-', linewidth=2, label='Benchmark', alpha=0.8)
ax.plot(x_grid, cdf_greed_c2, 'r-', linewidth=2, label='Greedy (right-skew)', alpha=0.8)
ax.plot(x_grid, cdf_fear_c2, 'b-', linewidth=2, label='Fearful (left-skew)', alpha=0.8)

ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('F_X(x)', fontsize=12)
ax.set_title('Figure 2(h): CDFs for Case 2', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(-6, 6)
ax.set_ylim(0, 1)
plt.tight_layout()

save_path = os.path.join(OUTPUT_DIR, 'Figure_2h_Case2_CDFs')
plt.savefig(save_path + '.png', dpi=300, bbox_inches='tight')
plt.savefig(save_path + '.pdf', bbox_inches='tight')
plt.close()
print(f"  Saved: {save_path}.png and .pdf")

# ----------------------------------------
# Figure 2(i): PWFs for Case 2
# ----------------------------------------
print("Generating Figure 2(i): PWFs for Case 2...")

w_greed_c2 = compute_pwf(u_grid, nts_bench_c2, nts_greed_c2)
w_fear_c2 = compute_pwf(u_grid, nts_bench_c2, nts_fear_c2)

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='45° line', alpha=0.7)
ax.plot(u_grid, w_greed_c2, 'r-', linewidth=2, label='Greedy (right-skew)')
ax.plot(u_grid, w_fear_c2, 'b-', linewidth=2, label='Fearful (left-skew)')

ax.set_xlabel('u', fontsize=12)
ax.set_ylabel('w(u)', fontsize=12)
ax.set_title('Figure 2(i): PWFs for the skew/asymmetry channel', fontsize=13)
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')
plt.tight_layout()

save_path = os.path.join(OUTPUT_DIR, 'Figure_2i_Case2_PWFs')
plt.savefig(save_path + '.png', dpi=300, bbox_inches='tight')
plt.savefig(save_path + '.pdf', bbox_inches='tight')
plt.close()
print(f"  Saved: {save_path}.png and .pdf")

print("  Case 2 Complete: All 3 figures generated")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("ALL FIGURES GENERATED SUCCESSFULLY")
print("="*80)

print("\nCase 1 (Scale/Volatility Channel):")
print("  - Figure 2(d): PWF (legend fixed, no xi values)")
print("  - Figure 2(e): Symmetric greed-fear index")
print("  - Figure 2(f): PDFs")

print("\nCase 2 (Skew/Asymmetry Channel):")
print("  - Figure 2(g): PDFs")
print("  - Figure 2(h): CDFs")
print("  - Figure 2(i): PWFs")

print(f"\nAll figures saved in: {os.path.abspath(OUTPUT_DIR)}")
print("\nParameters match Dr. Rachev's manuscript specifications exactly:")
print("  Common: mu=0, C=0.6, Y=1.2")
print("  Case 1: G=M=5, sigma varied (1.0, 1.4, 0.7)")
print("  Case 2: G/M varied (9/3, 3/9), sigma adjusted (0.98, 1.02)")
print("\nReady to send to Dr. Rachev!")
print("="*80)