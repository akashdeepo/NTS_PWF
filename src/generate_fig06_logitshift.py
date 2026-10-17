"""
Figure 6: information-theoretic indices for the scale (volatility) channel.

For a PWF w(p) and objective probability p:

1. G_FI(p) = [logit w(p) - logit p] / sqrt(p(1-p))   Fisher-normalized logit shift
2. S_JS(p) = sgn(w(p) - p) * sqrt(2 * JSD(p))        signed Jensen-Shannon divergence
3. E(p)    = d/dp [logit w(p) - logit p]              log-odds elasticity

plus the cumulative integral of each.

Parameters: C=0.6, Y=1.2, G=M=5; benchmark sigma=1.0, fearful sigma=1.4,
greedy sigma=0.7.

Output: outputs/case1_indices/Figure_2e{1,2,3,4}_*.{png,pdf}
The paper uses Figure_2e1_LogitShift_GFI.pdf (Fig 6). Its y-axis is clamped
to [-100, 100] because G_FI diverges as p approaches 0 or 1.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
# Path setup: works from any cwd (script at NTS_PWF/src/)
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, 'lib', 'temStaPy_v0.5'))
from temStaPy.distNTS import qnts, pnts

# Publication style: larger fonts relative to plot, smaller physical figsize
# so the text stays legible when the figure is scaled to 0.7\textwidth.
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 15,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
})

def cgmy_to_nts(C, G, M, Y, mu=0, sigma=1):
    """Convert CGMY parameters to NTS parameters"""
    alpha = Y
    theta = C * (G**Y + M**Y) / Y
    beta_raw = (M**Y - G**Y) / (M**Y + G**Y)
    gamma_param = sigma
    beta = beta_raw * sigma
    return [alpha, theta, beta, gamma_param, mu]

def compute_pwf(u_grid, prior_params, post_params):
    """Compute PWF: w(u) = F_post(Q_prior(u))"""
    q_prior = qnts(u_grid, prior_params)
    w = pnts(q_prior, post_params)
    return w

def safe_clip(arr, epsilon=1e-6):
    """Clip array to [epsilon, 1-epsilon]"""
    return np.clip(arr, epsilon, 1-epsilon)

def logit_shift_GFI(w, p, epsilon=1e-6):
    """
    Logit-shift greed-fear index with Fisher-information normalization
    G_FI(p) = [logit(w(p)) - logit(p)] / sqrt[p(1-p)]
    """
    w_clip = safe_clip(w, epsilon)
    p_clip = safe_clip(p, epsilon)

    logit_w = np.log(w_clip / (1 - w_clip))
    logit_p = np.log(p_clip / (1 - p_clip))

    G = logit_w - logit_p
    G_FI = G / np.sqrt(p_clip * (1 - p_clip))

    return G_FI

def kl_divergence_bernoulli(a, b, epsilon=1e-6):
    """KL divergence between Bernoulli(a) and Bernoulli(b)"""
    a_clip = safe_clip(a, epsilon)
    b_clip = safe_clip(b, epsilon)

    term1 = a_clip * np.log(a_clip / b_clip)
    term2 = (1 - a_clip) * np.log((1 - a_clip) / (1 - b_clip))

    return term1 + term2

def signed_jensen_shannon(w, p, epsilon=1e-6):
    """
    Signed Jensen-Shannon index
    S_JS(p) = sgn(w(p) - p) * sqrt(2 * JSD(p))
    """
    w_clip = safe_clip(w, epsilon)
    p_clip = safe_clip(p, epsilon)

    m = (w_clip + p_clip) / 2.0

    kl_wm = kl_divergence_bernoulli(w_clip, m, epsilon)
    kl_pm = kl_divergence_bernoulli(p_clip, m, epsilon)

    JSD = 0.5 * (kl_wm + kl_pm)

    sign = np.sign(w_clip - p_clip)
    S_JS = sign * np.sqrt(2 * JSD)

    return S_JS

def log_odds_elasticity(w, p, epsilon=1e-6):
    """
    Log-odds elasticity
    E(p) = d/dp[logit(w(p)) - logit(p)]
    """
    w_clip = safe_clip(w, epsilon)
    p_clip = safe_clip(p, epsilon)

    # Compute derivative of w using gradient
    dw_dp = np.gradient(w_clip, p_clip, edge_order=2)

    # E(p) = w'(p) / [w(p)(1-w(p))] - 1 / [p(1-p)]
    term1 = dw_dp / (w_clip * (1 - w_clip))
    term2 = 1.0 / (p_clip * (1 - p_clip))

    E = term1 - term2

    return E

OUTPUT_DIR = os.path.join(_ROOT, "outputs", "case1_indices")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*80)
print("GENERATING NEW FIGURE 2(e) - INFORMATION-THEORETIC INDICES")
print("="*80)

# Case 1 parameters
mu, C, Y = 0.0, 0.6, 1.2
G1, M1 = 5.0, 5.0
sigma_bench = 1.0
sigma_fear = 1.4
sigma_greed = 0.7

nts_bench = cgmy_to_nts(C, G1, M1, Y, mu, sigma_bench)
nts_fear = cgmy_to_nts(C, G1, M1, Y, mu, sigma_fear)
nts_greed = cgmy_to_nts(C, G1, M1, Y, mu, sigma_greed)

# Compute PWFs on fine grid
N = 2000
p_grid = np.linspace(1.0/(N+1), N/(N+1), N)

print(f"\nComputing PWFs on grid with {N} points...")
w_bench = p_grid  # Benchmark is identity
w_fear = compute_pwf(p_grid, nts_bench, nts_fear)
w_greed = compute_pwf(p_grid, nts_bench, nts_greed)

# ============================================================================
# Figure 2(e1): Logit-shift greed-fear index G_FI(p)
# ============================================================================
print("\nGenerating Figure 2(e1): Logit-shift index G_FI(p)...")

G_FI_bench = logit_shift_GFI(w_bench, p_grid)
G_FI_fear = logit_shift_GFI(w_fear, p_grid)
G_FI_greed = logit_shift_GFI(w_greed, p_grid)

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5, label='Zero line')
ax.plot(p_grid, G_FI_bench, 'k-', linewidth=1.5, label='Benchmark', alpha=0.5)
ax.plot(p_grid, G_FI_fear, 'b-', linewidth=2, label='Fearful (sigma=1.4)')
ax.plot(p_grid, G_FI_greed, 'r-', linewidth=2, label='Greedy (sigma=0.7)')

ax.set_xlabel('p', fontsize=12)
ax.set_ylabel('G_FI(p)', fontsize=12)
ax.set_title('Figure 2(e1): Logit-shift greed-fear index', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1)
ax.set_ylim(-100, 100)  # G_FI diverges near p=0 and p=1; clamp for readability
plt.tight_layout()

save_path = os.path.join(OUTPUT_DIR, 'Figure_2e1_LogitShift_GFI')
plt.savefig(save_path + '.png', dpi=300, bbox_inches='tight')
plt.savefig(save_path + '.pdf', bbox_inches='tight')
plt.close()
print(f"  Saved: {save_path}.png and .pdf")

# Compute aggregate scores
G_FI_score_fear = np.trapezoid(G_FI_fear, p_grid)
G_FI_score_greed = np.trapezoid(G_FI_greed, p_grid)
print(f"  Aggregate G_FI scores: Fearful={G_FI_score_fear:.4f}, Greedy={G_FI_score_greed:.4f}")

# ============================================================================
# Figure 2(e2): Signed Jensen-Shannon index S_JS(p)
# ============================================================================
print("\nGenerating Figure 2(e2): Signed Jensen-Shannon index S_JS(p)...")

S_JS_bench = signed_jensen_shannon(w_bench, p_grid)
S_JS_fear = signed_jensen_shannon(w_fear, p_grid)
S_JS_greed = signed_jensen_shannon(w_greed, p_grid)

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5, label='Zero line')
ax.plot(p_grid, S_JS_bench, 'k-', linewidth=1.5, label='Benchmark', alpha=0.5)
ax.plot(p_grid, S_JS_fear, 'b-', linewidth=2, label='Fearful (sigma=1.4)')
ax.plot(p_grid, S_JS_greed, 'r-', linewidth=2, label='Greedy (sigma=0.7)')

ax.set_xlabel('p', fontsize=12)
ax.set_ylabel('S_JS(p)', fontsize=12)
ax.set_title('Figure 2(e2): Signed Jensen-Shannon index', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1)
ax.set_ylim(-0.9, 0.9)  # Bounded by ln(2) ≈ 0.8326
plt.tight_layout()

save_path = os.path.join(OUTPUT_DIR, 'Figure_2e2_JensenShannon_SJS')
plt.savefig(save_path + '.png', dpi=300, bbox_inches='tight')
plt.savefig(save_path + '.pdf', bbox_inches='tight')
plt.close()
print(f"  Saved: {save_path}.png and .pdf")

# Compute aggregate scores
S_JS_score_fear = np.trapezoid(S_JS_fear, p_grid)
S_JS_score_greed = np.trapezoid(S_JS_greed, p_grid)
print(f"  Aggregate S_JS scores: Fearful={S_JS_score_fear:.4f}, Greedy={S_JS_score_greed:.4f}")

# ============================================================================
# Figure 2(e3): Log-odds elasticity E(p)
# ============================================================================
print("\nGenerating Figure 2(e3): Log-odds elasticity E(p)...")

E_bench = log_odds_elasticity(w_bench, p_grid)
E_fear = log_odds_elasticity(w_fear, p_grid)
E_greed = log_odds_elasticity(w_greed, p_grid)

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5, label='Zero line')
ax.plot(p_grid, E_bench, 'k-', linewidth=1.5, label='Benchmark', alpha=0.5)
ax.plot(p_grid, E_fear, 'b-', linewidth=2, label='Fearful (sigma=1.4)')
ax.plot(p_grid, E_greed, 'r-', linewidth=2, label='Greedy (sigma=0.7)')

ax.set_xlabel('p', fontsize=12)
ax.set_ylabel('E(p)', fontsize=12)
ax.set_title('Figure 2(e3): Log-odds elasticity', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1)
plt.tight_layout()

save_path = os.path.join(OUTPUT_DIR, 'Figure_2e3_Elasticity_E')
plt.savefig(save_path + '.png', dpi=300, bbox_inches='tight')
plt.savefig(save_path + '.pdf', bbox_inches='tight')
plt.close()
print(f"  Saved: {save_path}.png and .pdf")

# ============================================================================
# Figure 2(e4): Cumulative greed-fear scores
# ============================================================================
print("\nGenerating Figure 2(e4): Cumulative scores...")

# Cumulative integrals
C_GFI_fear = np.cumsum(G_FI_fear) * (p_grid[1] - p_grid[0])
C_GFI_greed = np.cumsum(G_FI_greed) * (p_grid[1] - p_grid[0])

C_SJS_fear = np.cumsum(S_JS_fear) * (p_grid[1] - p_grid[0])
C_SJS_greed = np.cumsum(S_JS_greed) * (p_grid[1] - p_grid[0])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Cumulative G_FI
ax1.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
ax1.plot(p_grid, C_GFI_fear, 'b-', linewidth=2, label='Fearful')
ax1.plot(p_grid, C_GFI_greed, 'r-', linewidth=2, label='Greedy')
ax1.set_xlabel('p', fontsize=12)
ax1.set_ylabel('Cumulative G_FI(p)', fontsize=12)
ax1.set_title('Cumulative Logit-shift', fontsize=11)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 1)

# Cumulative S_JS
ax2.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
ax2.plot(p_grid, C_SJS_fear, 'b-', linewidth=2, label='Fearful')
ax2.plot(p_grid, C_SJS_greed, 'r-', linewidth=2, label='Greedy')
ax2.set_xlabel('p', fontsize=12)
ax2.set_ylabel('Cumulative S_JS(p)', fontsize=12)
ax2.set_title('Cumulative Jensen-Shannon', fontsize=11)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 1)

plt.tight_layout()

save_path = os.path.join(OUTPUT_DIR, 'Figure_2e4_Cumulative_Scores')
plt.savefig(save_path + '.png', dpi=300, bbox_inches='tight')
plt.savefig(save_path + '.pdf', bbox_inches='tight')
plt.close()
print(f"  Saved: {save_path}.png and .pdf")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("ALL NEW FIGURE 2(e) VARIANTS GENERATED SUCCESSFULLY")
print("="*80)
print("\nGenerated 4 new figures:")
print("  1. Figure 2(e1): Logit-shift index G_FI(p)")
print("  2. Figure 2(e2): Signed Jensen-Shannon index S_JS(p)")
print("  3. Figure 2(e3): Log-odds elasticity E(p)")
print("  4. Figure 2(e4): Cumulative greed-fear scores")
print("\nAggregate scores:")
print(f"  G_FI:  Fearful={G_FI_score_fear:+.4f}, Greedy={G_FI_score_greed:+.4f}")
print(f"  S_JS:  Fearful={S_JS_score_fear:+.4f}, Greedy={S_JS_score_greed:+.4f}")
print("\nThese should show MUCH better separation than the old Gamma(p) index!")
print("="*80)
