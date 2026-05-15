"""
Generate Clean Zoomed Publication Figure for Case 2
====================================================

Creates a single, clean publication-quality figure showing:
- PWF Deviations with enhanced visualization
- Minimal clutter, maximum clarity
- Publication-ready formatting

Author: Akash Deep
Date: October 23, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Path setup: works from any cwd (script at NTS_PWF/src/)
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, 'lib', 'temStaPy_v0.5'))
from temStaPy.distNTS import dnts, pnts, qnts

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

# Output directory
OUTPUT_DIR = os.path.join(_ROOT, "outputs", "case2_deviation")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def cgmy_to_nts(C, G, M, Y, mu=0, sigma=1):
    """Convert CGMY parameters to NTS parameters"""
    alpha = Y
    theta = C * (G**Y + M**Y) / Y
    beta_raw = (M**Y - G**Y) / (M**Y + G**Y)
    gamma_param = sigma
    beta = beta_raw * sigma
    return [alpha, theta, beta, gamma_param, mu]

# Parameters
mu, C, Y = 0.0, 0.6, 1.2
G_bench, M_bench, sigma_bench = 5.0, 5.0, 1.00
G_fear, M_fear, sigma_fear = 1.0, 15.0, 0.85
G_greed, M_greed, sigma_greed = 15.0, 1.0, 0.85

# Convert to NTS
nts_bench = cgmy_to_nts(C, G_bench, M_bench, Y, mu, sigma_bench)
nts_fear = cgmy_to_nts(C, G_fear, M_fear, Y, mu, sigma_fear)
nts_greed = cgmy_to_nts(C, G_greed, M_greed, Y, mu, sigma_greed)

# Compute PWFs
N_u = 2000
u_grid = np.linspace(1.0/(N_u+1), N_u/(N_u+1), N_u)
q_bench = qnts(u_grid, nts_bench)
w_fear = pnts(q_bench, nts_fear)
w_greed = pnts(q_bench, nts_greed)

# Deviations
dev_fear = w_fear - u_grid
dev_greed = w_greed - u_grid

# Statistics
dev_fear_max = np.max(np.abs(dev_fear))
dev_greed_max = np.max(np.abs(dev_greed))
idx_fear_max = np.argmax(np.abs(dev_fear))
idx_greed_max = np.argmax(np.abs(dev_greed))

print("="*80)
print("GENERATING CLEAN ZOOMED DEVIATION FIGURE")
print("="*80)
print(f"\nMax Deviations:")
print(f"  Fearful: {100*dev_fear_max:.2f}% at u={u_grid[idx_fear_max]:.3f}")
print(f"  Greedy:  {100*dev_greed_max:.2f}% at u={u_grid[idx_greed_max]:.3f}")
print()

# ============================================================================
# CLEAN PUBLICATION FIGURE
# ============================================================================

fig, ax = plt.subplots(figsize=(7, 4.5))

# Zero reference line
ax.axhline(y=0, color='gray', linestyle='-', linewidth=2, alpha=0.4, zorder=1)

# Deviation curves - thicker, clearer
ax.plot(u_grid, 100*dev_fear, 'b-', linewidth=3.5, label='Fearful (left-skew)', zorder=3)
ax.plot(u_grid, 100*dev_greed, 'r-', linewidth=3.5, label='Greedy (right-skew)', zorder=3)

# Mark maximum deviations with larger, clearer markers
ax.plot(u_grid[idx_fear_max], 100*dev_fear[idx_fear_max], 'bo',
        markersize=14, markeredgewidth=2, markeredgecolor='darkblue',
        label=f'Fearful max: {100*dev_fear_max:.2f}% at u={u_grid[idx_fear_max]:.2f}',
        zorder=5)
ax.plot(u_grid[idx_greed_max], 100*dev_greed[idx_greed_max], 'ro',
        markersize=14, markeredgewidth=2, markeredgecolor='darkred',
        label=f'Greedy max: {100*dev_greed_max:.2f}% at u={u_grid[idx_greed_max]:.2f}',
        zorder=5)

# Subtle shading for interpretation zones
ax.axhspan(0, 4, alpha=0.05, color='red', zorder=0)
ax.axhspan(-4, 0, alpha=0.05, color='blue', zorder=0)

# Labels and title (fontsizes calibrated so 7x4.5" figure prints cleanly
# at width=0.7\textwidth without the title or y-label being clipped)
ax.set_xlabel('Objective probability $u$', fontsize=12)
ax.set_ylabel(r'PWF deviation $100\times[w(u)-u]$ (%)', fontsize=12)
ax.set_title('PWF Deviations — Skew Channel ($G/M=15/1$, $\\sigma=0.85$)',
             fontsize=12, pad=8)

# Legend
ax.legend(fontsize=10, loc='upper left', framealpha=0.95, edgecolor='black',
          fancybox=True, shadow=False)

# Grid
ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8, zorder=0)
ax.set_xlim(0, 1)
ax.set_ylim(-4.5, 4.5)

# Tick formatting
ax.tick_params(axis='both', which='major', labelsize=11, width=1.2, length=5)

plt.tight_layout()

# Save
save_path = os.path.join(OUTPUT_DIR, 'Figure_2i_CLEAN_Deviation')
plt.savefig(save_path + '.png', dpi=400, bbox_inches='tight', facecolor='white')
plt.savefig(save_path + '.pdf', bbox_inches='tight', facecolor='white')
plt.close()

print(f"Saved: {save_path}.png (400 DPI)")
print(f"Saved: {save_path}.pdf (vector)")
print("="*80)

# ============================================================================
# ZOOMED VERSION: Focus on [0.1, 0.9]
# ============================================================================

fig, ax = plt.subplots(figsize=(7, 4.5))

# Mask for zoomed region
zoom_mask = (u_grid >= 0.1) & (u_grid <= 0.9)
u_zoom = u_grid[zoom_mask]
dev_fear_zoom = dev_fear[zoom_mask]
dev_greed_zoom = dev_greed[zoom_mask]

# Zero reference line
ax.axhline(y=0, color='gray', linestyle='-', linewidth=2, alpha=0.4, zorder=1)

# Deviation curves
ax.plot(u_zoom, 100*dev_fear_zoom, 'b-', linewidth=4, label='Fearful (left-skew)', zorder=3)
ax.plot(u_zoom, 100*dev_greed_zoom, 'r-', linewidth=4, label='Greedy (right-skew)', zorder=3)

# Find max in zoomed region
idx_fear_zoom = np.argmin(dev_fear_zoom)  # Most negative
idx_greed_zoom = np.argmax(dev_greed_zoom)  # Most positive

ax.plot(u_zoom[idx_fear_zoom], 100*dev_fear_zoom[idx_fear_zoom], 'bo',
        markersize=16, markeredgewidth=2.5, markeredgecolor='darkblue',
        zorder=5)
ax.plot(u_zoom[idx_greed_zoom], 100*dev_greed_zoom[idx_greed_zoom], 'ro',
        markersize=16, markeredgewidth=2.5, markeredgecolor='darkred',
        zorder=5)

# Shading
ax.axhspan(0, 4, alpha=0.06, color='red', zorder=0)
ax.axhspan(-4, 0, alpha=0.06, color='blue', zorder=0)

# Labels
ax.set_xlabel('Objective probability $u$', fontsize=12)
ax.set_ylabel(r'PWF deviation $100\times[w(u)-u]$ (%)', fontsize=12)
ax.set_title('PWF Deviations (zoomed view) — Skew Channel ($G/M=15/1$, $\\sigma=0.85$)',
             fontsize=12, pad=8)

# Legend
ax.legend(fontsize=10, loc='upper left', framealpha=0.95, edgecolor='black',
          fancybox=True, shadow=False)

# Grid
ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8, zorder=0)
ax.set_xlim(0.1, 0.9)

# Auto y-limits based on data
y_margin = 0.3
ax.set_ylim(100*np.min(dev_fear_zoom) - y_margin,
            100*np.max(dev_greed_zoom) + y_margin)

# Tick formatting
ax.tick_params(axis='both', which='major', labelsize=11, width=1.2, length=5)

plt.tight_layout()

# Save
save_path = os.path.join(OUTPUT_DIR, 'Figure_2i_CLEAN_Deviation_ZOOMED')
plt.savefig(save_path + '.png', dpi=400, bbox_inches='tight', facecolor='white')
plt.savefig(save_path + '.pdf', bbox_inches='tight', facecolor='white')
plt.close()

print(f"\nSaved: {save_path}.png (400 DPI)")
print(f"Saved: {save_path}.pdf (vector)")
print("="*80)

# ============================================================================
# ULTRA-CLEAN VERSION: Minimal annotations
# ============================================================================

fig, ax = plt.subplots(figsize=(7, 4.2))

# Zero line
ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.6, zorder=1)

# Just the curves - clean and simple
ax.plot(u_grid, 100*dev_fear, 'b-', linewidth=3, label='Fearful', zorder=3)
ax.plot(u_grid, 100*dev_greed, 'r-', linewidth=3, label='Greedy', zorder=3)

# Labels
ax.set_xlabel('Probability u', fontsize=15)
ax.set_ylabel('Deviation from Identity (%)', fontsize=15)
ax.set_title('PWF Deviations: Skew Channel', fontsize=16, fontweight='bold')

# Legend
ax.legend(fontsize=13, loc='upper left', frameon=True)

# Grid
ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
ax.set_xlim(0, 1)
ax.set_ylim(-4, 4)

# Formatting
ax.tick_params(axis='both', labelsize=12)

plt.tight_layout()

# Save
save_path = os.path.join(OUTPUT_DIR, 'Figure_2i_ULTRACLEAN_Deviation')
plt.savefig(save_path + '.png', dpi=400, bbox_inches='tight', facecolor='white')
plt.savefig(save_path + '.pdf', bbox_inches='tight', facecolor='white')
plt.close()

print(f"\nSaved: {save_path}.png (400 DPI, ultra-clean)")
print(f"Saved: {save_path}.pdf (vector, ultra-clean)")
print("\n" + "="*80)
print("ALL CLEAN FIGURES GENERATED SUCCESSFULLY")
print("="*80)
print("\nGenerated 3 versions:")
print("  1. Figure_2i_CLEAN_Deviation - Full range with annotations")
print("  2. Figure_2i_CLEAN_Deviation_ZOOMED - Focused on [0.1, 0.9]")
print("  3. Figure_2i_ULTRACLEAN_Deviation - Minimal design")
print("\nAll at 400 DPI for maximum publication quality")
print("="*80)
