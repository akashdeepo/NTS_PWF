"""
Empirical NTS-PWF Analysis: SPY Returns
========================================

This script performs the empirical analysis requested by Dr. Rachev:
- Fit NTS parameters to SPY daily log-returns (P-measure)
- Create channel perturbations (scale, skew, tail)
- Compute PWFs and information-theoretic indices
- Generate publication-ready figures

Authors: Akash Deep et al.
Date: December 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import sys
import os
from datetime import datetime

# =============================================================================
# PATH SETUP - Add temStaPy and nts_utils to path
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # parent of src/ = repo root
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'lib', 'temStaPy_v0.5'))
sys.path.insert(0, SCRIPT_DIR)  # for nts_utils import

from temStaPy.distNTS import dnts, pnts, qnts, fitnts, moments_NTS
from nts_utils import (
    compute_pwf,
    logit_shift_GFI,
    signed_jensen_shannon,
    log_odds_elasticity,
    safe_clip
)

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Plotting style — bumped per second Lindquist review ("the text on figures
# is too small"). Combined with the smaller per-panel figsizes below, the
# fonts now print at a comfortable size when the figure is scaled to
# width=0.7\textwidth in LaTeX.
plt.rcParams.update({
    'font.size': 18,
    'axes.labelsize': 18,
    'axes.titlesize': 20,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# =============================================================================
# DATA LOADING
# =============================================================================
def load_spy_data():
    """Load SPY returns data from Bloomberg export."""
    print("Loading SPY data...")

    spy_path = os.path.join(DATA_DIR, 'spy_prices.csv')
    df = pd.read_csv(spy_path, parse_dates=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # Use pre-computed log returns, drop NaN
    returns = df['log_ret'].dropna().values

    print(f"  Loaded {len(returns)} daily log-returns")
    print(f"  Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"  Sample mean: {returns.mean()*100:.4f}% daily")
    print(f"  Sample std:  {returns.std()*100:.4f}% daily")
    print(f"  Sample skew: {pd.Series(returns).skew():.4f}")
    print(f"  Sample kurt: {pd.Series(returns).kurtosis():.4f} (excess)")

    return df, returns


# =============================================================================
# NTS CALIBRATION
# =============================================================================
def fit_nts_to_returns(returns, maxeval=500):
    """
    Fit NTS distribution to log-returns using temStaPy's fitnts.

    Returns NTS parameters: [alpha, theta, beta, gamma, mu]
    """
    print("\nFitting NTS distribution to SPY returns...")
    print("  (This may take a minute...)")

    # fitnts expects raw data and returns [alpha, theta, beta, gamma, mu]
    fitted_params = fitnts(returns, maxeval=maxeval)

    alpha, theta, beta, gamma, mu = fitted_params

    print("\n  Fitted NTS Parameters:")
    print(f"    alpha (tail index):    {alpha:.6f}")
    print(f"    theta (tempering):     {theta:.6f}")
    print(f"    beta  (skewness):      {beta:.6f}")
    print(f"    gamma (scale):         {gamma:.6f}")
    print(f"    mu    (location):      {mu:.6f}")

    # Compute moments from fitted distribution
    moments = moments_NTS(fitted_params)
    print("\n  Fitted Distribution Moments:")
    print(f"    Mean:     {moments['mean']:.6f}")
    print(f"    Variance: {moments['variance']:.6f}")
    print(f"    Skewness: {moments['skewness']:.4f}")
    print(f"    Kurtosis: {moments['excess kurtosis']:.4f} (excess)")

    return fitted_params, moments


# =============================================================================
# CHANNEL PERTURBATIONS
# =============================================================================
def create_channel_perturbations(fitted_params):
    """
    Create benchmark + fearful + greedy perturbations for each channel.

    Following Dr. Rachev's framework:
    - Scale channel: vary gamma (σ) with factors 0.7, 1.0, 1.4
    - Skew channel: vary beta while adjusting gamma for variance matching
    - Tail channel: vary alpha (Y) with values 0.9, 1.2, 1.6
    """
    alpha, theta, beta, gamma, mu = fitted_params

    channels = {}

    # -------------------------------------------------------------------------
    # SCALE CHANNEL (volatility variation)
    # -------------------------------------------------------------------------
    scale_factors = {'greedy': 0.7, 'benchmark': 1.0, 'fearful': 1.4}
    channels['scale'] = {}
    for name, factor in scale_factors.items():
        channels['scale'][name] = [alpha, theta, beta, gamma * factor, mu]

    print("\n  Scale Channel Parameters:")
    for name, params in channels['scale'].items():
        print(f"    {name:10s}: gamma = {params[3]:.6f}")

    # -------------------------------------------------------------------------
    # SKEW CHANNEL (asymmetry variation)
    # -------------------------------------------------------------------------
    # Vary beta: negative = left-skew (fearful), positive = right-skew (greedy)
    # Keep variance approximately matched by adjusting gamma

    beta_variations = {
        'fearful': beta - 0.005,    # More negative skew (left tail)
        'benchmark': beta,
        'greedy': beta + 0.005      # More positive skew (right tail)
    }

    channels['skew'] = {}
    for name, new_beta in beta_variations.items():
        # Simple variance matching: keep gamma same for now
        # (More sophisticated matching would compute exact gamma adjustment)
        channels['skew'][name] = [alpha, theta, new_beta, gamma, mu]

    print("\n  Skew Channel Parameters:")
    for name, params in channels['skew'].items():
        print(f"    {name:10s}: beta = {params[2]:.6f}")

    # -------------------------------------------------------------------------
    # TAIL CHANNEL (tail thickness variation)
    # -------------------------------------------------------------------------
    # Vary alpha: lower = thicker tails (fearful), higher = thinner tails (greedy)
    # Constrain to (0, 2) for valid NTS

    alpha_variations = {
        'fearful': max(0.5, alpha - 0.3),     # Thicker tails
        'benchmark': alpha,
        'greedy': min(1.8, alpha + 0.3)       # Thinner tails
    }

    channels['tail'] = {}
    for name, new_alpha in alpha_variations.items():
        channels['tail'][name] = [new_alpha, theta, beta, gamma, mu]

    print("\n  Tail Channel Parameters:")
    for name, params in channels['tail'].items():
        print(f"    {name:10s}: alpha = {params[0]:.6f}")

    return channels


# =============================================================================
# PWF AND INDICES COMPUTATION
# =============================================================================
def compute_pwf_and_indices(channels, n_points=500):
    """
    Compute PWFs and information-theoretic indices for all channels.
    """
    print("\nComputing PWFs and indices...")

    # Probability grid (avoid exact 0 and 1)
    p_grid = np.linspace(0.001, 0.999, n_points)

    results = {}

    for channel_name, scenarios in channels.items():
        print(f"  Processing {channel_name} channel...")

        results[channel_name] = {'p': p_grid}
        benchmark_params = scenarios['benchmark']

        for scenario_name, params in scenarios.items():
            # Compute PWF: w(p) = F_post(Q_prior(p))
            # Prior = benchmark, Posterior = perturbed scenario
            try:
                w = compute_pwf(p_grid, benchmark_params, params)
                w = safe_clip(w)  # Numerical safety

                # Compute indices
                G_FI = logit_shift_GFI(w, p_grid)
                S_JS = signed_jensen_shannon(w, p_grid)
                E = log_odds_elasticity(w, p_grid)

                results[channel_name][scenario_name] = {
                    'params': params,
                    'w': w,
                    'G_FI': G_FI,
                    'S_JS': S_JS,
                    'E': E
                }

            except Exception as e:
                print(f"    Warning: Error computing {scenario_name}: {e}")
                continue

    print("  Done.")
    return results, p_grid


# =============================================================================
# FIGURE GENERATION
# =============================================================================
def plot_fitted_distribution(returns, fitted_params, output_dir):
    """
    Plot histogram of returns vs fitted NTS density.
    """
    print("\nGenerating Figure: Fitted NTS Distribution...")

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    # Left: Histogram with fitted PDF
    ax1 = axes[0]

    # Histogram
    counts, bins, _ = ax1.hist(returns, bins=100, density=True, alpha=0.7,
                               color='steelblue', edgecolor='white', label='SPY Returns')

    # Fitted NTS PDF
    x_range = np.linspace(returns.min(), returns.max(), 500)
    pdf_fitted = dnts(x_range, fitted_params)
    ax1.plot(x_range, pdf_fitted, 'r-', lw=2, label='Fitted NTS')

    ax1.set_xlabel('Daily Log-Return')
    ax1.set_ylabel('Density')
    ax1.set_title('SPY Returns vs Fitted NTS Distribution')
    ax1.legend()
    ax1.set_xlim([-0.06, 0.06])

    # Right: Log-scale to show tails
    ax2 = axes[1]

    ax2.hist(returns, bins=100, density=True, alpha=0.7,
             color='steelblue', edgecolor='white', label='SPY Returns')
    ax2.plot(x_range, pdf_fitted, 'r-', lw=2, label='Fitted NTS')

    ax2.set_yscale('log')
    ax2.set_xlabel('Daily Log-Return')
    ax2.set_ylabel('Density (log scale)')
    ax2.set_title('Tail Behavior (Log Scale)')
    ax2.legend()
    ax2.set_xlim([-0.12, 0.12])
    ax2.set_ylim([1e-3, 100])

    plt.tight_layout()

    # Save
    png_path = os.path.join(output_dir, 'Figure_Empirical_NTS_Fit.png')
    pdf_path = os.path.join(output_dir, 'Figure_Empirical_NTS_Fit.pdf')
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)

    print(f"  Saved: {png_path}")
    return png_path


def plot_pwf_channels(results, p_grid, output_dir):
    """
    Plot PWFs for all three channels (3x1 subplot).
    """
    print("\nGenerating Figure: PWF Channels...")

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.5))

    channel_titles = {
        'scale': 'Scale Channel (Volatility)',
        'skew': 'Skew Channel (Asymmetry)',
        'tail': 'Tail Channel (Thickness)'
    }

    colors = {'benchmark': 'black', 'fearful': 'red', 'greedy': 'green'}
    linestyles = {'benchmark': '-', 'fearful': '--', 'greedy': '-.'}

    for idx, (channel_name, channel_results) in enumerate(results.items()):
        ax = axes[idx]

        # Plot diagonal reference
        ax.plot([0, 1], [0, 1], 'k:', alpha=0.5, lw=1, label='_nolegend_')

        for scenario_name in ['benchmark', 'fearful', 'greedy']:
            if scenario_name in channel_results and scenario_name != 'p':
                w = channel_results[scenario_name]['w']
                ax.plot(p_grid, w,
                       color=colors[scenario_name],
                       linestyle=linestyles[scenario_name],
                       lw=2, label=scenario_name.capitalize())

        ax.set_xlabel('Prior probability $p$')
        ax.set_ylabel('$w(p)$')
        ax.set_title(channel_titles.get(channel_name, channel_name))
        ax.legend(loc='lower right')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    png_path = os.path.join(output_dir, 'Figure_Empirical_PWF_Channels.png')
    pdf_path = os.path.join(output_dir, 'Figure_Empirical_PWF_Channels.pdf')
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)

    print(f"  Saved: {png_path}")
    return png_path


def plot_pwf_deviations(results, p_grid, output_dir):
    """
    Plot PWF deviations w(p) - p for all channels.
    """
    print("\nGenerating Figure: PWF Deviations...")

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.5))

    channel_titles = {
        'scale': 'Scale Channel',
        'skew': 'Skew Channel',
        'tail': 'Tail Channel'
    }

    colors = {'benchmark': 'black', 'fearful': 'red', 'greedy': 'green'}
    linestyles = {'benchmark': '-', 'fearful': '--', 'greedy': '-.'}

    for idx, (channel_name, channel_results) in enumerate(results.items()):
        ax = axes[idx]

        # Zero reference line
        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5, lw=1)

        for scenario_name in ['benchmark', 'fearful', 'greedy']:
            if scenario_name in channel_results and scenario_name != 'p':
                w = channel_results[scenario_name]['w']
                deviation = w - p_grid
                ax.plot(p_grid, deviation * 100,  # Convert to percentage
                       color=colors[scenario_name],
                       linestyle=linestyles[scenario_name],
                       lw=2, label=scenario_name.capitalize())

        ax.set_xlabel('Prior probability $p$')
        ax.set_ylabel('$w(p) - p$ (%)')
        ax.set_title(channel_titles.get(channel_name, channel_name))
        ax.legend(loc='best')
        ax.set_xlim([0, 1])
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    png_path = os.path.join(output_dir, 'Figure_Empirical_PWF_Deviations.png')
    pdf_path = os.path.join(output_dir, 'Figure_Empirical_PWF_Deviations.pdf')
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)

    print(f"  Saved: {png_path}")
    return png_path


def plot_indices(results, p_grid, output_dir):
    """
    Plot information-theoretic indices (G_FI, S_JS, E) for scale channel.
    """
    print("\nGenerating Figure: Information-Theoretic Indices...")

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.5))

    # Use scale channel (most dramatic effects)
    channel_results = results['scale']

    colors = {'benchmark': 'black', 'fearful': 'red', 'greedy': 'green'}
    linestyles = {'benchmark': '-', 'fearful': '--', 'greedy': '-.'}

    index_names = ['G_FI', 'S_JS', 'E']
    index_titles = [
        r'Logit-Shift Index $G_{FI}(p)$',
        r'Jensen-Shannon Index $S_{JS}(p)$',
        r'Elasticity $E(p)$'
    ]

    for idx, (index_name, title) in enumerate(zip(index_names, index_titles)):
        ax = axes[idx]

        # Zero reference
        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5, lw=1)

        for scenario_name in ['benchmark', 'fearful', 'greedy']:
            if scenario_name in channel_results and scenario_name != 'p':
                index_values = channel_results[scenario_name][index_name]

                # Clip extreme values for visualization
                if index_name == 'G_FI':
                    index_values = np.clip(index_values, -300, 300)
                elif index_name == 'E':
                    index_values = np.clip(index_values, -50, 50)

                ax.plot(p_grid, index_values,
                       color=colors[scenario_name],
                       linestyle=linestyles[scenario_name],
                       lw=2, label=scenario_name.capitalize())

        ax.set_xlabel('Prior probability $p$')
        ax.set_ylabel(index_name.replace('_', '$_{') + '}$')
        ax.set_title(title)
        ax.legend(loc='best')
        ax.set_xlim([0, 1])
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    png_path = os.path.join(output_dir, 'Figure_Empirical_Indices.png')
    pdf_path = os.path.join(output_dir, 'Figure_Empirical_Indices.pdf')
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)

    print(f"  Saved: {png_path}")
    return png_path


def plot_all_channels_indices(results, p_grid, output_dir):
    """
    Plot G_FI index for all three channels (comparison).
    """
    print("\nGenerating Figure: G_FI Across All Channels...")

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.5))

    channel_titles = {
        'scale': 'Scale Channel',
        'skew': 'Skew Channel',
        'tail': 'Tail Channel'
    }

    colors = {'benchmark': 'black', 'fearful': 'red', 'greedy': 'green'}
    linestyles = {'benchmark': '-', 'fearful': '--', 'greedy': '-.'}

    for idx, (channel_name, channel_results) in enumerate(results.items()):
        ax = axes[idx]

        # Zero reference
        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5, lw=1)

        for scenario_name in ['benchmark', 'fearful', 'greedy']:
            if scenario_name in channel_results and scenario_name != 'p':
                G_FI = channel_results[scenario_name]['G_FI']
                G_FI_clipped = np.clip(G_FI, -300, 300)

                ax.plot(p_grid, G_FI_clipped,
                       color=colors[scenario_name],
                       linestyle=linestyles[scenario_name],
                       lw=2, label=scenario_name.capitalize())

        ax.set_xlabel('Prior probability $p$')
        ax.set_ylabel(r'$G_{FI}(p)$')
        ax.set_title(f'{channel_titles[channel_name]}: Logit-Shift Index')
        ax.legend(loc='best')
        ax.set_xlim([0, 1])
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    png_path = os.path.join(output_dir, 'Figure_Empirical_GFI_AllChannels.png')
    pdf_path = os.path.join(output_dir, 'Figure_Empirical_GFI_AllChannels.pdf')
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)

    print(f"  Saved: {png_path}")
    return png_path


# =============================================================================
# SUMMARY TABLE GENERATION
# =============================================================================
def generate_summary_table(returns, fitted_params, moments, output_dir):
    """
    Generate summary table of calibration results.
    """
    print("\nGenerating Summary Table...")

    alpha, theta, beta, gamma, mu = fitted_params

    # Sample statistics
    sample_stats = {
        'N': len(returns),
        'Mean': returns.mean(),
        'Std Dev': returns.std(),
        'Skewness': pd.Series(returns).skew(),
        'Kurtosis': pd.Series(returns).kurtosis(),
        'Min': returns.min(),
        'Max': returns.max()
    }

    # Fitted parameters
    fitted_dict = {
        'alpha': alpha,
        'theta': theta,
        'beta': beta,
        'gamma': gamma,
        'mu': mu
    }

    # Create DataFrame for export
    summary_df = pd.DataFrame({
        'Parameter': list(fitted_dict.keys()),
        'Value': list(fitted_dict.values())
    })

    stats_df = pd.DataFrame({
        'Statistic': list(sample_stats.keys()),
        'Value': list(sample_stats.values())
    })

    moments_df = pd.DataFrame({
        'Moment': list(moments.keys()),
        'Fitted Value': list(moments.values())
    })

    # Save to CSV
    summary_path = os.path.join(output_dir, 'Table_Fitted_Parameters.csv')
    summary_df.to_csv(summary_path, index=False)

    stats_path = os.path.join(output_dir, 'Table_Sample_Statistics.csv')
    stats_df.to_csv(stats_path, index=False)

    moments_path = os.path.join(output_dir, 'Table_Fitted_Moments.csv')
    moments_df.to_csv(moments_path, index=False)

    print(f"  Saved: {summary_path}")
    print(f"  Saved: {stats_path}")
    print(f"  Saved: {moments_path}")

    # Also create a combined LaTeX-ready table
    latex_table = f"""
% NTS Calibration Results - SPY Daily Returns (2015-2025)
% Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

\\begin{{table}}[htbp]
\\centering
\\caption{{NTS Parameter Estimates from SPY Daily Log-Returns (2015--2025)}}
\\label{{tab:empirical_nts_fit}}
\\begin{{tabular}}{{lcc}}
\\toprule
Parameter & Symbol & Estimate \\\\
\\midrule
Tail index & $\\alpha$ & {alpha:.4f} \\\\
Tempering & $\\theta$ & {theta:.4f} \\\\
Skewness & $\\beta$ & {beta:.6f} \\\\
Scale & $\\gamma$ & {gamma:.6f} \\\\
Location & $\\mu$ & {mu:.6f} \\\\
\\midrule
\\multicolumn{{3}}{{l}}{{\\textit{{Sample Statistics}}}} \\\\
Observations & $N$ & {len(returns):,} \\\\
Sample Mean & & {returns.mean():.6f} \\\\
Sample Std Dev & & {returns.std():.6f} \\\\
Sample Skewness & & {pd.Series(returns).skew():.4f} \\\\
Sample Kurtosis & & {pd.Series(returns).kurtosis():.4f} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""

    latex_path = os.path.join(output_dir, 'Table_NTS_Calibration.tex')
    with open(latex_path, 'w') as f:
        f.write(latex_table)

    print(f"  Saved: {latex_path}")

    return summary_df, stats_df


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    """Main execution function."""
    print("="*70)
    print("NTS-PWF EMPIRICAL ANALYSIS: SPY RETURNS")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    # Step 1: Load data
    df, returns = load_spy_data()

    # Step 2: Fit NTS to returns
    fitted_params, moments = fit_nts_to_returns(returns, maxeval=500)

    # Step 3: Create channel perturbations
    channels = create_channel_perturbations(fitted_params)

    # Step 4: Compute PWFs and indices
    results, p_grid = compute_pwf_and_indices(channels)

    # Step 5: Generate figures
    print("\n" + "="*70)
    print("GENERATING FIGURES")
    print("="*70)

    plot_fitted_distribution(returns, fitted_params, OUTPUT_DIR)
    plot_pwf_channels(results, p_grid, OUTPUT_DIR)
    plot_pwf_deviations(results, p_grid, OUTPUT_DIR)
    plot_indices(results, p_grid, OUTPUT_DIR)
    plot_all_channels_indices(results, p_grid, OUTPUT_DIR)

    # Step 6: Generate summary tables
    print("\n" + "="*70)
    print("GENERATING TABLES")
    print("="*70)

    generate_summary_table(returns, fitted_params, moments, OUTPUT_DIR)

    # Summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nGenerated outputs in: {OUTPUT_DIR}")
    print("\nFigures:")
    print("  - Figure_Empirical_NTS_Fit.png/pdf")
    print("  - Figure_Empirical_PWF_Channels.png/pdf")
    print("  - Figure_Empirical_PWF_Deviations.png/pdf")
    print("  - Figure_Empirical_Indices.png/pdf")
    print("  - Figure_Empirical_GFI_AllChannels.png/pdf")
    print("\nTables:")
    print("  - Table_Fitted_Parameters.csv")
    print("  - Table_Sample_Statistics.csv")
    print("  - Table_Fitted_Moments.csv")
    print("  - Table_NTS_Calibration.tex")
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)


if __name__ == "__main__":
    main()
