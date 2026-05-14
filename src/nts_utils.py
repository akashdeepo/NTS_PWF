"""
NTS Utility Functions
=====================

Common utilities for NTS distribution analysis and PWF computation.

This module provides:
- CGMY to NTS parameter conversion
- PWF computation
- Information-theoretic indices
- Variance calculations
"""

import numpy as np
import sys
import os

# Path setup: works from any cwd (file at NTS_PWF/src/nts_utils.py)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, 'lib', 'temStaPy_v0.5'))

from temStaPy.distNTS import dnts, pnts, qnts


def cgmy_to_nts(C, G, M, Y, mu=0, sigma=1):
    """
    Convert CGMY parameters to NTS parameters for temStaPy.

    Parameters:
    -----------
    C : float
        Activity parameter
    G : float
        Right tempering parameter
    M : float
        Left tempering parameter
    Y : float
        Tail index (0 < Y < 2)
    mu : float
        Location parameter (default: 0)
    sigma : float
        Scale parameter (default: 1)

    Returns:
    --------
    list : [alpha, theta, beta, gamma, mu]
        NTS parameters for temStaPy
    """
    alpha = Y
    theta = C * (G**Y + M**Y) / Y
    beta_raw = (M**Y - G**Y) / (M**Y + G**Y)
    gamma_param = sigma
    beta = beta_raw * sigma
    return [alpha, theta, beta, gamma_param, mu]


def compute_variance_nts(C, G, M, Y, sigma):
    """
    Compute theoretical variance of NTS distribution.

    Formula: Var(X) = σ² + C(G^(Y-2) + M^(Y-2))/(2-Y)

    Parameters:
    -----------
    C, G, M, Y : float
        CGMY parameters
    sigma : float
        Scale parameter

    Returns:
    --------
    float : Theoretical variance
    """
    if Y >= 2:
        raise ValueError("Variance undefined for Y >= 2")

    jump_var = C * (G**(Y-2) + M**(Y-2)) / (2 - Y)
    total_var = sigma**2 + jump_var
    return total_var


def compute_pwf(u_grid, prior_params, post_params):
    """
    Compute Probability Weighting Function.

    Formula: w(u) = F_post(Q_prior(u))

    Parameters:
    -----------
    u_grid : array
        Probability grid (0, 1)
    prior_params : list
        NTS parameters for prior distribution
    post_params : list
        NTS parameters for posterior distribution

    Returns:
    --------
    array : PWF values w(u)
    """
    q_prior = qnts(u_grid, prior_params)
    w = pnts(q_prior, post_params)
    return w


def safe_clip(arr, epsilon=1e-6):
    """Clip array to [epsilon, 1-epsilon] to avoid log(0)"""
    return np.clip(arr, epsilon, 1-epsilon)


def logit_shift_GFI(w, p, epsilon=1e-6):
    """
    Logit-shift greed-fear index with Fisher-information normalization.

    Formula: G_FI(p) = [logit(w(p)) - logit(p)] / sqrt[p(1-p)]
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
    Signed Jensen-Shannon index.

    Formula: S_JS(p) = sgn(w(p) - p) * sqrt(2 * JSD(p))
    where JSD(p) = 0.5 * [KL(w||m) + KL(p||m)], m = (w+p)/2
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
    Log-odds elasticity (derivative index).

    Formula: E(p) = d/dp[logit(w(p)) - logit(p)]
           = w'(p) / [w(p)(1-w(p))] - 1 / [p(1-p)]
    """
    w_clip = safe_clip(w, epsilon)
    p_clip = safe_clip(p, epsilon)

    # Compute derivative using numerical gradient
    dw_dp = np.gradient(w_clip, p_clip, edge_order=2)

    # Elasticity formula
    term1 = dw_dp / (w_clip * (1 - w_clip))
    term2 = 1.0 / (p_clip * (1 - p_clip))

    E = term1 - term2

    return E


def jensen_shannon_divergence(w, p, epsilon=1e-6):
    """
    Jensen-Shannon divergence between w and p.

    Formula: JSD(p) = 0.5 * [KL(w||m) + KL(p||m)]
    where m = (w+p)/2
    """
    w_clip = safe_clip(w, epsilon)
    p_clip = safe_clip(p, epsilon)

    m = (w_clip + p_clip) / 2.0

    kl_wm = kl_divergence_bernoulli(w_clip, m, epsilon)
    kl_pm = kl_divergence_bernoulli(p_clip, m, epsilon)

    JSD = 0.5 * (kl_wm + kl_pm)

    return JSD
