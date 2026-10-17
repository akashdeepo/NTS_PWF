"""
Shared model code for the PWF-NTS paper.

The normal tempered-stable (NTS) distribution is used in the parameterization
of Kim, Rachev, Bianchi and Fabozzi (2010), which is the one implemented by
the vendored temStaPy library:

    X = mu + beta (T - 1) + gamma sqrt(T) Z,     Z ~ N(0, 1) independent of T,

where T is a classical tempered-stable subordinator with index alpha/2 in
(0, 1), tempering parameter theta > 0 and E[T] = 1, so that

    Var(T) = (2 - alpha) / (2 theta).

The parameter vector is [alpha, theta, beta, gamma, mu] throughout:
alpha in (0, 2) is the tail index, theta > 0 the tempering (lower = heavier
tails), beta the asymmetry, gamma > 0 the diffusive scale and mu the location.

Functions
---------
nts_moments           mean, variance, skewness, excess kurtosis (closed form)
beta_for_skewness     beta (and the variance-preserving gamma) that hits a
                      target skewness
channel_specs         the three perturbation channels around any benchmark
compute_pwf           w(p) = F_post(Q_prior(p))
pwf_derivative        w'(p) = f_post(Q_prior(p)) / f_prior(Q_prior(p))
logit_shift_GFI, signed_jensen_shannon, log_odds_elasticity
                      the three information-theoretic indices
aggregate_indices     integrated (scalar) versions of the indices
"""

import os
import sys

import numpy as np
from scipy.optimize import brentq

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, 'lib', 'temStaPy_v0.5'))

from temStaPy.distNTS import dnts, pnts, qnts  # noqa: E402

EPS = 1e-6          # probability clipping for the logit-based indices

# Stylized benchmark of Section 4: unit variance, symmetric, excess kurtosis 3.
STYLIZED_BENCHMARK = [1.2, 0.4, 0.0, 1.0, 0.0]


# ---------------------------------------------------------------------------
# Moments and channel construction
# ---------------------------------------------------------------------------
def nts_moments(params):
    """Closed-form mean, variance, skewness and excess kurtosis of NTS(params).

    With v = Var(T) = (2 - alpha)/(2 theta), the cumulants of T are
    k2 = v, k3 = v (2 - alpha/2)/theta, k4 = v (2 - alpha/2)(3 - alpha/2)/theta^2,
    and X = mu + beta (T - 1) + gamma sqrt(T) Z gives
    k2(X) = gamma^2 + beta^2 v,
    k3(X) = beta^3 k3 + 3 beta gamma^2 v,
    k4(X) = beta^4 k4 + 6 beta^2 gamma^2 k3 + 3 gamma^4 v.
    """
    alpha, theta, beta, gamma, mu = params
    v = (2.0 - alpha) / (2.0 * theta)
    a = 1.0 - alpha / 2.0                 # = v * theta
    k3T = v * (2.0 - alpha / 2.0) / theta
    k4T = v * (2.0 - alpha / 2.0) * (3.0 - alpha / 2.0) / theta ** 2
    var = gamma ** 2 + beta ** 2 * v
    k3 = beta ** 3 * k3T + 3.0 * beta * gamma ** 2 * v
    k4 = beta ** 4 * k4T + 6.0 * beta ** 2 * gamma ** 2 * k3T + 3.0 * gamma ** 4 * v
    return {
        'mean': mu,
        'variance': var,
        'skewness': k3 / var ** 1.5,
        'excess kurtosis': k4 / var ** 2,
    }


def nts_variance(params):
    return nts_moments(params)['variance']


def gamma_for_variance(alpha, theta, beta, target_var):
    """gamma such that NTS(alpha, theta, beta, gamma) has the target variance."""
    v = (2.0 - alpha) / (2.0 * theta)
    rest = target_var - beta ** 2 * v
    if rest <= 0:
        raise ValueError('beta too large for the target variance')
    return float(np.sqrt(rest))


def beta_for_skewness(alpha, theta, target_skew, target_var, mu=0.0):
    """Parameters with the given skewness at the given variance.

    beta is solved by root finding; gamma is then set so that the variance is
    unchanged. Returns the full parameter vector.
    """
    v = (2.0 - alpha) / (2.0 * theta)
    bmax = 0.999 * np.sqrt(target_var / v)

    def f(b):
        g = gamma_for_variance(alpha, theta, b, target_var)
        return nts_moments([alpha, theta, b, g, mu])['skewness'] - target_skew

    b = brentq(f, -bmax, bmax)
    return [alpha, theta, b, gamma_for_variance(alpha, theta, b, target_var), mu]


def channel_specs(benchmark, scale_factors=(1.4, 0.7), skew_shift=0.5, tail_shift=0.3):
    """The three perturbation channels around a benchmark NTS.

    scale : gamma multiplied by 1.4 (fearful) or 0.7 (greedy); other parameters fixed
    skew  : skewness lowered (fearful) or raised (greedy) by `skew_shift`, with
            gamma adjusted so the variance is unchanged
    tail  : alpha lowered (fearful, heavier tails) or raised (greedy) by
            `tail_shift`; variance is unchanged automatically when beta = 0 and
            is restored through gamma otherwise

    Returns {channel: {'benchmark': params, 'fearful': params, 'greedy': params}}.
    """
    alpha, theta, beta, gamma, mu = benchmark
    m = nts_moments(benchmark)
    var0, skew0 = m['variance'], m['skewness']
    hi, lo = scale_factors
    specs = {
        'scale': {
            'benchmark': list(benchmark),
            'fearful': [alpha, theta, beta, gamma * hi, mu],
            'greedy': [alpha, theta, beta, gamma * lo, mu],
        },
        'skew': {
            'benchmark': list(benchmark),
            'fearful': beta_for_skewness(alpha, theta, skew0 - skew_shift, var0, mu),
            'greedy': beta_for_skewness(alpha, theta, skew0 + skew_shift, var0, mu),
        },
        'tail': {
            'benchmark': list(benchmark),
            'fearful': [alpha - tail_shift, theta, beta,
                        gamma_for_variance(alpha - tail_shift, theta, beta, var0), mu],
            'greedy': [alpha + tail_shift, theta, beta,
                       gamma_for_variance(alpha + tail_shift, theta, beta, var0), mu],
        },
    }
    return specs


# ---------------------------------------------------------------------------
# PWF
# ---------------------------------------------------------------------------
def probability_grid(n=2000):
    """n interior points p_i = i/(n+1), avoiding 0 and 1."""
    return np.arange(1, n + 1) / (n + 1.0)


def compute_pwf(p, prior, post):
    """w(p) = F_post(Q_prior(p))."""
    q = qnts(np.asarray(p, dtype=float), prior)
    return pnts(q, post)


def pwf_derivative(p, prior, post):
    """w'(p) = f_post(Q_prior(p)) / f_prior(Q_prior(p)), evaluated exactly."""
    q = qnts(np.asarray(p, dtype=float), prior)
    return dnts(q, post) / dnts(q, prior)


def safe_clip(arr, epsilon=EPS):
    return np.clip(arr, epsilon, 1.0 - epsilon)


# ---------------------------------------------------------------------------
# Indices
# ---------------------------------------------------------------------------
def classical_index(w, p):
    """gamma(p) = (w(p) - p) / (p (1 - p))."""
    return (w - p) / (p * (1.0 - p))


def logit_shift_GFI(w, p, epsilon=EPS):
    """G_FI(p) = [logit w(p) - logit p] / sqrt(p (1 - p))."""
    w = safe_clip(w, epsilon)
    p = safe_clip(p, epsilon)
    G = np.log(w / (1 - w)) - np.log(p / (1 - p))
    return G / np.sqrt(p * (1 - p))


def _kl_bernoulli(a, b):
    return a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))


def jensen_shannon_divergence(w, p, epsilon=EPS):
    """JSD between Bernoulli(w) and Bernoulli(p), natural logarithms."""
    w = safe_clip(w, epsilon)
    p = safe_clip(p, epsilon)
    m = 0.5 * (w + p)
    return 0.5 * (_kl_bernoulli(w, m) + _kl_bernoulli(p, m))


def signed_jensen_shannon(w, p, epsilon=EPS):
    """S_JS(p) = sgn(w(p) - p) sqrt(2 JSD(p)), bounded by sqrt(2 ln 2)."""
    return np.sign(w - p) * np.sqrt(2.0 * jensen_shannon_divergence(w, p, epsilon))


def log_odds_elasticity(w, p, wprime, epsilon=EPS):
    """E(p) = d/dp [logit w(p) - logit p] = w'(p)/(w(1-w)) - 1/(p(1-p))."""
    w = safe_clip(w, epsilon)
    p = safe_clip(p, epsilon)
    return wprime / (w * (1 - w)) - 1.0 / (p * (1 - p))


def all_indices(p, prior, post):
    """PWF, its derivative and the three indices on the grid p."""
    w = compute_pwf(p, prior, post)
    wp = pwf_derivative(p, prior, post)
    return {
        'w': w,
        'wprime': wp,
        'gamma': classical_index(w, p),
        'G_FI': logit_shift_GFI(w, p),
        'S_JS': signed_jensen_shannon(w, p),
        'E': log_odds_elasticity(w, p, wp),
    }


def aggregate_indices(p, idx):
    """Integrals over p of G_FI, S_JS and |E|, plus |G_FI| and |S_JS|."""
    return {
        'int_GFI': float(np.trapezoid(idx['G_FI'], p)),
        'int_abs_GFI': float(np.trapezoid(np.abs(idx['G_FI']), p)),
        'int_SJS': float(np.trapezoid(idx['S_JS'], p)),
        'int_abs_SJS': float(np.trapezoid(np.abs(idx['S_JS']), p)),
        'int_abs_E': float(np.trapezoid(np.abs(idx['E']), p)),
    }
