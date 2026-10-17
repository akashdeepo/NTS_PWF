# Probability-Weighting Functions for Normal Tempered-Stable Laws

Code and data that reproduce every figure and table in

> Akash Deep, Bhathiya Divelgama, A. Alexandre Trindade, W. Brent Lindquist, Svetlozar T. Rachev, Frank J. Fabozzi.
> **Probability-Weighting Functions for Normal Tempered-Stable Laws: Orthogonal Channels and Information-Theoretic Indices.** Working paper, 2026.

All 14 figures and both tables regenerate from a clean checkout in under a minute.

## The idea in brief

A probability-weighting function (PWF) describes how an agent turns objective probabilities into decision weights. Instead of assuming a parametric shape (Prelec, Tversky–Kahneman), the paper derives the PWF from a change in beliefs about returns. A *prior* return distribution is mapped to a *posterior* one:

$$w(p) = F_{\text{post}}\big(Q_{\text{prior}}(p)\big), \qquad p \in [0,1],$$

where $Q_{\text{prior}}$ is the prior quantile function and $F_{\text{post}}$ is the posterior CDF. If the two distributions are equal, then $w(p) = p$. Any departure from the diagonal measures how much more (or less) weight the posterior places on that region of outcomes.

Both distributions are **normal tempered-stable (NTS)**, which captures the semi-heavy tails and skewness of real returns while keeping a closed-form characteristic function. The posterior differs from the prior along one of three **orthogonal channels**:

| Channel | Parameter changed | Interpretation |
|---|---|---|
| Scale (volatility) | $\sigma$ | more or less dispersion |
| Skew (asymmetry) | tempering ratio $G/M$ | heavier left or right tail |
| Tail thickness | Blumenthal–Getoor index $Y$ | more or less jump activity |

Each PWF is summarised by three indices:

- **Fisher-normalised logit shift:** $G_{\text{FI}}(p) = \dfrac{\operatorname{logit} w(p) - \operatorname{logit} p}{\sqrt{p(1-p)}}$
- **Signed Jensen–Shannon divergence:** $S_{\text{JS}}(p) = \operatorname{sgn}\big(w(p)-p\big)\sqrt{2\,\mathrm{JSD}(p)}$
- **Log-odds elasticity:** $E(p) = \dfrac{d}{dp}\big[\operatorname{logit} w(p) - \operatorname{logit} p\big]$

The empirical section fits NTS to SPY daily log-returns (2015–2025) and repeats the channel analysis around the fitted parameters.

## Quick start

Requires Python 3.10 or later.

```bash
git clone https://github.com/akashdeepo/NTS_PWF.git
cd NTS_PWF
pip install -r requirements.txt

# Regenerate everything (about 40 s in total)
for f in src/generate_fig*.py; do python "$f"; done
```

Each script can also be run on its own and from any working directory. Output goes to `outputs/`, which git ignores.

## Paper figures and tables → scripts

| Paper | Script (`src/`) | Output file (`outputs/`) |
|---|---|---|
| Fig. 1 | `generate_fig01_overview.py` | `figure_overview/Figure_2a_Overview_PDFs.pdf` |
| Fig. 2 | `generate_fig02to05_case1.py` | `case1_overview/Figure_2b_NTS_CDFs.pdf` |
| Fig. 3 | 〃 | `case1_overview/Figure_2c_NTS_Quantiles.pdf` |
| Fig. 4 | 〃 | `case1_overview/Figure_2d_Case1_PWF.pdf` |
| Fig. 5 | 〃 | `case1_overview/Figure_2e_Case1_GreedFearIndex.pdf` |
| Fig. 6 | `generate_fig06_logitshift.py` | `case1_indices/Figure_2e1_LogitShift_GFI.pdf` |
| Fig. 7 | `generate_fig07_case2_deviation.py` | `case2_deviation/Figure_2i_CLEAN_Deviation.pdf` |
| Fig. 8 | `generate_fig08to09_case3.py` | `case3_tail/Figure_2l_PWFs_Case3.pdf` |
| Fig. 9 | 〃 | `case3_tail/Figure_2n_InfoMetrics_Case3.pdf` |
| Fig. 10 | `generate_fig10to14_empirical.py` | `Figure_Empirical_NTS_Fit.pdf` |
| Fig. 11 | 〃 | `Figure_Empirical_PWF_Channels.pdf` |
| Fig. 12 | 〃 | `Figure_Empirical_PWF_Deviations.pdf` |
| Fig. 13 | 〃 | `Figure_Empirical_GFI_AllChannels.pdf` |
| Fig. 14 | 〃 | `Figure_Empirical_Indices.pdf` |
| Tables 1–2 | 〃 | `Table_Sample_Statistics.csv`, `Table_Fitted_Parameters.csv`, `Table_NTS_Calibration.tex` |

The scripts also write some supplementary panels that the paper doesn't use, e.g. the cumulative index plots in `case1_indices/`. Each script's docstring lists its parameters and outputs.

## Repository layout

```
src/
  nts_utils.py                 shared maths: CGMY→NTS conversion, PWF construction, the three indices
  generate_fig*.py             one script per group of paper figures (table above)
lib/temStaPy_v0.5/             vendored NTS library (density, CDF, quantile, fitting)
data/
  spy_prices.csv               SPY adjusted close and daily log-return, 2015-01-02 to 2025-12-03
  rates_3m.csv                 3-month rate (not used by the paper's figures)
  spy_dividend_yield.csv       dividend yield (not used by the paper's figures)
outputs/                       generated figures and tables (git-ignored)
```

### Parameterisation

The paper states the channels in CGMY form $(C, G, M, Y, \sigma)$. `temStaPy` uses the NTS vector $[\alpha, \theta, \beta, \gamma, \mu]$. `nts_utils.cgmy_to_nts()` converts between the two, and every PWF in the paper is built by `nts_utils.compute_pwf()`. To change a formula, change it in `nts_utils.py` rather than in an individual script.

### Empirical calibration

`generate_fig10to14_empirical.py` fits NTS to $N = 2{,}746$ SPY daily log-returns using `temStaPy.fitnts`, in two steps:

1. Location $\mu$ and scale $\gamma$ are set from the sample mean and standard deviation.
2. The shape parameters $(\alpha, \theta, \beta)$ are chosen to minimise the mean squared distance between the standardised NTS CDF and a kernel-smoothed empirical CDF.

The sample has skewness −0.58 and excess kurtosis 14.5. The fitted parameters are $\alpha = 1.07$, $\theta = 0.204$, $\beta = -0.00088$, $\gamma = 0.0112$ and $\mu = 0.00050$. The optimiser's result can differ in the fourth significant figure across numpy/scipy versions.

## Dependencies

`numpy`, `scipy`, `matplotlib`, `pandas` and `statsmodels` (see `requirements.txt`).

`temStaPy` is included under `lib/` so the results don't depend on a particular release of that library. The upstream project is [temStaPy](https://github.com/jay-gne/temStaPy) (also on PyPI), and its own licence applies to `lib/temStaPy_v0.5/`.

## Citation

```bibtex
@unpublished{deep2026pwf,
  title  = {Probability-Weighting Functions for Normal Tempered-Stable Laws:
            Orthogonal Channels and Information-Theoretic Indices},
  author = {Deep, Akash and Divelgama, Bhathiya and Trindade, A. Alexandre and
            Lindquist, W. Brent and Rachev, Svetlozar T. and Fabozzi, Frank J.},
  note   = {Working paper},
  year   = {2026}
}
```

## Licence

The code in `src/` is released under the MIT licence (see [LICENSE](LICENSE)).

## Contact

Akash Deep, Department of Mathematics and Statistics, Texas Tech University: akash.deep@ttu.edu
