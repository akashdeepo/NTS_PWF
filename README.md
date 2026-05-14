# NTS_PWF — Probability-Weighting Functions for Normal Tempered-Stable Laws

Reproducibility code for the paper

> **Probability-Weighting Functions for Normal Tempered-Stable Laws: Orthogonal Channels and Information-Theoretic Indices.**
> Akash Deep, Bhathiya Divelgama, A. Alexandre Trindade, W. Brent Lindquist, Svetlozar T. Rachev, Frank J. Fabozzi. *Computational Economics* (submitted, 2026).

The code (i) builds Normal Tempered-Stable (NTS) probability densities, CDFs, and quantile functions via the `temStaPy` library, (ii) constructs probability-weighting functions (PWFs) through quantile mapping under three orthogonal NTS parameter channels (scale/volatility, skew/asymmetry, tail-thickness), (iii) computes Fisher-normalized logit-shift, signed Jensen–Shannon, and log-odds elasticity indices, and (iv) calibrates the NTS distribution to daily SPY log-returns over 2015–2025 to demonstrate the framework empirically.

---

## Repository layout

```
NTS_PWF/
├── src/                                 Canonical figure generators
│   ├── generate_fig01_overview.py       Fig 1: multi-curve density overview
│   ├── generate_fig02to05_case1.py      Figs 2–5: CDFs, quantiles, Case-1 PWF + γ(p)
│   ├── generate_fig06_logitshift.py     Fig 6: Fisher-normalized logit-shift G_FI(p)
│   ├── generate_fig07_case2_deviation.py Fig 7: skew-channel PWF deviation δ(p)
│   ├── generate_fig08to09_case3.py      Figs 8–9: tail-channel PWF + info metrics
│   ├── generate_fig10to14_empirical.py  Figs 10–14: SPY empirical analysis
│   └── nts_utils.py                     Shared helpers (CGMY↔NTS conversion, indices)
├── lib/temStaPy_v0.5/                   NTS distribution library (vendored)
├── data/                                SPY daily log-returns + risk-free rate (2015–2025)
│   ├── spy_prices.csv
│   ├── rates_3m.csv
│   └── spy_dividend_yield.csv
└── outputs/                             Figures written here when scripts run (gitignored)
```

---

## Setup

Python 3.10+ recommended.

```bash
pip install -r requirements.txt
```

`temStaPy` is vendored under `lib/`; the scripts add it to `sys.path` automatically.

---

## Reproducing the paper's figures

Each script is self-contained and writes to a subdirectory of `outputs/`.

```bash
# From the repo root (paths below are repo-relative; scripts use absolute paths internally so any cwd works)
python src/generate_fig01_overview.py            # → outputs/figure_overview/Figure_2a_Overview_PDFs.pdf
python src/generate_fig02to05_case1.py           # → outputs/case1_overview/Figure_2{b,c,d,e,f,g,h,i}_*.pdf
python src/generate_fig06_logitshift.py          # → outputs/case1_indices/Figure_2e{1,2,3,4}_*.pdf
python src/generate_fig07_case2_deviation.py     # → outputs/case2_deviation/Figure_2i_CLEAN_Deviation.pdf
python src/generate_fig08to09_case3.py           # → outputs/case3_tail/Figure_2{j,k,l,m,n}_*.pdf
python src/generate_fig10to14_empirical.py       # → outputs/Figure_Empirical_*.{pdf,png} + outputs/Table_*.{csv,tex}
```

The empirical script also writes four summary tables alongside the figures: `Table_Fitted_Parameters.csv`, `Table_Sample_Statistics.csv`, `Table_Fitted_Moments.csv`, and `Table_NTS_Calibration.tex`.

Figure-name mapping to the paper (manuscript filename → generator output):

| Paper figure | Manuscript filename | Generator output |
|---|---|---|
| Fig 1 | `fig_2a.pdf` | `Figure_2a_Overview_PDFs.pdf` |
| Fig 2 | `Figure_2b_NTS_CDFs.pdf` | (same) |
| Fig 3 | `Figure_2c_NTS_Quantiles.pdf` | (same) |
| Fig 4 | `Figure_2d_Case1_PWF.pdf` | (same) |
| Fig 5 | `Figure_2e_Case1_GreedFearIndex.pdf` | (same) |
| Fig 6 | `figure2e1.pdf` | `Figure_2e1_LogitShift_GFI.pdf` |
| Fig 7 | `figure2i.pdf` | `Figure_2i_CLEAN_Deviation.pdf` |
| Figs 8–9 | `figure2l.pdf`, `figure2n.pdf` | `Figure_2l_PWFs_Case3.pdf`, `Figure_2n_InfoMetrics_Case3.pdf` |
| Figs 10–14 | `Figure_Empirical_*.pdf` | (same) |

---

## Data

`data/spy_prices.csv` contains adjusted closing prices for the SPDR S&P 500 ETF Trust (SPY) over January 2015 – December 2025 (N = 2,746 observations), obtained from Bloomberg. `rates_3m.csv` and `spy_dividend_yield.csv` are used for risk-neutral extensions.

The CSVs are redistributed here for reproducibility of the empirical results in the paper.

---

## Citation

```bibtex
@article{deep2026pwf,
  title   = {Probability-Weighting Functions for Normal Tempered-Stable Laws: Orthogonal Channels and Information-Theoretic Indices},
  author  = {Deep, Akash and Divelgama, Bhathiya and Trindade, A. Alexandre and Lindquist, W. Brent and Rachev, Svetlozar T. and Fabozzi, Frank J.},
  journal = {Computational Economics},
  year    = {2026},
  note    = {submitted}
}
```

---

## License

MIT (see [LICENSE](LICENSE)).

---

## Contact

Akash Deep — akash.deep@ttu.edu
Department of Mathematics and Statistics, Texas Tech University.
