# Code for "Exact Finite-Sample Inference for Lindley-Type Lifetime Distributions under Ranked Set Sampling"

Python 3, numpy, scipy, mpmath (>= 1.3), matplotlib.

- `pe.py`: polynomial-exponential families (GOLD, Akash); closed-form order statistics (Theorem 1);
  rational Laplace transform of the RSS/SRS total (Theorem 2); self-calibrating multiprecision Talbot inversion;
  exact confidence intervals and median-unbiased estimators (Theorem 3).
- `conv.py`: FFT-convolution law of the total (used for the design study; verified against `pe.py`).
- `gold.py`: GOLD RSS likelihood and Fisher-information helpers (RSS-MLE simulation only).
- `validate.py`: checks against direct formulas, closed-form SRS law, de Hoog inversion and Monte Carlo.
- `study_ab.py` (Tables 1-2, exact), `mle_mc.py` (RSS-MLE simulation), `study_c.py` (Table 3, Figure 1),
  `check_copula.py` (concomitant-ranking check), `app.py` (Tables 4-6), `design.py` (Table 7, Figure 2).
- `gen_numbers.py`, `make_tables.py`: produce `numbers.tex`, all tables and figures from `results/*.json`.

Run the study scripts from this folder, then `python3 gen_numbers.py && python3 make_tables.py`
(they write to `paper/`; adjust paths if needed).
