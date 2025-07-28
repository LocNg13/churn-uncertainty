# Churn‑Uncertainty Toolkit

Code used for Master's thesis "Identifying Risky Customers with Venn–Abers Calibration for Churn Data". This repository contains three **self‑contained Python utilities** that let you:

1. train / calibrate an XGBoost‑based churn model with the proposed **Multi‑Split BA‑IVAP** pipeline (`pipeline.py`);
2. analyse the resulting prediction‑interval widths** in depth (`width_analysis.py`);
3. quantify calibration quality & pair‑wise statistical relationships** among the different width definitions (`correlation_analysis.py`).

Together they show **how uncertainty estimates can be used to find customers whose churn risk is inherently ambiguous**—and therefore worth extra attention.



---
## Repository contents
| File | Purpose |
|------|---------|
| **`pipeline.py`** | End‑to‑end workflow: data load → feature engineering → model training → *multi‑split BA‑IVAP* calibration & width computation → CSV of predictions.<br/>*Requires GPU XGBoost – CUDA ≥ 11.2.* |
| **`width_analysis.py`** | Reads the test‑set prediction CSV and generates<br/>  • `width_summary.csv` (descriptive stats)<br/>  • histograms / KDEs, lift‑curves, scatter & correlation plots<br/>  • `churner_distribution_by_uncertainty.csv`. |
| **`correlation_analysis.py`** | Quick check of calibration quality:<br/>  • decile‑trend test (does width ↑ → churn ↑ ?)<br/>  • pairwise Pearson/Spearman correlations (`width_pairwise_correlation.csv`)<br/>  • two 2 × 2 grids (distribution & trend). |
| **`VennABERS.py`** | Script from <https://github.com/ptocca/VennABERS> – Paolo Toccaceli’s reference implementation of the Inductive Venn–Abers Predictor (IVAP, Vovk et al. 2015). |

---

## Requirements
| Package | Version tested |
|---------|----------------|
| Python  | 3.10 |
| numpy, pandas, scipy, seaborn, matplotlib | latest PyPI |
| XGBoost | **≥ 1.7 GPU build** |
| RAPIDS cuDF/cuML *(optional)* | 25.04 |
| CUDA Toolkit *(for `pipeline.py`)* | **11.2 or newer** |


