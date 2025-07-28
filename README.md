# Churn‑Uncertainty Toolkit

Code used for Master's thesis "Identifying Risky Customers with Venn–Abers Calibration for Churn Data". This repository contains three **self‑contained Python utilities** that let you:

1. train / calibrate an XGBoost‑based churn model with the proposed **Multi‑Split BA‑IVAP** pipeline (`pipeline.py`);
2. analyse the resulting prediction‑interval widths** in depth (`width_analysis.py`);
3. quantify calibration quality & pair‑wise statistical relationships** among the different width definitions (`correlation_analysis.py`).

Together they show **how uncertainty estimates can be used to find “knife‑edge” customers whose churn risk is inherently ambiguous**—and therefore worth extra attention.



---

## Repository contents
`pipeline.py` - End‑to‑end pipeline: data load → feature engineering → model training → calibration and widths calculations → CSV of predictions.  *Runs **GPU XGBoost** – CUDA ≥ 11.2 required.* 
`width_analysis.py`- Reads the test‑set prediction CSV and produces:<br>  • descriptive stats ( `width_summary.csv` )<br>  • histograms/KDEs, lift curves, scatter & correlation plots<br>  • churn‑distribution table 
`correlation_analysis.py`- Lightweight script focusing on:<br>  • decile‑trend check (does width ↑ → churn ↑?)<br>  • pair‑wise Pearson/Spearman correlations (single CSV)<br>  • small 2×2 grids (distribution & trend). 
`VennABERS.py` - taken directly from https://github.com/ptocca/VennABERS. Paolo Toccaceli’s reference implementation of the Inductive Venn–Abers Predictor (IVAP) described by Vovk et al. (2015).
---

## Requirements

| Package | Version tested |
|---------|----------------|
| Python  | 3.10 |
| numpy, pandas, scipy, seaborn, matplotlib | current PyPI |
| XGBoost  | **>= 1.7 GPU build** |
| RAPIDS cuDF / cuML *(optional)* | 25.04 |
| CUDA Toolkit | **11.2+** (for `pipeline.py`) |




