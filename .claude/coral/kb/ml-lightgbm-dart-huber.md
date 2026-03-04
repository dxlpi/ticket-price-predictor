# DART + Huber Loss Incompatibility in LightGBM

## Rule
Never pair Huber/MAE loss with DART boosting in LightGBM unless convergence without early stopping is guaranteed. DART does not support early stopping, and Huber's L1-like gradients create an unstable optimization landscape when combined with DART dropout, causing catastrophic overfitting.

## Why
LightGBM DART mode ignores `early_stopping_rounds`. With MSE loss, DART's dropout (drop_rate=0.1, skip_drop=0.5) provides implicit regularization sufficient to run all N trees without diverging. With Huber loss (alpha=0.5) in log-space, most errors exceed 0.5 so gradients are nearly L1 — weaker and flatter than L2. Combined with DART dropout, this produced train L1=0.07 vs valid L1=6.23, MAE regression from $150 (DART+MSE) to $382 (DART+Huber).

## Pattern
```python
# SAFE: DART + MSE (default v28 config)
params = {"boosting_type": "dart", "objective": "regression", "drop_rate": 0.1, "skip_drop": 0.5}

# SAFE: GBDT + Huber (early stopping works, early stops ~iter 56)
params = {"boosting_type": "gbdt", "objective": "huber", "alpha": 0.5, "early_stopping_rounds": 100}

# DANGEROUS: DART + Huber — catastrophic overfitting, MAE 2.5x worse
params = {"boosting_type": "dart", "objective": "huber", "alpha": 0.5}
```
For Huber to approach MSE behavior: set alpha > 5.0. GBDT+Huber ($170) is still worse than DART+MSE ($150) — DART's regularization genuinely helps this dataset.
