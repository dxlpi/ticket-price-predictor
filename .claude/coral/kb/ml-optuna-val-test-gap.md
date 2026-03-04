# Optuna Validation vs Test MAE Gap

## Rule
Do not trust single-split Optuna val MAE as the final performance estimate. Use cross-validation (`--cv --n-folds 5`) for robust evaluation. Best trial val MAE can be 15-20% optimistic vs test MAE.

## Why
Observed gap: Optuna best trial val MAE $127.30 vs full-pipeline test MAE $148.48. Three causes:
1. **Temporal distribution shift**: Val (middle segment) is closer to training data than test (latest). Optuna optimizes for val idiosyncrasies.
2. **Hyperparameter overfitting**: 50 trials on one val split overfits to that split.
3. **Dataset growth**: More events/artists = harder generalization problem. Maintaining similar MAE as dataset grows 2x is arguably better generalization, not regression.

## Pattern
```bash
# Risky: single-split tuning
python scripts/tune_model.py --n-trials 50

# Better: cross-validated tuning
python scripts/tune_model.py --n-trials 50 --cv --n-folds 5

# Always compare: Optuna best trial val MAE vs full retrain test MAE
# A gap of >$15 signals val overfitting — reduce n-trials or add CV
```
Also: feature importance distribution improvement (top feature 60% → 44%) is a structural signal worth tracking independently of MAE. Broader signal = better generalization.
