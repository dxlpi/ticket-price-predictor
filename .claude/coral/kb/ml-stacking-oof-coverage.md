# Stacking OOF Coverage: Track Which Samples Get Predictions
Promoted: 2026-04-03

## Rule
In expanding-window temporal CV for stacking ensembles, the first `fold_size` samples are NEVER in any validation fold — they only appear in training. Their OOF predictions stay at 0, poisoning the meta-learner. Always track which samples received OOF predictions via a boolean mask and train the meta-learner only on covered samples.

## Why
Without tracking, the meta-learner trains on rows where some base model predictions are 0 (the initial value) rather than actual predictions. This corrupts the learned weights and degrades ensemble performance.

## Pattern
Wrong:
```python
oof_preds = np.zeros((n_samples, n_models))
for train_idx, val_idx in folds:
    # train and predict...
    oof_preds[val_idx] = predictions
meta_model.fit(oof_preds, y_train)  # includes zero-filled rows!
```

Right:
```python
oof_preds = np.zeros((n_samples, n_models))
oof_covered = np.zeros(n_samples, dtype=bool)
for train_idx, val_idx in folds:
    # train and predict...
    oof_preds[val_idx] = predictions
    oof_covered[val_idx] = True
meta_model.fit(oof_preds[oof_covered], y_train[oof_covered])
```
