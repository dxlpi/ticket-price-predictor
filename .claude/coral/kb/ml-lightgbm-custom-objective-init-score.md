# LightGBM Custom Objective init_score Requirement
Promoted: 2026-04-03

## Rule
When using a custom callable objective (`fobj`) in LightGBM 4.6+, predictions start at 0 (not at the mean of the training target). For log-space targets (all positive), this means ALL gradients have the same sign → clipped to the same value → zero split gain → model never learns. Fix: set `init_score=mean(y_train)` on the LightGBM Dataset object, store the offset, and add it back in `predict()`.

## Why
Without this, custom objectives like asymmetric Huber produce "No further splits with positive gain" warnings and MAE degrades catastrophically (e.g., $253 vs $86 baseline). The built-in objectives (huber, regression) handle init_score automatically, but custom callables don't.

## Pattern
Wrong:
```python
train_data = lgb.Dataset(X_train, label=y_train)
model = lgb.train(params, train_data, fobj=custom_loss)
```

Right:
```python
offset = float(np.mean(y_train))
train_data = lgb.Dataset(X_train, label=y_train, init_score=np.full(len(y_train), offset))
val_data = lgb.Dataset(X_val, label=y_val, init_score=np.full(len(y_val), offset))
model = lgb.train(params, train_data, valid_sets=[val_data], fobj=custom_loss)
# In predict: return model.predict(X) + offset
```
