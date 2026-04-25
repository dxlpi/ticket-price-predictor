# XGBoost Segfault from OpenBLAS/OpenMP Threading Conflict in pytest
Promoted: 2026-03-27 | Updated: 2026-03-27
## Rule
When running the full pytest suite, set `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` to prevent XGBoost from segfaulting. Without these, XGBoost segfaults in `test_fit_and_predict` when any test that imports LightGBM or sklearn (which initializes OpenBLAS threading) runs first in the same process.
## Why
XGBoost uses OpenMP for parallelism. When OpenBLAS (used by LightGBM/sklearn) initializes its thread pool first, it leaves global thread state that conflicts with XGBoost's OpenMP initialization, causing a segfault at `xgboost/core.py DMatrix._init`. The crash only happens when both libraries are loaded in the same process — isolated XGBoost tests pass fine.
## Pattern
```makefile
# Wrong — segfaults when LightGBM tests run before XGBoost tests
test:
	pytest

# Right — single-threaded BLAS/OMP prevents the conflict
test:
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 pytest
```
The fix is in `Makefile`'s `test` target. This is a known macOS + Anaconda environment issue.
