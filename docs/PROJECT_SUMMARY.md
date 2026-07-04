# Project Summary — Ticket Price Predictor

> A portfolio- and resume-oriented summary of the project: what it does, how it's
> built, the engineering decisions behind it, and quantified outcomes. For live model
> metrics see [`model-card.md`](model-card.md); for the obstacle log see [`issues/`](issues/).

---

## Elevator Pitch

An end-to-end machine learning system that predicts secondary-market (resale) ticket
prices at the seat-zone level for live events. It ingests event metadata, scrapes real
resale listings, enriches them with artist-popularity and regional-demand signals, and
serves calibrated price predictions (with 95% confidence intervals) through a stacking
ensemble of gradient-boosted models behind a FastAPI web layer.

---

## Problem & Motivation

Resale ticket prices are opaque and volatile — the same seat can swing hundreds of
dollars depending on artist demand, days-to-event, venue, region, and inventory
depletion. Buyers can't tell whether a listing is fairly priced, and sellers can't tell
whether their ticket will sell. This project builds a data-driven fair-price estimate
and a companion sell-through probability model to make that market legible.

---

## What It Does

- **Price prediction** — estimates the fair resale price for a given artist / venue /
  seat zone, with a 95% confidence interval and a price-direction signal.
- **Value ranking** — `ListingRanker` scores live listings by *predicted fair price ÷
  actual price*; listings above 1.0 are surfaced as "best value."
- **Sale probability (CVR)** — a companion classifier estimates the probability a
  listing sells within a fixed window, enabling commerce-style ranking.
- **Web serving** — a FastAPI app exposes interactive lookups over the trained corpus.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language / tooling | Python 3.12, `uv` package manager |
| ML | LightGBM, scikit-learn, Optuna (hyperparameter tuning); CatBoost / XGBoost / FT-Transformer explored |
| Data | pandas, PyArrow / Parquet, Pydantic (schema validation) |
| Ingestion | Ticketmaster Discovery API, Setlist.fm, YouTube Music (`ytmusicapi`), Last.fm |
| Scraping | Playwright (VividSeats / StubHub) with anti-detection patterns |
| Serving | FastAPI |
| Quality | ruff (lint/format), mypy (strict), pytest (+ pytest-asyncio) — 860+ tests |
| Infra | AWS EC2 (t3.micro) hourly collection via systemd timer |

---

## Architecture Highlights

- **Layered, dependency-directed design** — external APIs → clients → schemas →
  scrapers/ingestion → storage → normalization/preprocessing → ML. A hard invariant
  keeps the `ml/` layer out of lower layers, enforced by documented dependency rules.
- **Repository pattern** over Parquet storage (event / listing / snapshot repositories)
  handling partitioning, deduplication, and schema enforcement.
- **Dual-schema data models** — every entity has both a Pydantic model (runtime
  validation) and a PyArrow `parquet_schema()` (storage typing), kept in sync.
- **Feature pipeline** — 11 feature domains, 80+ engineered features, each extractor
  implementing a common `fit()` / `extract()` interface and registered in an
  orchestrating pipeline.

---

## ML Engineering Rigor

The decisions that make this more than a notebook model:

- **Split-before-fit (leakage prevention)** — raw data is split *temporally with artist
  stratification* **before** any feature extraction. The feature pipeline is fitted on
  the training split only and applied to val/test independently. A dedicated
  leakage-guardian review gate protects this invariant.
- **Bayesian smoothing everywhere** — all group-level statistics (artist, region,
  venue, event/zone/section target encodings) are Bayesian-smoothed with per-domain
  factors, preventing small-sample memorization.
- **Fallback chains** — when group-level data is sparse, predictions fall through a
  hierarchy (event_zone → event → artist_zone → artist → global) for robustness on
  cold-start events.
- **Stacking ensemble** — `StackingEnsembleV2` combines a Huber-loss LightGBM, a deeper
  LightGBM, and a residual model into a Ridge meta-learner, trained with
  temporally-sorted out-of-fold folds to avoid meta-level leakage.
- **Honest metric scoping** — the headline metric (`primary_mae`) is measured on the
  in-scope (seen-event) slice the system actually serves, with out-of-scope numbers
  retained separately as diagnostics rather than buried.
- **Target/feature transforms** — `np.log1p` target transform for skewed prices;
  price-based features log-transformed to align; zero-variance features removed.

---

## Quantified Outcomes

**Price model** (v38, production — full benchmark in [`model-card.md`](model-card.md)):

| Metric | Value |
|--------|-------|
| primary MAE (seen events, served scope) | **$28.88** |
| overall MAE (incl. out-of-scope diagnostics) | $80.92 |
| R² | 0.63 |
| MAPE | 41.6% |
| RMSE | $148.85 |

**Progression** — mean absolute error was driven down **~63% overall** (and far more on
the served slice) across iterative modeling work:

```
v18  $216.88   →   v21  $141.95   →   v34  $84.76   →   v36  $83.63   →   v38  $80.92 overall / $28.88 primary
```

**Sale-probability model (CVR v3):** AUC-ROC 0.7653, ECE 0.1525.

**Data scale (v37/v38 corpus):** ~347K listings, ~3,868 events, ~1,807 artists.

**Codebase:** 860+ automated tests across 49 modules; strict mypy + ruff gating.

---

## Challenges Overcome

Documented in [`docs/issues/`](issues/) — a sample of non-obvious problems solved:

- **Data leakage in the training pipeline** — reworked the flow to split raw data before
  fitting features, eliminating target-encoding leakage that had inflated metrics.
- **Unseen-event bottleneck** — quantified that held-out *unseen* events drove the bulk
  of error (seen $52.75 vs unseen $128.91 MAE at v36); reframed the headline metric to
  the served scope and traced the ceiling to data coverage, not model capacity, via an
  arithmetic-floor analysis.
- **Counter-intuitive experiments** — showed empirically that deduplication *hurt*
  (−$6.79), that segment-aware outlier capping *hurt* (−$6.07), and that listing-context
  features added noise — pruning them rather than assuming they'd help.
- **Degenerate CVR labels** — a disappearance-based labeling scheme produced 99% "sold"
  and near-random AUC (0.48); replaced it with an inventory-depletion labeler
  (~20% positive rate) that lifted AUC to 0.77, plus fixing early-stopping to monitor
  logloss rather than a degenerate validation AUC.
- **Scraper reliability** — Playwright scrapers with anti-detection patterns feeding an
  hourly EC2 collection loop, handling partial captures gracefully.

---

## Resume Bullets (ready to adapt)

- Built an end-to-end ML system predicting secondary-market ticket prices, reducing mean
  absolute error ~63% across iterative modeling (from $216 to $81 overall MAE; $28.88 on
  the served, in-scope slice) on a ~347K-listing / ~1,800-artist corpus.
- Designed a leak-free training pipeline (split-before-fit with artist-stratified
  temporal splits, Bayesian-smoothed target encoding, out-of-fold stacking) enforced by
  automated review gates — eliminating a data-leakage class of bug.
- Engineered 80+ features across 11 domains (target encoding, popularity, regional,
  temporal, structural) and a stacking ensemble of gradient-boosted models
  (LightGBM/CatBoost/XGBoost) with Optuna tuning, outputting calibrated 95% CIs.
- Developed a companion sale-probability (CVR) classifier — redesigned label
  construction (inventory depletion) to lift AUC-ROC from 0.48 to 0.77.
- Built the data platform end to end: Ticketmaster/Last.fm/YouTube Music ingestion,
  Playwright resale scrapers with anti-detection, Parquet storage via a repository
  pattern with dual Pydantic/PyArrow schemas, and a FastAPI serving layer.
- Maintained engineering rigor: 860+ tests, strict mypy, ruff, layered dependency rules,
  and a documented issue/ADR log; hourly automated collection on AWS EC2.

---

## Skills Demonstrated

**Machine learning:** gradient boosting, ensemble/stacking, quantile regression,
hyperparameter optimization, feature engineering, target encoding, calibration,
leakage prevention, honest evaluation design.

**Data engineering:** API integration, resilient web scraping, columnar (Parquet)
storage, schema design & evolution, data-quality validation, preprocessing pipelines.

**Software engineering:** layered architecture, dependency management, type-safe Python
(strict mypy), comprehensive testing, CI-style quality gates, FastAPI service design.

**Judgment:** rigorous experiment tracking, willingness to disprove intuitive ideas,
root-cause analysis, and metric definitions that reflect real product scope.
