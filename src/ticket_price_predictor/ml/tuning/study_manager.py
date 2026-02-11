"""Optuna study management and persistence."""

import json
from pathlib import Path
from typing import Any

import optuna

from ticket_price_predictor.ml.training.splitter import DataSplit
from ticket_price_predictor.ml.tuning.objective import create_objective


class StudyManager:
    """Manages Optuna studies for hyperparameter tuning."""

    def __init__(
        self,
        study_name: str,
        storage_dir: Path = Path("data/optuna/studies"),
        trials_dir: Path = Path("data/optuna/trials"),
    ):
        self.study_name = study_name
        self.storage_dir = Path(storage_dir)
        self.trials_dir = Path(trials_dir)

        # Create directories
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.trials_dir.mkdir(parents=True, exist_ok=True)

        # SQLite storage URL
        self.storage = f"sqlite:///{self.storage_dir}/{study_name}.db"

    def create_study(
        self,
        direction: str = "minimize",
        pruner: optuna.pruners.BasePruner | None = None,
        sampler: optuna.samplers.BaseSampler | None = None,
        load_if_exists: bool = True,
    ) -> optuna.Study:
        """Create or load Optuna study."""

        if pruner is None:
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=10,
                n_warmup_steps=20,
                interval_steps=10,
            )

        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            direction=direction,
            pruner=pruner,
            sampler=sampler,
            load_if_exists=load_if_exists,
        )

        return study

    def optimize(
        self,
        split: DataSplit,
        n_trials: int = 50,
        timeout: int | None = None,
        search_space: str = "aggressive",
        penalize_dominance: bool = True,
        n_jobs: int = 1,
    ) -> optuna.Study:
        """Run optimization."""

        study = self.create_study()

        objective = create_objective(
            split=split,
            search_space=search_space,
            penalize_dominance=penalize_dominance,
        )

        print(f"Starting optimization: {self.study_name}")
        print(f"  Trials: {n_trials}")
        print(f"  Search space: {search_space}")
        print(f"  Penalize dominance: {penalize_dominance}")
        print(f"  Storage: {self.storage}")

        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            show_progress_bar=True,
        )

        print("\nOptimization complete!")
        print(f"  Best trial: {study.best_trial.number}")
        print(f"  Best MAE: ${study.best_value:.2f}")

        return study

    def save_trial_metadata(self, trial: optuna.trial.FrozenTrial) -> Path:
        """Save detailed trial metadata to JSON."""

        trial_dir = self.trials_dir / self.study_name
        trial_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "trial_id": trial.number,
            "status": trial.state.name,
            "value": trial.value if trial.value else None,
            "params": trial.params,
            "user_attrs": trial.user_attrs,
            "datetime_start": (trial.datetime_start.isoformat() if trial.datetime_start else None),
            "datetime_complete": (
                trial.datetime_complete.isoformat() if trial.datetime_complete else None
            ),
            "duration_seconds": (
                (trial.datetime_complete - trial.datetime_start).total_seconds()
                if trial.datetime_complete and trial.datetime_start
                else None
            ),
        }

        path = trial_dir / f"trial_{trial.number:04d}.json"
        with open(path, "w") as f:
            json.dump(metadata, f, indent=2)

        return path

    def get_best_params(self) -> dict[str, Any]:
        """Get best hyperparameters from study."""
        study = self.create_study()
        return dict(study.best_trial.params)

    def get_top_k_trials(self, k: int = 5) -> list[optuna.trial.FrozenTrial]:
        """Get top K trials by value."""
        study = self.create_study()
        return sorted(study.trials, key=lambda t: t.value if t.value else float("inf"))[:k]
