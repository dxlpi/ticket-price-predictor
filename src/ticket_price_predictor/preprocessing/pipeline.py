"""Pipeline orchestrator for chaining preprocessors."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from .base import Preprocessor, ProcessingResult
from .cleaners import DuplicateHandler, PriceOutlierHandler, TextNormalizer
from .config import PreprocessingConfig
from .transformers import (
    EventMetadataJoiner,
    MissingValueImputer,
    SeatZoneEnricher,
    TemporalFeatureEnricher,
    TypeConverter,
)
from .validators import ReferentialValidator, SchemaValidator, TemporalValidator

logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """Chains multiple preprocessors into a pipeline.

    Features:
    - Sequential execution of preprocessors
    - Aggregates issues and metrics from all stages
    - Optional checkpoint support for debugging/resuming
    - Error handling with detailed stage tracking
    """

    def __init__(
        self,
        stages: list[Preprocessor],
        checkpoint_dir: str | Path | None = None,
        name: str = "pipeline",
    ):
        """Initialize preprocessing pipeline.

        Args:
            stages: List of preprocessors to execute in order
            checkpoint_dir: Optional directory for saving intermediate results
            name: Name of the pipeline (used for logging and checkpoints)
        """
        self.stages = stages
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.name = name

        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def process(self, df: pd.DataFrame) -> ProcessingResult:
        """Process DataFrame through all pipeline stages.

        Args:
            df: Input DataFrame to process

        Returns:
            ProcessingResult with final data and aggregated issues/metrics
        """
        current_df = df
        all_issues: list[str] = []
        all_metrics: dict[str, Any] = {"stages": {}}

        logger.info(f"Starting pipeline '{self.name}' with {len(self.stages)} stages")

        for i, stage in enumerate(self.stages):
            stage_name = stage.__class__.__name__
            logger.info(f"Stage {i + 1}/{len(self.stages)}: {stage_name}")

            try:
                # Process stage
                result = stage.process(current_df)

                # Update current dataframe
                current_df = result.data

                # Aggregate issues
                if result.issues:
                    stage_issues = [f"[{stage_name}] {issue}" for issue in result.issues]
                    all_issues.extend(stage_issues)
                    logger.warning(f"{stage_name} raised {len(result.issues)} issues")

                # Aggregate metrics
                all_metrics["stages"][stage_name] = result.metrics

                # Save checkpoint if configured
                if self.checkpoint_dir:
                    checkpoint_path = (
                        self.checkpoint_dir / f"{self.name}_stage_{i + 1}_{stage_name}.parquet"
                    )
                    current_df.to_parquet(checkpoint_path)
                    logger.debug(f"Checkpoint saved: {checkpoint_path}")

            except Exception as e:
                error_msg = f"Stage {stage_name} failed: {str(e)}"
                logger.error(error_msg, exc_info=True)
                all_issues.append(f"[CRITICAL] {error_msg}")

                # Save failure checkpoint
                if self.checkpoint_dir:
                    failure_path = (
                        self.checkpoint_dir
                        / f"{self.name}_stage_{i + 1}_{stage_name}_FAILED.parquet"
                    )
                    current_df.to_parquet(failure_path)
                    logger.info(f"Failure checkpoint saved: {failure_path}")

                # Continue or abort based on severity
                # For now, we continue processing (graceful degradation)
                # But mark the stage as failed in metrics
                all_metrics["stages"][stage_name] = {"status": "FAILED", "error": str(e)}

        # Add summary metrics
        all_metrics["total_issues"] = len(all_issues)
        all_metrics["final_row_count"] = len(current_df)
        all_metrics["final_column_count"] = len(current_df.columns)

        logger.info(
            f"Pipeline '{self.name}' complete: {len(current_df)} rows, "
            f"{len(current_df.columns)} columns, {len(all_issues)} issues"
        )

        return ProcessingResult(data=current_df, issues=all_issues, metrics=all_metrics)

    def resume_from_checkpoint(self, stage_index: int) -> pd.DataFrame:
        """Resume processing from a saved checkpoint.

        Args:
            stage_index: Stage number to resume from (1-indexed)

        Returns:
            DataFrame from the checkpoint

        Raises:
            FileNotFoundError: If checkpoint doesn't exist
            ValueError: If checkpoint_dir is not configured
        """
        if not self.checkpoint_dir:
            raise ValueError("Cannot resume: checkpoint_dir not configured")

        if stage_index < 1 or stage_index > len(self.stages):
            raise ValueError(f"Invalid stage_index: {stage_index} (must be 1-{len(self.stages)})")

        stage_name = self.stages[stage_index - 1].__class__.__name__
        checkpoint_path = (
            self.checkpoint_dir / f"{self.name}_stage_{stage_index}_{stage_name}.parquet"
        )

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        return pd.read_parquet(checkpoint_path)


class PipelineBuilder:
    """Factory for creating preset and custom pipelines.

    Can be used as a static factory or instantiated with config for convenience:

        # Static usage
        pipeline = PipelineBuilder.build_listings_pipeline(config=config)

        # Instance usage
        builder = PipelineBuilder(config)
        pipeline = builder.build_preset("listings")
    """

    def __init__(self, config: PreprocessingConfig | None = None):
        """Initialize builder with optional configuration.

        Args:
            config: Preprocessing configuration (uses defaults if None)
        """
        self.config = config or PreprocessingConfig()

    def build_preset(
        self,
        dataset_type: str,
        events_df: pd.DataFrame | None = None,
        checkpoint_dir: str | Path | None = None,
    ) -> PreprocessingPipeline:
        """Build a preset pipeline for the given dataset type.

        Args:
            dataset_type: One of "listings", "events", "snapshots"
            events_df: Optional events DataFrame for joins/validation
            checkpoint_dir: Optional checkpoint directory

        Returns:
            PreprocessingPipeline for the specified dataset type

        Raises:
            ValueError: If dataset_type is not recognized
        """
        if dataset_type == "listings":
            return self.build_listings_pipeline(
                events_df=events_df, config=self.config, checkpoint_dir=checkpoint_dir
            )
        elif dataset_type == "events":
            return self.build_events_pipeline(config=self.config, checkpoint_dir=checkpoint_dir)
        elif dataset_type == "snapshots":
            return self.build_snapshots_pipeline(
                events_df=events_df, config=self.config, checkpoint_dir=checkpoint_dir
            )
        else:
            raise ValueError(
                f"Unknown dataset type: {dataset_type}. Must be one of: listings, events, snapshots"
            )

    def build_default(
        self,
        checkpoint_dir: str | Path | None = None,
    ) -> PreprocessingPipeline:
        """Build a default pipeline (listings without event join).

        Args:
            checkpoint_dir: Optional checkpoint directory

        Returns:
            PreprocessingPipeline with default configuration
        """
        return self.build_listings_pipeline(config=self.config, checkpoint_dir=checkpoint_dir)

    @staticmethod
    def build_listings_pipeline(
        events_df: pd.DataFrame | None = None,
        config: PreprocessingConfig | None = None,
        checkpoint_dir: str | Path | None = None,
    ) -> PreprocessingPipeline:
        """Build standard preprocessing pipeline for ticket listings.

        Pipeline stages:
        1. Schema validation
        2. Text normalization (artist, venue, city)
        3. Event metadata join (venue_capacity)
        4. Type conversion (timestamps, prices)
        5. Seat zone enrichment
        6. Temporal features (hour_of_day, days_to_event, is_weekend)
        7. Missing value imputation
        8. Price outlier detection
        9. Duplicate detection
        10. Temporal validation
        11. Referential validation

        Args:
            events_df: Optional events DataFrame for joining venue_capacity
            config: Optional preprocessing configuration
            checkpoint_dir: Optional checkpoint directory

        Returns:
            PreprocessingPipeline configured for listings
        """
        config = config or PreprocessingConfig()
        stages: list[Preprocessor] = []

        # 1. Schema validation
        stages.append(SchemaValidator("listings"))

        # 2. Text normalization
        stages.append(TextNormalizer(config))

        # 3. Event metadata join (if events provided)
        if events_df is not None:
            stages.append(EventMetadataJoiner(events_df, config))

        # 4. Type conversion
        stages.append(TypeConverter(config))

        # 5. Seat zone enrichment
        stages.append(SeatZoneEnricher(config))

        # 6. Temporal features
        stages.append(TemporalFeatureEnricher(config))

        # 7. Missing value imputation
        stages.append(MissingValueImputer(config))

        # 8. Price outlier detection
        stages.append(PriceOutlierHandler(config))

        # 9. Duplicate detection
        stages.append(DuplicateHandler(time_window_hours=6))

        # 10. Temporal validation
        stages.append(TemporalValidator(allow_past_events=True))

        # 11. Referential validation
        stages.append(ReferentialValidator("listings", events_df=events_df))

        return PreprocessingPipeline(stages, checkpoint_dir=checkpoint_dir, name="listings")

    @staticmethod
    def build_events_pipeline(
        config: PreprocessingConfig | None = None,
        checkpoint_dir: str | Path | None = None,
    ) -> PreprocessingPipeline:
        """Build standard preprocessing pipeline for event metadata.

        Pipeline stages:
        1. Schema validation
        2. Text normalization
        3. Type conversion
        4. Missing value imputation
        5. Temporal validation

        Args:
            config: Optional preprocessing configuration
            checkpoint_dir: Optional checkpoint directory

        Returns:
            PreprocessingPipeline configured for events
        """
        config = config or PreprocessingConfig()
        stages: list[Preprocessor] = [
            SchemaValidator("events"),
            TextNormalizer(config),
            TypeConverter(config),
            MissingValueImputer(config),
            TemporalValidator(allow_past_events=True),
        ]

        return PreprocessingPipeline(stages, checkpoint_dir=checkpoint_dir, name="events")

    @staticmethod
    def build_snapshots_pipeline(
        events_df: pd.DataFrame | None = None,
        config: PreprocessingConfig | None = None,
        checkpoint_dir: str | Path | None = None,
    ) -> PreprocessingPipeline:
        """Build standard preprocessing pipeline for price snapshots.

        Pipeline stages:
        1. Schema validation
        2. Type conversion
        3. Price outlier detection
        4. Temporal validation
        5. Referential validation

        Args:
            events_df: Optional events DataFrame for referential validation
            config: Optional preprocessing configuration
            checkpoint_dir: Optional checkpoint directory

        Returns:
            PreprocessingPipeline configured for snapshots
        """
        config = config or PreprocessingConfig()
        stages: list[Preprocessor] = [
            SchemaValidator("snapshots"),
            TypeConverter(config),
            PriceOutlierHandler(config),
            TemporalValidator(allow_past_events=True),
            ReferentialValidator("snapshots", events_df=events_df),
        ]

        return PreprocessingPipeline(stages, checkpoint_dir=checkpoint_dir, name="snapshots")

    @staticmethod
    def build_custom_pipeline(
        stages: list[Preprocessor],
        name: str = "custom",
        checkpoint_dir: str | Path | None = None,
    ) -> PreprocessingPipeline:
        """Build a custom pipeline with specified stages.

        Args:
            stages: List of preprocessor instances
            name: Name for the pipeline
            checkpoint_dir: Optional checkpoint directory

        Returns:
            PreprocessingPipeline with custom configuration
        """
        return PreprocessingPipeline(stages, checkpoint_dir=checkpoint_dir, name=name)
