"""
Correction pipeline framework.

Each correction is a discrete, testable unit that can be enabled/disabled
via configuration. The CorrectionPipeline applies corrections sequentially,
logging the effect of each step.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd
from loguru import logger

from wrds_data.exceptions import CorrectionError, SchemaValidationError


class CorrectionStep(ABC):
    """
    Abstract base for a single data correction.

    Each correction should be:
        - Independent: can be enabled/disabled without affecting others.
        - Testable: can be tested with synthetic data in isolation.
        - Documented: includes a description and academic citation where applicable.

    Subclasses must implement ``apply()`` and optionally override
    ``required_columns`` and ``validate_input()``.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name (e.g. 'ShareCodeFilter')."""

    @property
    @abstractmethod
    def description(self) -> str:
        """What this correction does, with citation if applicable."""

    @property
    def required_columns(self) -> list[str]:
        """Columns that must be present in the input DataFrame."""
        return []

    def validate_input(self, df: pd.DataFrame) -> None:
        """
        Check that the input DataFrame has the required columns.

        Raises:
            SchemaValidationError: If required columns are missing.
        """
        missing = [c for c in self.required_columns if c not in df.columns]
        if missing:
            raise SchemaValidationError(missing, context=self.name)

    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the correction to a DataFrame.

        Must return a new DataFrame (or the same one unmodified).
        Must NOT modify the input in-place.

        Args:
            df: Input DataFrame with canonical column names.

        Returns:
            Corrected DataFrame.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class CorrectionPipeline:
    """
    Applies a sequence of CorrectionStep instances to a DataFrame.

    Steps are applied in order. Each step logs the row count before/after
    so data loss is transparent.

    Usage::

        pipeline = CorrectionPipeline([
            ShareCodeFilter(config),
            ExchangeCodeFilter(config),
            PriceSignCorrection(config),
        ])
        corrected_df = pipeline.run(raw_df)
    """

    def __init__(self, steps: list[CorrectionStep]) -> None:
        self._steps = steps

    @property
    def steps(self) -> list[CorrectionStep]:
        return list(self._steps)

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all steps sequentially.

        Args:
            df: Input DataFrame.

        Returns:
            Corrected DataFrame after all steps.

        Raises:
            CorrectionError: If any step fails.
        """
        logger.info(
            f"Running correction pipeline ({len(self._steps)} steps) "
            f"on {len(df):,} rows"
        )

        for step in self._steps:
            rows_before = len(df)
            try:
                step.validate_input(df)
                df = step.apply(df)
            except SchemaValidationError:
                raise
            except Exception as e:
                raise CorrectionError(step.name, str(e)) from e

            rows_after = len(df)
            delta = rows_after - rows_before
            if delta != 0:
                logger.info(
                    f"  [{step.name}] {rows_before:,} → {rows_after:,} rows "
                    f"({delta:+,})"
                )
            else:
                logger.debug(f"  [{step.name}] {rows_after:,} rows (no change)")

        logger.info(f"Correction pipeline complete: {len(df):,} rows")
        return df

    def describe(self) -> str:
        """Return a human-readable summary of the pipeline."""
        lines = [f"CorrectionPipeline ({len(self._steps)} steps):"]
        for i, step in enumerate(self._steps, 1):
            lines.append(f"  {i}. {step.name}: {step.description}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._steps)

    def __repr__(self) -> str:
        step_names = [s.name for s in self._steps]
        return f"CorrectionPipeline(steps={step_names})"
