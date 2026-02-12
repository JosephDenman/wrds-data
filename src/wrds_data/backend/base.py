"""Abstract base for data backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from typing import Any

import pandas as pd

from wrds_data.datasets.base import DatasetDefinition


class DataBackend(ABC):
    """
    Abstract interface for accessing WRDS data.

    Implementations include:
        - WRDSBackend: live connection to WRDS PostgreSQL
        - ParquetBackend: local Parquet file reader
    """

    @abstractmethod
    def query(
        self,
        dataset: DatasetDefinition,
        columns: list[str] | None = None,
        filters: dict[str, Any] | None = None,
        date_range: tuple[date, date] | None = None,
    ) -> pd.DataFrame:
        """
        Fetch data from a dataset.

        Args:
            dataset: The dataset definition to query.
            columns: Canonical column names to fetch. None = all columns.
            filters: Column-value filters (e.g. {"permno": 10107}).
                Values can be scalars or lists (for IN clauses).
            date_range: (start_date, end_date) inclusive range on the
                dataset's date column.

        Returns:
            DataFrame with canonical column names.

        Raises:
            DataNotAvailableError: If the data cannot be retrieved.
            ConnectionError: If the backend cannot connect.
        """

    @abstractmethod
    def is_available(self, dataset: DatasetDefinition) -> bool:
        """Check whether this backend can serve the given dataset."""

    @abstractmethod
    def close(self) -> None:
        """Release any resources (connections, file handles)."""
