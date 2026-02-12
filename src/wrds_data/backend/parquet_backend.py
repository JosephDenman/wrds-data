"""
Local Parquet file backend.

Reads pre-downloaded Parquet files organized by dataset name and year.
Supports chunked files (one per year) for efficient date-range filtering.

Directory layout::

    {cache_dir}/
        crsp_daily/
            crsp_daily_2010.parquet
            crsp_daily_2011.parquet
            ...
        compustat_annual/
            compustat_annual_1960_1969.parquet
            compustat_annual_1970_1979.parquet
            ...
        ccm_link/
            ccm_link.parquet
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from wrds_data.backend.base import DataBackend
from wrds_data.config import StorageConfig
from wrds_data.datasets.base import DatasetDefinition
from wrds_data.exceptions import DataNotAvailableError


class ParquetBackend(DataBackend):
    """
    Backend that reads from local Parquet files.

    Files are organized as ``{cache_dir}/{dataset_name}/{dataset_name}_{year}.parquet``
    for year-chunked datasets, or ``{cache_dir}/{dataset_name}/{dataset_name}.parquet``
    for single-file datasets.
    """

    def __init__(self, config: StorageConfig) -> None:
        self._cache_dir = Path(config.cache_dir)

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    def query(
        self,
        dataset: DatasetDefinition,
        columns: list[str] | None = None,
        filters: dict[str, Any] | None = None,
        date_range: tuple[date, date] | None = None,
    ) -> pd.DataFrame:
        """
        Read data from local Parquet files.

        Optimizes I/O by only loading files whose year range overlaps
        the requested date range.
        """
        dataset_dir = self._cache_dir / dataset.name

        if not dataset_dir.exists():
            raise DataNotAvailableError(
                f"No local data for '{dataset.name}'. "
                f"Expected directory: {dataset_dir}. "
                f"Run provider.download(datasets=['{dataset.name}']) first."
            )

        # Find all Parquet files for this dataset
        parquet_files = sorted(dataset_dir.glob("*.parquet"))
        if not parquet_files:
            raise DataNotAvailableError(
                f"No Parquet files found in {dataset_dir}"
            )

        # Filter files by year range if date_range is provided
        if date_range is not None:
            start_year, end_year = date_range[0].year, date_range[1].year
            parquet_files = self._filter_files_by_year(
                parquet_files, start_year, end_year
            )
            if not parquet_files:
                raise DataNotAvailableError(
                    f"No local data for '{dataset.name}' covering "
                    f"{date_range[0]} to {date_range[1]}"
                )

        # Read and concatenate
        dfs: list[pd.DataFrame] = []
        for fpath in parquet_files:
            try:
                df = pd.read_parquet(fpath)
                dfs.append(df)
            except Exception as e:
                logger.warning(f"Failed to read {fpath}: {e}")

        if not dfs:
            raise DataNotAvailableError(
                f"All Parquet files for '{dataset.name}' failed to read"
            )

        df = pd.concat(dfs, ignore_index=True)

        # Rename WRDS columns → canonical if needed
        # (files may have WRDS column names if downloaded raw)
        rename_map = dataset.wrds_to_canonical
        rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
        if rename_map:
            df = df.rename(columns=rename_map)

        # Apply date range filter (files are by year, so we need exact filtering)
        if date_range is not None:
            date_col = dataset.date_column
            if date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col])
                start_dt = pd.Timestamp(date_range[0])
                end_dt = pd.Timestamp(date_range[1])
                df = df[(df[date_col] >= start_dt) & (df[date_col] <= end_dt)]

        # Apply column filters
        if filters:
            for col_name, value in filters.items():
                if col_name not in df.columns:
                    continue
                if isinstance(value, (list, tuple, set)):
                    df = df[df[col_name].isin(value)]
                else:
                    df = df[df[col_name] == value]

        # Select requested columns
        if columns is not None:
            available = [c for c in columns if c in df.columns]
            df = df[available]

        logger.debug(
            f"Read {len(df):,} rows from {len(parquet_files)} file(s) "
            f"for '{dataset.name}'"
        )
        return df

    def is_available(self, dataset: DatasetDefinition) -> bool:
        """Check if local Parquet files exist for this dataset."""
        dataset_dir = self._cache_dir / dataset.name
        if not dataset_dir.exists():
            return False
        return len(list(dataset_dir.glob("*.parquet"))) > 0

    def close(self) -> None:
        """No resources to release for file-based backend."""

    @staticmethod
    def _filter_files_by_year(
        files: list[Path], start_year: int, end_year: int
    ) -> list[Path]:
        """
        Filter Parquet files to only those overlapping [start_year, end_year].

        Supports filename patterns:
            - dataset_2010.parquet (single year)
            - dataset_2010_2019.parquet (year range)
            - dataset.parquet (no year, always included)
        """
        result: list[Path] = []
        for fpath in files:
            stem = fpath.stem

            # Try year range pattern: name_YYYY_YYYY
            range_match = re.search(r"_(\d{4})_(\d{4})$", stem)
            if range_match:
                file_start = int(range_match.group(1))
                file_end = int(range_match.group(2))
                if file_start <= end_year and file_end >= start_year:
                    result.append(fpath)
                continue

            # Try single year pattern: name_YYYY
            year_match = re.search(r"_(\d{4})$", stem)
            if year_match:
                file_year = int(year_match.group(1))
                if start_year <= file_year <= end_year:
                    result.append(fpath)
                continue

            # No year in filename — always include
            result.append(fpath)

        return result
