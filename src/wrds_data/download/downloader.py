"""
Chunked, resumable bulk downloader for WRDS datasets.

Downloads entire WRDS tables to local Parquet files, chunked by year.
Supports resume (skips completed chunks) and failed-chunk markers.

Usage::

    downloader = BulkDownloader(wrds_backend, storage_config)
    downloader.download_all()           # All registered datasets
    downloader.download(["crsp_daily"]) # Specific datasets

CLI::

    python -m wrds_data.download --datasets crsp_daily compustat_annual
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
from loguru import logger
from tqdm import tqdm

from wrds_data.backend.wrds_backend import WRDSBackend
from wrds_data.config import StorageConfig
from wrds_data.datasets.base import DatasetDefinition
from wrds_data.datasets.registry import DatasetRegistry
from wrds_data.exceptions import DownloadError


class BulkDownloader:
    """
    Downloads WRDS datasets to local Parquet files.

    Each dataset is split into year-based chunks. Existing chunks
    are skipped (resume support). Failed chunks are marked with
    a .FAILED file for retry.
    """

    def __init__(
        self,
        backend: WRDSBackend,
        storage: StorageConfig,
        start_year: int = 1960,
        end_year: int | None = None,
    ) -> None:
        self._backend = backend
        self._storage = storage
        self._start_year = start_year
        self._end_year = end_year or date.today().year

    def download(
        self,
        dataset_names: list[str] | None = None,
        force: bool = False,
    ) -> dict[str, int]:
        """
        Download datasets to local Parquet files.

        Args:
            dataset_names: Names of datasets to download. None = all registered.
            force: If True, re-download even if local files exist.

        Returns:
            Dict mapping dataset name → total rows downloaded.
        """
        registry = DatasetRegistry.instance()

        if dataset_names is None:
            dataset_names = registry.list()

        results: dict[str, int] = {}

        for ds_name in dataset_names:
            try:
                dataset = registry.get(ds_name)
                total_rows = self._download_dataset(dataset, force=force)
                results[ds_name] = total_rows
            except Exception as e:
                logger.error(f"Failed to download '{ds_name}': {e}")
                results[ds_name] = -1

        return results

    def _download_dataset(
        self,
        dataset: DatasetDefinition,
        force: bool = False,
    ) -> int:
        """Download a single dataset in year-based chunks."""
        output_dir = Path(self._storage.cache_dir) / dataset.name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Reference tables (e.g. crsp_names, ccm_link) should be downloaded
        # in full without date filtering. Their date_column represents validity
        # ranges, not observation timestamps, so date-chunking misses records
        # with date_column values outside the requested range.
        if dataset.is_reference_table:
            return self._download_reference_table(dataset, output_dir, force)

        chunk_years = dataset.default_chunk_years
        year_ranges = self._compute_year_ranges(
            self._start_year, self._end_year, chunk_years
        )

        total_rows = 0
        logger.info(
            f"Downloading '{dataset.name}' ({len(year_ranges)} chunks, "
            f"{chunk_years} year(s) each)"
        )

        for start_yr, end_yr in tqdm(year_ranges, desc=dataset.name, unit="chunk"):
            # Determine output filename
            if start_yr == end_yr:
                fname = f"{dataset.name}_{start_yr}.parquet"
            else:
                fname = f"{dataset.name}_{start_yr}_{end_yr}.parquet"

            output_path = output_dir / fname
            failed_marker = output_dir / f"{fname}.FAILED"

            # Skip if already downloaded (unless force)
            if output_path.exists() and not force:
                # Count existing rows
                try:
                    existing = pd.read_parquet(output_path)
                    total_rows += len(existing)
                    logger.debug(f"  Skipping {fname} (exists, {len(existing):,} rows)")
                    continue
                except Exception:
                    logger.warning(f"  Corrupt file {fname}, re-downloading")

            # Clean up old failed marker
            if failed_marker.exists():
                failed_marker.unlink()

            try:
                start_date = date(start_yr, 1, 1)
                end_date = date(end_yr, 12, 31)

                df = self._backend.query(
                    dataset,
                    date_range=(start_date, end_date),
                )

                if len(df) == 0:
                    logger.debug(f"  No data for {fname}")
                    continue

                df.to_parquet(output_path, index=False)
                total_rows += len(df)
                logger.debug(f"  Saved {fname}: {len(df):,} rows")

            except Exception as e:
                # Mark as failed for retry
                failed_marker.touch()
                raise DownloadError(
                    f"Failed to download chunk {fname}: {e}"
                ) from e

        logger.info(
            f"Downloaded '{dataset.name}': {total_rows:,} total rows"
        )
        return total_rows

    def _download_reference_table(
        self,
        dataset: DatasetDefinition,
        output_dir: Path,
        force: bool = False,
    ) -> int:
        """
        Download a reference/dimension table in full (no date filtering).

        Reference tables like crsp_names and ccm_link store validity date
        ranges (namedt/nameendt, linkdt/linkenddt). Date-chunking would miss
        records with start dates outside the requested range — e.g. a stock
        listed in 1985 that is still actively trading would be excluded from
        a 2010-2026 download.
        """
        fname = f"{dataset.name}.parquet"
        output_path = output_dir / fname
        failed_marker = output_dir / f"{fname}.FAILED"

        logger.info(f"Downloading '{dataset.name}' (reference table, full download)")

        # Skip if already downloaded (unless force)
        if output_path.exists() and not force:
            try:
                existing = pd.read_parquet(output_path)
                logger.info(
                    f"  Skipping {fname} (exists, {len(existing):,} rows)"
                )
                return len(existing)
            except Exception:
                logger.warning(f"  Corrupt file {fname}, re-downloading")

        # Clean up old failed marker
        if failed_marker.exists():
            failed_marker.unlink()

        try:
            # No date_range → full table download
            df = self._backend.query(dataset)

            if len(df) == 0:
                logger.warning(f"  No data returned for {dataset.name}")
                return 0

            # Remove any old date-chunked files that may exist from previous
            # downloads (e.g. crsp_names_2010_2026.parquet)
            for old_file in output_dir.glob(f"{dataset.name}_*.parquet"):
                logger.info(f"  Removing old chunked file: {old_file.name}")
                old_file.unlink()

            df.to_parquet(output_path, index=False)
            logger.info(
                f"  Downloaded '{dataset.name}': {len(df):,} rows"
            )
            return len(df)

        except Exception as e:
            failed_marker.touch()
            raise DownloadError(
                f"Failed to download reference table '{dataset.name}': {e}"
            ) from e

    @staticmethod
    def _compute_year_ranges(
        start_year: int, end_year: int, chunk_years: int
    ) -> list[tuple[int, int]]:
        """Compute (start_year, end_year) tuples for chunked downloading."""
        ranges: list[tuple[int, int]] = []
        yr = start_year
        while yr <= end_year:
            chunk_end = min(yr + chunk_years - 1, end_year)
            ranges.append((yr, chunk_end))
            yr = chunk_end + 1
        return ranges

    def status(self) -> dict[str, dict[str, int]]:
        """
        Check download status for all registered datasets.

        Returns:
            Dict mapping dataset_name → {"files": N, "rows": M, "failed": F}
        """
        registry = DatasetRegistry.instance()
        result: dict[str, dict[str, int]] = {}

        for ds_name in registry.list():
            ds_dir = Path(self._storage.cache_dir) / ds_name
            if not ds_dir.exists():
                result[ds_name] = {"files": 0, "rows": 0, "failed": 0}
                continue

            parquet_files = list(ds_dir.glob("*.parquet"))
            failed_files = list(ds_dir.glob("*.FAILED"))

            total_rows = 0
            for f in parquet_files:
                try:
                    total_rows += len(pd.read_parquet(f))
                except Exception:
                    pass

            result[ds_name] = {
                "files": len(parquet_files),
                "rows": total_rows,
                "failed": len(failed_files),
            }

        return result
