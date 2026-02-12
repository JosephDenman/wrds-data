"""Custom exception hierarchy for wrds-data."""


class WRDSDataError(Exception):
    """Base exception for all wrds-data errors."""


class ConfigurationError(WRDSDataError):
    """Invalid or missing configuration."""


class ConnectionError(WRDSDataError):
    """Failed to connect to WRDS or read local data."""


class DatasetNotFoundError(WRDSDataError):
    """Requested dataset is not registered."""

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        super().__init__(f"Dataset '{dataset_name}' is not registered. "
                         f"Use DatasetRegistry.list() to see available datasets.")


class DataNotAvailableError(WRDSDataError):
    """Requested data is not available (missing local file, no WRDS connection, etc.)."""


class CorrectionError(WRDSDataError):
    """Error during data correction pipeline execution."""

    def __init__(self, step_name: str, message: str):
        self.step_name = step_name
        super().__init__(f"Correction '{step_name}' failed: {message}")


class DownloadError(WRDSDataError):
    """Error during bulk data download."""


class TickerResolutionError(WRDSDataError):
    """Could not resolve ticker to PERMNO or vice versa."""

    def __init__(self, identifier: str, as_of: str = ""):
        self.identifier = identifier
        date_msg = f" as of {as_of}" if as_of else ""
        super().__init__(f"Could not resolve '{identifier}'{date_msg}. "
                         f"The identifier may not exist in CRSP or may have changed over time.")


class SchemaValidationError(WRDSDataError):
    """DataFrame does not match expected schema (missing columns, wrong dtypes)."""

    def __init__(self, missing_columns: list[str], context: str = ""):
        self.missing_columns = missing_columns
        ctx = f" in {context}" if context else ""
        super().__init__(f"Missing required columns{ctx}: {missing_columns}")
