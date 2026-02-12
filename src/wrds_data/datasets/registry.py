"""
Extensible dataset registry.

Pre-registers all built-in WRDS datasets. Users can register custom datasets
for tables not covered by the library.
"""

from __future__ import annotations

from wrds_data.datasets.base import DatasetDefinition
from wrds_data.exceptions import DatasetNotFoundError


class DatasetRegistry:
    """
    Singleton registry of known WRDS dataset definitions.

    Built-in datasets are registered automatically on first access.
    Users can register additional datasets via ``register()``.

    Usage::

        registry = DatasetRegistry.instance()
        crsp = registry.get("crsp_daily")
        registry.register(my_custom_dataset)
    """

    _instance: DatasetRegistry | None = None
    _initialized: bool = False

    def __init__(self) -> None:
        self._datasets: dict[str, DatasetDefinition] = {}

    @classmethod
    def instance(cls) -> DatasetRegistry:
        """Return the singleton registry, creating it on first call."""
        if cls._instance is None:
            cls._instance = cls()
        if not cls._initialized:
            cls._register_builtins(cls._instance)
            cls._initialized = True
        return cls._instance

    @staticmethod
    def _register_builtins(registry: DatasetRegistry) -> None:
        """Register all built-in dataset definitions."""
        # Import here to avoid circular imports
        from wrds_data.datasets.ccm import CCM_LINK
        from wrds_data.datasets.compustat import (
            COMPUSTAT_ANNUAL,
            COMPUSTAT_QUARTERLY,
        )
        from wrds_data.datasets.crsp import (
            CRSP_DAILY,
            CRSP_DELISTING,
            CRSP_MONTHLY,
            CRSP_NAMES,
        )

        for dataset in [
            CRSP_DAILY,
            CRSP_MONTHLY,
            CRSP_DELISTING,
            CRSP_NAMES,
            COMPUSTAT_ANNUAL,
            COMPUSTAT_QUARTERLY,
            CCM_LINK,
        ]:
            registry.register(dataset)

    def register(self, dataset: DatasetDefinition) -> None:
        """Register a dataset definition. Overwrites if name already exists."""
        self._datasets[dataset.name] = dataset

    def get(self, name: str) -> DatasetDefinition:
        """
        Look up a dataset by name.

        Raises:
            DatasetNotFoundError: If the name is not registered.
        """
        if name not in self._datasets:
            raise DatasetNotFoundError(name)
        return self._datasets[name]

    def list(self) -> list[str]:
        """Return sorted list of registered dataset names."""
        return sorted(self._datasets.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._datasets

    def __len__(self) -> int:
        return len(self._datasets)
