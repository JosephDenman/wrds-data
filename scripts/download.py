from pathlib import Path
from wrds_data import WRDSDataProvider, WRDSDataConfig
from wrds_data.config import StorageConfig, WRDSConnectionConfig

config = WRDSDataConfig(
    connection=WRDSConnectionConfig(username="your_username", password="your_password"),
    storage=StorageConfig(
        cache_dir=Path("/Volumes/CorsairDrive/wrds_data"),  # macOS mount path
        backend="auto",
    ),
)

provider = WRDSDataProvider(config)
provider.download()  # everything goes to the Corsair drive