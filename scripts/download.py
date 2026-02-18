from pathlib import Path
from wrds_data import WRDSDataProvider, WRDSDataConfig
from wrds_data.config import StorageConfig, WRDSConnectionConfig

username = "YOUR_USERNAME"
password = "YOUR_PASSWORD"
config = WRDSDataConfig(
    connection=WRDSConnectionConfig(username=username, password=password),
    storage=StorageConfig(
        cache_dir=Path("/Users/josephdenman/PycharmProjects/wrds-data/data"),  # macOS mount path
        backend="auto",
    ),
)

provider = WRDSDataProvider(config)
provider.download(datasets=['crsp_daily', 'crsp_names', 'compustat_annual', 'ccm_link'], start_year=1995)  # everything goes to the Corsair drive