import io
import pandas as pd
from databricks.sdk import WorkspaceClient
from loguru import logger

from src.entity.artifact_entity import DataIngestionArtifact
from src.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(
        self,
        config: DataIngestionConfig,
    ) -> None:
        self.config = config

    def _get_client(self) -> WorkspaceClient:
        """Lazily initialize the Databricks client only when fetching remote volumes."""
        logger.debug("Connecting to Databricks WorkspaceClient session")
        return WorkspaceClient()

    def _read_volume_csv(
        self, client: WorkspaceClient, source_path: str
    ) -> pd.DataFrame:
        logger.info("Fetching remote dataset from Databricks Volume: {}", source_path)

        response = client.files.download(source_path)

        with response.contents as file:
            content = file.read()

        payload_mb = len(content) / (1024 * 1024)
        logger.debug("Downloaded {:.2f} MB payload from {}", payload_mb, source_path)

        data = pd.read_csv(io.BytesIO(content))

        logger.debug(
            "Parsed CSV stream into memory | Dimensions: {:,} rows × {} columns",
            data.shape[0],
            data.shape[1],
        )

        return data

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        logger.info("Starting data ingestion component")

        train_file_path = self.config.train_file_path
        test_file_path = self.config.test_file_path

        # Cache check: avoid re-downloading if local raw files are already present
        if train_file_path.exists() and test_file_path.exists():
            logger.info(
                "Local cache hit at '{}' and '{}' — skipping remote volume pull",
                train_file_path,
                test_file_path,
            )
            return DataIngestionArtifact(
                train_file_path=train_file_path,
                test_file_path=test_file_path,
            )

        # Prepare target directories for raw dumps
        logger.debug("Preparing local destination folder: {}", train_file_path.parent)
        train_file_path.parent.mkdir(parents=True, exist_ok=True)
        test_file_path.parent.mkdir(parents=True, exist_ok=True)

        client = self._get_client()

        logger.info("Pulling raw partitions from Databricks Unity Catalog volumes...")
        train_data = self._read_volume_csv(client, self.config.train_source_path)
        test_data = self._read_volume_csv(client, self.config.test_source_path)

        # Serialize directly to disk for downstream pipeline components
        logger.info("Persisting raw datasets locally...")
        train_data.to_csv(train_file_path, index=False)
        logger.success(
            "Saved train partition -> {} ({:,} rows, {} cols)",
            train_file_path,
            train_data.shape[0],
            train_data.shape[1],
        )

        test_data.to_csv(test_file_path, index=False)
        logger.success(
            "Saved test partition -> {} ({:,} rows, {} cols)",
            test_file_path,
            test_data.shape[0],
            test_data.shape[1],
        )

        logger.info("Data ingestion completed")

        return DataIngestionArtifact(
            train_file_path=train_file_path,
            test_file_path=test_file_path,
        )