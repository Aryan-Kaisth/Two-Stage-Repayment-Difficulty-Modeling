import io
import pandas as pd
from databricks.sdk import WorkspaceClient
from loguru import logger

from src.entity.artifact_entity import DataIngestionArtifact
from src.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(
        self,
        config: DataIngestionConfig = DataIngestionConfig(),
    ) -> None:
        self.config = config

    def _get_client(self) -> WorkspaceClient:
        """Lazily initializes the Databricks client only when required."""
        logger.debug("Initializing Databricks WorkspaceClient...")
        return WorkspaceClient()

    def _read_volume_csv(
        self, client: WorkspaceClient, source_path: str
    ) -> pd.DataFrame:
        logger.info(f"Extracting dataset from Databricks Volume: {source_path}")

        response = client.files.download(source_path)

        with response.contents as file:
            content = file.read()

        data = pd.read_csv(io.BytesIO(content))

        logger.debug(
            f"Successfully parsed DataFrame from Databricks | "
            f"Shape: {data.shape[0]:,} rows, {data.shape[1]} columns"
        )

        return data

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        logger.info("Starting Data Ingestion Component")

        # Notice we updated the names here to match our new Config schema
        train_file_path = self.config.train_file_path
        test_file_path = self.config.test_file_path

        # Idempotency Check (Cache validation)
        if train_file_path.exists() and test_file_path.exists():
            logger.info(
                "Cached datasets detected locally. "
                "Skipping Databricks extraction to save network bandwidth and compute."
            )
            return DataIngestionArtifact(
                train_file_path=train_file_path,
                test_file_path=test_file_path,
            )

        # Ensure local directories exist before saving
        logger.debug(
            f"Ensuring local directory structure exists at: {train_file_path.parent}"
        )
        train_file_path.parent.mkdir(parents=True, exist_ok=True)
        test_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Extract data from source
        client = self._get_client()

        train_data = self._read_volume_csv(client, self.config.train_source_path)
        test_data = self._read_volume_csv(client, self.config.test_source_path)

        # Load/Save to local disk
        logger.info("Persisting datasets to local storage...")
        train_data.to_csv(train_file_path, index=False)
        logger.success(f"Train dataset serialized and saved to: {train_file_path}")

        test_data.to_csv(test_file_path, index=False)
        logger.success(f"Test dataset serialized and saved to: {test_file_path}")

        logger.info("Data Ingestion Component completed successfully.")

        # Return the properly named Artifact
        return DataIngestionArtifact(
            train_file_path=train_file_path,
            test_file_path=test_file_path,
        )
