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
        self.client = WorkspaceClient()

    def _read_volume_csv(self, source_path: str) -> pd.DataFrame:
        logger.debug("Reading CSV from Databricks Volume: {}", source_path,)

        response = self.client.files.download(source_path)

        with response.contents as file:
            content = file.read()

        data = pd.read_csv(io.BytesIO(content))

        logger.debug(
            "Successfully read {} rows and {} columns",
            data.shape[0],
            data.shape[1],
        )

        return data

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        train_raw_path = self.config.train_raw_path
        test_raw_path = self.config.test_raw_path

        if train_raw_path.exists() and test_raw_path.exists():
            logger.info(
                "Raw data already exists. Skipping ingestion."
            )

            return DataIngestionArtifact(
                train_data_path=train_raw_path,
                test_data_path=test_raw_path,
            )

        logger.info("Starting data ingestion")

        train_data = self._read_volume_csv(self.config.train_source_path)

        test_data = self._read_volume_csv(self.config.test_source_path)

        train_data.to_csv(train_raw_path, index=False)
        test_data.to_csv(test_raw_path, index=False)

        logger.info("Train data saved to {}", train_raw_path,)
        logger.info("Test data saved to {}", test_raw_path,)
        logger.info("Data ingestion completed successfully")

        return DataIngestionArtifact(
            train_data_path=train_raw_path,
            test_data_path=test_raw_path,
        )