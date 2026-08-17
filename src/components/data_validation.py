from datetime import datetime
import json
from typing import Any, Dict, Tuple

from evidently import Report
from evidently.presets import DataDriftPreset
from evidently.metrics import ValueDrift
from loguru import logger
import pandas as pd
import pandera.pandas as pa

from configs.schema import (
    ApplicationTestSchema,
    ApplicationTrainSchema,
)
from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
)
from src.entity.config_entity import DataValidationConfig


class DataValidation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        config: DataValidationConfig = DataValidationConfig(),
    ) -> None:
        self.config = config
        self.data_ingestion_artifact = data_ingestion_artifact

    def _load_and_preprocess_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_path = self.data_ingestion_artifact.train_data_path
        test_path = self.data_ingestion_artifact.test_data_path

        logger.debug("Loading train data from {}", train_path)
        train_df = pd.read_csv(train_path)

        logger.debug("Loading test data from {}", test_path)
        test_df = pd.read_csv(test_path)

        # Standardize column headers to lowercase
        train_df.columns = train_df.columns.str.lower()
        test_df.columns = test_df.columns.str.lower()

        return train_df, test_df

    def _validate_schema(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> Tuple[bool, Dict[str, Any]]:
        validation_status = True
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "train_validation": {"status": True, "errors": []},
            "test_validation": {"status": True, "errors": []},
        }

        # Validate Train Split
        try:
            ApplicationTrainSchema.validate(train_df, lazy=True)
            logger.info("Train dataset passed schema validation")
        except pa.errors.SchemaErrors as err:
            validation_status = False
            report["train_validation"]["status"] = False
            report["train_validation"]["errors"] = err.failure_cases.to_dict(
                orient="records"
            )
            logger.error(
                "Train dataset failed schema validation with {} issues",
                len(err.failure_cases),
            )

        # Validate Test Split
        try:
            ApplicationTestSchema.validate(test_df, lazy=True)
            logger.info("Test dataset passed schema validation")
        except pa.errors.SchemaErrors as err:
            validation_status = False
            report["test_validation"]["status"] = False
            report["test_validation"]["errors"] = err.failure_cases.to_dict(
                orient="records"
            )
            logger.error(
                "Test dataset failed schema validation with {} issues",
                len(err.failure_cases),
            )

        return validation_status, report

    def _detect_dataset_drift(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> Dict[str, Any]:
        logger.info("Running Evidently data & target drift analysis")

        # Base metrics: Dataset-level feature drift
        metrics_list = [DataDriftPreset(include_tests=True)] # included tests

        # If target exists in both, add ValueDrift for the target
        if "target" in train_df.columns and "target" in test_df.columns:
            metrics_list.append(ValueDrift(column="target"))
            reference_df = train_df
            current_df = test_df
        else:
            # Fallback if target is missing in test
            reference_df = train_df.drop(columns=["target"], errors="ignore")
            current_df = test_df.drop(columns=["target"], errors="ignore")

        drift_report = Report(metrics=metrics_list)
        drift_results = drift_report.run(
            reference_data=reference_df,
            current_data=current_df,
        )

        # Parse raw Evidently JSON output into a Python dict
        drift_dict = json.loads(drift_results.json())

        # Save with 4-space indentation
        with open(self.config.drift_file_path, "w") as f:
            json.dump(drift_dict, f, indent=4)

        logger.info(
            "Data & target drift report saved successfully to {}",
            self.config.drift_file_path,
        )

        return drift_dict

    def _save_validation_artifacts(
        self, validation_status: bool, report: Dict[str, Any]
    ) -> None:
        # Save status file (Gatekeeper)
        with open(self.config.status_file_path, "w") as f:
            f.write(f"Validation status: {validation_status}\n")

        # Save schema validation report (JSON)
        with open(self.config.report_file_path, "w") as f:
            json.dump(report, f, indent=4)

        logger.info(
            "Schema report saved to {} | Status written to {}",
            self.config.report_file_path,
            self.config.status_file_path,
        )

    def initiate_data_validation(self) -> DataValidationArtifact:
        logger.info("Starting data validation pipeline")

        self.config.report_dir.mkdir(parents=True, exist_ok=True)

        # Load & preprocess
        train_df, test_df = self._load_and_preprocess_data()

        # Validate schemas
        validation_status, report = self._validate_schema(train_df, test_df)

        # Detect and save data & target drift
        self._detect_dataset_drift(train_df, test_df)

        # Save validation files
        self._save_validation_artifacts(validation_status, report)

        logger.info(
            "Validation pipeline finished with status: {}", validation_status
        )

        return DataValidationArtifact(
            validation_status=validation_status,
            status_file_path=self.config.status_file_path,
            report_file_path=self.config.report_file_path,
            drift_file_path=self.config.drift_file_path,
            train_data_path=self.data_ingestion_artifact.train_data_path,
            test_data_path=self.data_ingestion_artifact.test_data_path,
        )