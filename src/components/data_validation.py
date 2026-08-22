from datetime import datetime
import json
from typing import Any, Dict, Optional, Tuple

from evidently import Report
from evidently.metrics import ValueDrift
from evidently.presets import DataDriftPreset
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
        data_ingestion_artifact: Optional[DataIngestionArtifact] = None,
        config: Optional[DataValidationConfig] = None,
    ) -> None:
        self.config = config if config is not None else DataValidationConfig()
        self.data_ingestion_artifact = data_ingestion_artifact

    def _load_and_preprocess_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self.data_ingestion_artifact is None:
            raise ValueError(
                "DataIngestionArtifact must be provided to load and preprocess training/testing data."
            )

        train_path = self.data_ingestion_artifact.train_file_path
        test_path = self.data_ingestion_artifact.test_file_path

        logger.debug("Loading training partition from path: {}", train_path)
        train_df = pd.read_csv(train_path)

        logger.debug("Loading test partition from path: {}", test_path)
        test_df = pd.read_csv(test_path)

        # Normalize column casing across all raw inputs
        train_df.columns = train_df.columns.str.lower()
        test_df.columns = test_df.columns.str.lower()

        logger.debug(
            "Partitions loaded and normalized | Train: {:,} rows × {} cols | Test: {:,} rows × {} cols",
            train_df.shape[0],
            train_df.shape[1],
            test_df.shape[0],
            test_df.shape[1],
        )

        return train_df, test_df

    def _validate_schema(
        self,
        train_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        if train_df is None and test_df is None:
            raise ValueError("At least one DataFrame (train_df or test_df) must be provided for schema validation.")

        validation_status = True
        report: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Check training schema contract
        if train_df is not None:
            report["train_validation"] = {"status": True, "errors": []}
            try:
                logger.debug("Running Pandera schema checks on train partition...")
                ApplicationTrainSchema.validate(train_df, lazy=True)
                logger.success("Train schema validation passed")
            except pa.errors.SchemaErrors as err:
                validation_status = False
                report["train_validation"]["status"] = False
                report["train_validation"]["errors"] = err.failure_cases.to_dict(orient="records")
                logger.error(
                    "Train schema validation failed with {:,} schema violations",
                    len(err.failure_cases),
                )

        # Check test/inference schema contract
        if test_df is not None:
            report["test_validation"] = {"status": True, "errors": []}
            try:
                logger.debug("Running Pandera schema checks on test partition...")
                ApplicationTestSchema.validate(test_df, lazy=True)
                logger.success("Test schema validation passed")
            except pa.errors.SchemaErrors as err:
                validation_status = False
                report["test_validation"]["status"] = False
                report["test_validation"]["errors"] = err.failure_cases.to_dict(orient="records")
                logger.error(
                    "Test schema validation failed with {:,} schema violations",
                    len(err.failure_cases),
                )

        return validation_status, report

    def _detect_dataset_drift(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> Dict[str, Any]:
        logger.info("Evaluating feature and target distribution drift via Evidently...")

        # Feature distribution shifts across splits
        metrics_list = [DataDriftPreset(include_tests=True)]

        # Include label drift if ground-truth target exists in both sets
        if "target" in train_df.columns and "target" in test_df.columns:
            logger.debug("Target column found in both datasets — tracking label distribution drift")
            metrics_list.append(ValueDrift(column="target"))
            reference_df = train_df
            current_df = test_df
        else:
            logger.debug("Target column missing in test set — evaluating covariate drift only")
            reference_df = train_df.drop(columns=["target"], errors="ignore")
            current_df = test_df.drop(columns=["target"], errors="ignore")

        drift_report = Report(metrics=metrics_list)
        drift_results = drift_report.run(
            reference_data=reference_df,
            current_data=current_df,
        )

        drift_dict = json.loads(drift_results.json())

        with open(self.config.drift_file_path, "w") as f:
            json.dump(drift_dict, f, indent=4)

        logger.success("Drift analysis written to -> {}", self.config.drift_file_path)

        return drift_dict

    def _save_validation_artifacts(
        self, validation_status: bool, report: Dict[str, Any]
    ) -> None:
        # Gatekeeper token used by orchestrators to halt pipeline if schema breaks
        with open(self.config.status_file_path, "w") as f:
            f.write(f"Validation status: {validation_status}\n")

        # Detailed schema inspection payload
        with open(self.config.report_file_path, "w") as f:
            json.dump(report, f, indent=4)

        logger.success(
            "Validation status written: {} | Report JSON: {}",
            self.config.status_file_path,
            self.config.report_file_path,
        )

    def initiate_data_validation(self) -> DataValidationArtifact:
        if self.data_ingestion_artifact is None:
            raise ValueError(
                "DataIngestionArtifact is required to execute initiate_data_validation()."
            )

        logger.info("Starting data validation pipeline")

        logger.debug("Ensuring report directory exists: {}", self.config.report_dir)
        self.config.report_dir.mkdir(parents=True, exist_ok=True)

        train_df, test_df = self._load_and_preprocess_data()
        validation_status, report = self._validate_schema(train_df, test_df)

        self._detect_dataset_drift(train_df, test_df)
        self._save_validation_artifacts(validation_status, report)

        if validation_status:
            logger.success("Data validation stage passed successfully")
        else:
            logger.warning("Data validation stage flagged schema/drift issues")

        return DataValidationArtifact(
            validation_status=validation_status,
            status_file_path=self.config.status_file_path,
            report_file_path=self.config.report_file_path,
            drift_file_path=self.config.drift_file_path,
            train_file_path=self.data_ingestion_artifact.train_file_path,
            test_file_path=self.data_ingestion_artifact.test_file_path,
        )