from dataclasses import dataclass
from pathlib import Path
from src.constants import (
    APPLICATION_TEST_SOURCE,
    APPLICATION_TRAIN_SOURCE,
    PREPROCESSOR_OBJECT_FILE_PATH,
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
    TRANSFORMATION_ARTIFACTS_DIR,
    VALIDATION_REPORT_DIR,
)


@dataclass(frozen=True)
class DataIngestionConfig:
    train_source_path: str = APPLICATION_TRAIN_SOURCE
    test_source_path: str = APPLICATION_TEST_SOURCE
    train_raw_path: Path = RAW_DATA_PATH / "applications_train.csv"
    test_raw_path: Path = RAW_DATA_PATH / "applications_test.csv"


@dataclass(frozen=True)
class DataValidationConfig:
    report_dir: Path = VALIDATION_REPORT_DIR
    status_file_path: Path = VALIDATION_REPORT_DIR / "status.txt"
    report_file_path: Path = VALIDATION_REPORT_DIR / "validation_report.json"
    drift_file_path: Path = VALIDATION_REPORT_DIR / "data_drift_report.json"


@dataclass(frozen=True)
class DataTransformationConfig:
    processed_data_path: Path = PROCESSED_DATA_PATH
    processed_train_path: Path = PROCESSED_DATA_PATH / "processed_train.parquet"
    processed_test_path: Path = PROCESSED_DATA_PATH / "processed_test.parquet"
    transformation_artifacts_dir: Path = TRANSFORMATION_ARTIFACTS_DIR
    preprocessor_object_file_path: Path = PREPROCESSOR_OBJECT_FILE_PATH
