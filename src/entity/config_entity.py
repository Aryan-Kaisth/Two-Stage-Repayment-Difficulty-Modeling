from dataclasses import dataclass
from pathlib import Path

from src.constants import (
    RAW_DATA_DIR,
    DATABRICKS_TRAIN_SOURCE_PATH,
    DATABRICKS_TEST_SOURCE_PATH,
    DATA_VALIDATION_DIR,
    DATA_TRANSFORMATION_DIR,
    MODEL_EVALUATION_DIR,
    MODEL_TRAINER_DIR,
    MODEL_CALIBRATION_DIR,
)


@dataclass(frozen=True)
class DataIngestionConfig:
    train_source_path: str = DATABRICKS_TRAIN_SOURCE_PATH
    test_source_path: str = DATABRICKS_TEST_SOURCE_PATH
    train_file_path: Path = RAW_DATA_DIR / "train.csv"
    test_file_path: Path = RAW_DATA_DIR / "test.csv"


@dataclass(frozen=True)
class DataValidationConfig:
    report_dir: Path = DATA_VALIDATION_DIR
    status_file_path: Path = DATA_VALIDATION_DIR / "status.txt"
    report_file_path: Path = DATA_VALIDATION_DIR / "validation_report.json"
    drift_file_path: Path = DATA_VALIDATION_DIR / "drift_report.json"


@dataclass(frozen=True)
class DataTransformationConfig:
    transformation_artifacts_dir: Path = DATA_TRANSFORMATION_DIR
    processed_train_path: Path = DATA_TRANSFORMATION_DIR / "train_processed.parquet"
    processed_test_path: Path = DATA_TRANSFORMATION_DIR / "test_processed.parquet"
    preprocessor_path: Path = DATA_TRANSFORMATION_DIR / "preprocessor.joblib"


@dataclass(frozen=True)
class ModelTrainerConfig:
    model_trainer_dir: Path = MODEL_TRAINER_DIR
    target_column: str = "target"
    model_type: str = "lightgbm"


@dataclass(frozen=True)
class ModelEvaluationConfig:
    model_evaluation_dir: Path = MODEL_EVALUATION_DIR
    metric_report_path: Path = MODEL_EVALUATION_DIR / "evaluation_report.json"
    threshold: float = 0.75
    improvement_delta: float = 0.0


@dataclass(frozen=True)
class ModelCalibrationConfig:
    calibration_dir: Path = MODEL_CALIBRATION_DIR
    calibrated_model_path: Path = MODEL_CALIBRATION_DIR / "model.joblib"
    metric_report_path: Path = MODEL_CALIBRATION_DIR / "calibration_report.json"
    calibration_method: str = "sigmoid"
