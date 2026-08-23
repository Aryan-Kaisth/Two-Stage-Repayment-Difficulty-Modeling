from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    train_source_path: str
    test_source_path: str
    train_file_path: Path
    test_file_path: Path


@dataclass(frozen=True)
class DataValidationConfig:
    report_dir: Path
    status_file_path: Path
    report_file_path: Path
    drift_file_path: Path


@dataclass(frozen=True)
class DataTransformationConfig:
    transformation_artifacts_dir: Path
    processed_train_path: Path
    processed_test_path: Path
    preprocessor_path: Path


@dataclass(frozen=True)
class ModelTrainerConfig:
    model_trainer_dir: Path
    oof_predictions_path: Path
    trained_model_path: Path
    metadata_path: Path
    target_column: str
    model_type: str


@dataclass(frozen=True)
class ModelEvaluationConfig:
    model_evaluation_dir: Path
    metric_report_path: Path
    threshold: float
    improvement_delta: float


@dataclass(frozen=True)
class ModelCalibrationConfig:
    calibration_dir: Path
    calibrated_model_path: Path
    metric_report_path: Path
    calibration_method: str
