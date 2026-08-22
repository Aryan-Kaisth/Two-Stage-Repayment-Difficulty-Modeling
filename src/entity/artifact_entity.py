from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionArtifact:
    train_file_path: Path
    test_file_path: Path


@dataclass(frozen=True)
class DataValidationArtifact:
    validation_status: bool
    status_file_path: Path
    report_file_path: Path
    drift_file_path: Path
    train_file_path: Path
    test_file_path: Path


@dataclass(frozen=True)
class DataTransformationArtifact:
    processed_train_path: Path
    processed_test_path: Path
    preprocessor_path: Path


@dataclass(frozen=True)
class ModelTrainerArtifact:
    trained_model_path: str
    metrics_file_path: Path
    metric_value: float
    oof_predictions_path: Path


@dataclass(frozen=True)
class ModelEvaluationArtifact:
    is_model_accepted: bool
    evaluated_model_uri: str
    metric_report_path: Path
    improvement_delta: float


@dataclass(frozen=True)
class ModelCalibrationArtifact:
    is_model_calibrated: bool
    calibrated_model_path: Path
    original_log_loss: float
    calibrated_log_loss: float
    metric_report_path: Path