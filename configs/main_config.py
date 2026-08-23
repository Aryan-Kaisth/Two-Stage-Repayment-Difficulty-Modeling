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
from src.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
    ModelCalibrationConfig,
)

seed = 42

target_col = "target"

te_cols = [
    "name_contract_type",
    "name_type_suite",
    "name_income_type",
    "name_family_status",
    "name_housing_type",
    "occupation_type",
    "weekday_appr_process_start",
    "organization_type",
    "fondkapremont_mode",
    "housetype_mode",
    "wallsmaterial_mode",
]

drop_cols = ["sk_id_curr"]

# Global model parameters
target_col: str = "target"

# Step Configuration Instances
ingestion_config = DataIngestionConfig(
    train_source_path=DATABRICKS_TRAIN_SOURCE_PATH,
    test_source_path=DATABRICKS_TEST_SOURCE_PATH,
    train_file_path=RAW_DATA_DIR / "train.csv",
    test_file_path=RAW_DATA_DIR / "test.csv",
)

validation_config = DataValidationConfig(
    report_dir=DATA_VALIDATION_DIR,
    status_file_path=DATA_VALIDATION_DIR / "status.txt",
    report_file_path=DATA_VALIDATION_DIR / "validation_report.json",
    drift_file_path=DATA_VALIDATION_DIR / "drift_report.json",
)

transformation_config = DataTransformationConfig(
    transformation_artifacts_dir=DATA_TRANSFORMATION_DIR,
    processed_train_path=DATA_TRANSFORMATION_DIR / "train_processed.parquet",
    processed_test_path=DATA_TRANSFORMATION_DIR / "test_processed.parquet",
    preprocessor_path=DATA_TRANSFORMATION_DIR / "preprocessor.joblib",
)

trainer_config = ModelTrainerConfig(
    model_trainer_dir=MODEL_TRAINER_DIR,
    oof_predictions_path=MODEL_TRAINER_DIR / "oof_predictions.parquet",
    trained_model_path=MODEL_TRAINER_DIR / "candidate_model.joblib",
    metadata_path=MODEL_TRAINER_DIR / "metadata.json",
    target_column=target_col,
    model_type="lightgbm",
)

evaluation_config = ModelEvaluationConfig(
    model_evaluation_dir=MODEL_EVALUATION_DIR,
    metric_report_path=MODEL_EVALUATION_DIR / "evaluation_report.json",
    threshold=0.75,
    improvement_delta=0.0,
)

calibration_config = ModelCalibrationConfig(
    calibration_dir=MODEL_CALIBRATION_DIR,
    calibrated_model_path=MODEL_CALIBRATION_DIR / "calibrated_model.joblib",
    metric_report_path=MODEL_CALIBRATION_DIR / "calibration_report.json",
    calibration_method="isotonic",
)
