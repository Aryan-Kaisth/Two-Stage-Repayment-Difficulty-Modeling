from pathlib import Path

# Data Ingestion Constants
RAW_DATA_DIR: Path = Path("data/raw")

DATABRICKS_TRAIN_SOURCE_PATH: str = (
    "/Volumes/workspace/default/project_home_credit_data/applications_train.csv"
)
DATABRICKS_TEST_SOURCE_PATH: str = (
    "/Volumes/workspace/default/project_home_credit_data/applications_test.csv"
)

# Data Validation Constants
DATA_VALIDATION_DIR: Path = Path("artifacts/data_validation")

# Data Transformation Constants
DATA_TRANSFORMATION_DIR: Path = Path("artifacts/data_transformation")

# Model Trainer Artifact
MODEL_TRAINER_DIR: Path = Path("artifacts/model_trainer")

# Model Evaluation Constants
MODEL_EVALUATION_DIR: Path = Path("artifacts/model_evaluation")
MODEL_REGISTRY_NAME: str = "HomeCreditDefaultModel"

MODEL_CALIBRATION_DIR: str = Path("artifacts/model_calibration")
