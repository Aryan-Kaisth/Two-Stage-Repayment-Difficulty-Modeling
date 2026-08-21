import logging
import sys
import warnings
from loguru import logger

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.components.model_calibration import ModelCalibration

# Suppress noisy MLflow internal logs and cloudpickle warnings
logging.getLogger("mlflow").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")
warnings.filterwarnings("ignore", category=FutureWarning, module="mlflow")


class TrainingPipeline:
    def __init__(self, model_name: str = "xgboost"):
        self.model_name = model_name

    def run_pipeline(self):
        try:
            logger.info("Initializing end-to-end MLOps training workflow...")

            # 1. Data Ingestion
            logger.info("Executing data ingestion component.")
            ingestion = DataIngestion()
            ingestion_artifact = ingestion.initiate_data_ingestion()

            # 2. Data Validation
            logger.info("Executing data validation component.")
            validation = DataValidation(
                data_ingestion_artifact=ingestion_artifact
            )
            validation_artifact = validation.initiate_data_validation()

            if not validation_artifact.validation_status:
                logger.error("Data validation failed. Halting execution pipeline.")
                sys.exit(1)
            logger.info("Data validation passed successfully.")

            # 3. Data Transformation
            logger.info("Executing data transformation component.")
            transformation = DataTransformation(
                data_validation_artifact=validation_artifact
            )
            transformation_artifact = transformation.initiate_data_transformation()

            # 4. Model Training (Stratified CV + OOF Generation)
            logger.info(f"Executing model training component for family: [{self.model_name}].")
            trainer = ModelTrainer(
                data_transformation_artifact=transformation_artifact,
                model_name=self.model_name,
            )
            model_trainer_artifact = trainer.initiate_model_training()

            # 5. Model Evaluation & Registry Gate (Holdout Test Gate)
            logger.info("Executing model evaluation and registry promotion gate.")
            evaluation = ModelEvaluation(
                data_transformation_artifact=transformation_artifact,
                model_trainer_artifact=model_trainer_artifact,
            )
            evaluation_artifact = evaluation.initiate_model_evaluation()

            logger.info(f"Registry promotion gate status -> Accepted: {evaluation_artifact.is_model_accepted}")

            # 6. Model Calibration & Production Export (OOF + FrozenEstimator)
            logger.info("Executing model calibration and final production packaging.")
            calibration = ModelCalibration(
                data_transformation_artifact=transformation_artifact,
                model_trainer_artifact=model_trainer_artifact,
                model_evaluation_artifact=evaluation_artifact,
            )
            calibration_artifact = calibration.initiate_model_calibration()

            logger.info(f"Calibration status -> Applied: {calibration_artifact.is_model_calibrated}")
            logger.success(f"Training pipeline completed successfully. Final artifact stored at: {calibration_artifact.calibrated_model_path}")

        except Exception as e:
            logger.exception(f"Pipeline execution failed with critical error: {e}")
            raise e