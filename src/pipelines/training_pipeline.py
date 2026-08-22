import logging
import sys
import warnings
from loguru import logger

from configs.main_config import (
    ingestion_config,
    validation_config,
    transformation_config,
    trainer_config,
    evaluation_config,
    calibration_config,
)
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.components.model_calibration import ModelCalibration

# Mute noisy internal MLflow tracking logs
logging.getLogger("mlflow").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")
warnings.filterwarnings("ignore", category=FutureWarning, module="mlflow")


class TrainingPipeline:
    def __init__(self):
        """Top-level pipeline orchestrator consuming global configs."""
        pass

    def run_pipeline(self):
        try:
            logger.info("Initiating end-to-end training pipeline run")

            # Ingest raw partitions
            ingestion = DataIngestion(config=ingestion_config)
            ingestion_artifact = ingestion.initiate_data_ingestion()

            # Schema and data drift validation gate
            validation = DataValidation(
                config=validation_config,
                data_ingestion_artifact=ingestion_artifact,
            )
            validation_artifact = validation.initiate_data_validation()

            if not validation_artifact.validation_status:
                logger.error("Data validation checks failed — aborting training run")
                sys.exit(1)

            # Feature engineering, anomaly imputation, and encoding
            transformation = DataTransformation(
                config=transformation_config,
                data_validation_artifact=validation_artifact,
            )
            transformation_artifact = transformation.initiate_data_transformation()

            # Cross-validation and out-of-fold probability generation
            trainer = ModelTrainer(
                config=trainer_config,
                data_transformation_artifact=transformation_artifact,
            )
            model_trainer_artifact = trainer.initiate_model_training()

            # Holdout evaluation and registry threshold check
            evaluation = ModelEvaluation(
                config=evaluation_config,
                data_transformation_artifact=transformation_artifact,
                model_trainer_artifact=model_trainer_artifact,
            )
            evaluation_artifact = evaluation.initiate_model_evaluation()

            # Post-hoc probability calibration and production bundle export
            calibration = ModelCalibration(
                config=calibration_config,
                data_transformation_artifact=transformation_artifact,
                model_trainer_artifact=model_trainer_artifact,
                model_evaluation_artifact=evaluation_artifact,
            )
            calibration_artifact = calibration.initiate_model_calibration()

            logger.success(
                "Training pipeline finished successfully | Final production artifact: {}",
                calibration_artifact.calibrated_model_path,
            )

        except Exception as e:
            logger.exception("Training pipeline execution failed: {}", e)
            raise e


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run_pipeline()