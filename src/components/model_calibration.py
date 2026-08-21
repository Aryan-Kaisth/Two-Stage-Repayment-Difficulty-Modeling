import json
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import log_loss

from configs.config import drop_cols, target_col
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelCalibrationArtifact,
    ModelEvaluationArtifact,
    ModelTrainerArtifact,
)
from src.entity.config_entity import ModelCalibrationConfig


class ModelCalibration:
    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
        model_evaluation_artifact: ModelEvaluationArtifact,
        config: ModelCalibrationConfig = ModelCalibrationConfig(),
    ):
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_artifact = model_trainer_artifact
        self.model_evaluation_artifact = model_evaluation_artifact
        self.config = config

        self.target_col = target_col.lower() if isinstance(target_col, str) else "target"
        self.drop_cols = [c.lower() for c in ([drop_cols] if isinstance(drop_cols, str) else drop_cols)]

    def initiate_model_calibration(self) -> ModelCalibrationArtifact:
        logger.info("Starting Model Calibration Component (OOF Validation & Frozen Estimator)...")

        # 1. Gate Check from ModelEvaluation
        if not self.model_evaluation_artifact.is_model_accepted:
            logger.warning("Model was rejected in Evaluation. Halting Calibration.")
            return ModelCalibrationArtifact(
                is_model_calibrated=False,
                calibrated_model_path=self.config.calibrated_model_path,
                original_log_loss=0.0,
                calibrated_log_loss=0.0,
                metric_report_path=self.config.metric_report_path,
            )

        self.config.calibration_dir.mkdir(parents=True, exist_ok=True)

        # 2. Load Out-of-Fold (OOF) Predictions generated during 5-Fold Training
        oof_path = Path(self.model_trainer_artifact.oof_predictions_path)
        if not oof_path.exists():
            raise FileNotFoundError(f"OOF predictions file not found at: {oof_path}")

        logger.info(f"Loading OOF predictions from: {oof_path}")
        oof_df = pd.read_parquet(oof_path)

        y_oof = oof_df["target"].to_numpy()
        oof_probs = oof_df["oof_probability"].to_numpy()

        # 3. Compute Uncalibrated OOF Log Loss Baseline
        clipped_oof_probs = np.clip(oof_probs, 1e-15, 1 - 1e-15)
        uncal_logloss = float(log_loss(y_oof, clipped_oof_probs))
        logger.info(f"Uncalibrated OOF Baseline Log Loss: {uncal_logloss:.5f}")

        # 4. Load the Champion Fitted Model from MLflow
        model_uri = self.model_evaluation_artifact.evaluated_model_uri
        logger.info(f"Loading champion estimator from: {model_uri}")
        champion_model = mlflow.sklearn.load_model(model_uri)

        # 5. Load preprocessed training dataset to fit calibration layer
        train_df = pd.read_parquet(self.data_transformation_artifact.processed_train_path)
        cols_to_drop = [c for c in self.drop_cols if c in train_df.columns]
        if cols_to_drop:
            train_df = train_df.drop(columns=cols_to_drop)

        X_train = train_df.drop(columns=[self.target_col])
        y_train = train_df[self.target_col]

        # 6. Fit CalibratedClassifierCV using FrozenEstimator
        logger.info(
            f"Fitting calibration mapping via method='{self.config.calibration_method}' using FrozenEstimator..."
        )
        calibrated_model = CalibratedClassifierCV(
            estimator=FrozenEstimator(champion_model),
            method=self.config.calibration_method,
            cv=None,
        )
        calibrated_model.fit(X_train, y_train)

        # 7. Evaluate Calibrated Log Loss
        calibrated_probs = calibrated_model.predict_proba(X_train)[:, 1]
        cal_logloss = float(log_loss(y_train, calibrated_probs))
        logger.info(f"Calibrated Log Loss: {cal_logloss:.5f}")

        # 8. Decision Logic (Deploy whichever achieves lower log loss)
        is_model_calibrated = False
        decision_reason = ""

        if cal_logloss < uncal_logloss:
            decision_reason = f"Calibration improved Log Loss by {uncal_logloss - cal_logloss:.5f}."
            logger.success(f"{decision_reason} Deploying calibrated model wrapper.")
            final_model_to_save = calibrated_model
            is_model_calibrated = True
        else:
            decision_reason = f"Calibration did not improve Log Loss (Difference: {cal_logloss - uncal_logloss:.5f})."
            logger.warning(f"{decision_reason} Deploying pure uncalibrated model.")
            final_model_to_save = champion_model
            is_model_calibrated = False

        # 9. Persist Winning Model Locally for Deployment
        joblib.dump(final_model_to_save, self.config.calibrated_model_path)
        logger.info(f"Production Model artifact saved to: {self.config.calibrated_model_path}")

        # 10. Log Metrics & Model to MLflow
        with mlflow.start_run(run_name="model_calibration_check"):
            mlflow.log_metric("uncalibrated_oof_log_loss", uncal_logloss)
            mlflow.log_metric("calibrated_log_loss", cal_logloss)
            mlflow.log_param("was_calibration_applied", is_model_calibrated)
            mlflow.log_param("calibration_method", self.config.calibration_method)

            mlflow.sklearn.log_model(
                sk_model=final_model_to_save,
                name="calibrated_production_model",
                serialization_format="cloudpickle",
            )

        # 11. Persist Calibration Audit Report
        calibration_report = {
            "is_model_calibrated": is_model_calibrated,
            "calibration_method": self.config.calibration_method,
            "uncalibrated_oof_log_loss": uncal_logloss,
            "calibrated_log_loss": cal_logloss,
            "decision_reason": decision_reason,
            "model_path": str(self.config.calibrated_model_path),
        }

        with open(self.config.metric_report_path, "w") as f:
            json.dump(calibration_report, f, indent=4)

        return ModelCalibrationArtifact(
            is_model_calibrated=is_model_calibrated,
            calibrated_model_path=self.config.calibrated_model_path,
            original_log_loss=uncal_logloss,
            calibrated_log_loss=cal_logloss,
            metric_report_path=self.config.metric_report_path,
        )