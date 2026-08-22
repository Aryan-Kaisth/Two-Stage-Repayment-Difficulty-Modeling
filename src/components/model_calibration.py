import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from loguru import logger
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from configs.main_config import drop_cols, target_col
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
        config: ModelCalibrationConfig,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
        model_evaluation_artifact: ModelEvaluationArtifact,
    ):
        self.config = config
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_artifact = model_trainer_artifact
        self.model_evaluation_artifact = model_evaluation_artifact

        self.target_col = target_col.lower() if isinstance(target_col, str) else "target"
        self.drop_cols = [
            c.lower()
            for c in ([drop_cols] if isinstance(drop_cols, str) else drop_cols)
        ]

    def initiate_model_calibration(self) -> ModelCalibrationArtifact:
        logger.info("Starting Model Calibration Component...")
        self.config.calibration_dir.mkdir(parents=True, exist_ok=True)

        # 1. Load the uncalibrated base candidate model via joblib
        base_model_path = Path(self.model_trainer_artifact.trained_model_path)
        logger.info(f"Loading uncalibrated candidate model from: {base_model_path}")
        if not base_model_path.exists():
            raise FileNotFoundError(
                f"Uncalibrated candidate model artifact not found at: {base_model_path}"
            )

        base_model = joblib.load(base_model_path)

        # 2. Load holdout test dataset for calibration fitting & verification
        logger.info("Loading holdout test dataset for probability calibration...")
        test_df = pd.read_parquet(self.data_transformation_artifact.processed_test_path)

        cols_to_drop = [c for c in self.drop_cols if c in test_df.columns]
        if cols_to_drop:
            test_df = test_df.drop(columns=cols_to_drop)

        if self.target_col not in test_df.columns:
            raise ValueError(
                f"Target column '{self.target_col}' missing from test dataset."
            )

        X_test = test_df.drop(columns=[self.target_col])
        y_test = test_df[self.target_col]

        # 3. Calculate uncalibrated baseline metrics
        raw_probs = base_model.predict_proba(X_test)[:, 1]
        original_log_loss = float(log_loss(y_test, raw_probs))
        original_brier_score = float(brier_score_loss(y_test, raw_probs))
        original_roc_auc = float(roc_auc_score(y_test, raw_probs))

        logger.info(
            f"Pre-Calibration Metrics -> LogLoss: {original_log_loss:.5f} | "
            f"Brier Score: {original_brier_score:.5f} | ROC-AUC: {original_roc_auc:.5f}"
        )

        # 4. Fit Probability Calibrator wrapping FrozenEstimator
        logger.info(
            f"Fitting CalibratedClassifierCV using FrozenEstimator and method: [{self.config.calibration_method}]"
        )
        frozen_base = FrozenEstimator(base_model)
        calibrated_model = CalibratedClassifierCV(
            estimator=frozen_base,
            method=self.config.calibration_method,
        )
        calibrated_model.fit(X_test, y_test)

        # 5. Evaluate post-calibration metrics
        cal_probs = calibrated_model.predict_proba(X_test)[:, 1]
        calibrated_log_loss = float(log_loss(y_test, cal_probs))
        calibrated_brier_score = float(brier_score_loss(y_test, cal_probs))
        calibrated_roc_auc = float(roc_auc_score(y_test, cal_probs))

        logger.info(
            f"Post-Calibration Metrics -> LogLoss: {calibrated_log_loss:.5f} | "
            f"Brier Score: {calibrated_brier_score:.5f} | ROC-AUC: {calibrated_roc_auc:.5f}"
        )

        # 6. Save final production model artifact
        calibrated_path = Path(self.config.calibrated_model_path)
        joblib.dump(calibrated_model, calibrated_path)
        logger.success(f"Production calibrated model saved to: {calibrated_path}")

        # 7. Retrieve run_id and log calibration metrics to MLflow
        run_id = None
        metrics_file = Path(self.model_trainer_artifact.metrics_file_path)
        if metrics_file.exists():
            try:
                with open(metrics_file, "r") as f:
                    metadata = json.load(f)
                    run_id = metadata.get("run_id")
            except Exception as e:
                logger.warning(f"Could not read run_id from trainer metadata: {e}")

        if run_id:
            try:
                with mlflow.start_run(run_id=run_id):
                    mlflow.log_metrics({
                        "original_log_loss": original_log_loss,
                        "calibrated_log_loss": calibrated_log_loss,
                        "original_brier_score": original_brier_score,
                        "calibrated_brier_score": calibrated_brier_score,
                        "calibrated_roc_auc": calibrated_roc_auc,
                    })
                    mlflow.log_artifact(str(calibrated_path))
            except Exception as e:
                logger.warning(f"Could not attach calibration metrics to MLflow run {run_id}: {e}")

        # 8. Save calibration audit report
        calibration_report = {
            "is_model_calibrated": True,
            "calibration_method": self.config.calibration_method,
            "original_log_loss": original_log_loss,
            "calibrated_log_loss": calibrated_log_loss,
            "original_brier_score": original_brier_score,
            "calibrated_brier_score": calibrated_brier_score,
            "original_roc_auc": original_roc_auc,
            "calibrated_roc_auc": calibrated_roc_auc,
            "log_loss_delta": round(original_log_loss - calibrated_log_loss, 5),
            "calibrated_model_path": str(calibrated_path),
            "run_id": run_id,
        }

        with open(self.config.metric_report_path, "w") as f:
            json.dump(calibration_report, f, indent=4)

        return ModelCalibrationArtifact(
            is_model_calibrated=True,
            calibrated_model_path=calibrated_path,
            original_log_loss=original_log_loss,
            calibrated_log_loss=calibrated_log_loss,
            metric_report_path=self.config.metric_report_path,
        )