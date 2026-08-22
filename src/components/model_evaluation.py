import json
from pathlib import Path
from typing import Optional

import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import log_loss, precision_score, recall_score, roc_auc_score

from configs.main_config import drop_cols, target_col
from src.constants import MODEL_REGISTRY_NAME
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelEvaluationArtifact,
    ModelTrainerArtifact,
)
from src.entity.config_entity import ModelEvaluationConfig


class ModelEvaluation:
    def __init__(
        self,
        config: ModelEvaluationConfig,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
    ):
        self.config = config
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_artifact = model_trainer_artifact

        self.target_col = target_col.lower() if isinstance(target_col, str) else "target"
        self.drop_cols = [c.lower() for c in ([drop_cols] if isinstance(drop_cols, str) else drop_cols)]
        self.client = MlflowClient()

    def _get_registered_champion_score(self) -> Optional[float]:
        """Fetches the test ROC-AUC score of the current registered champion in MLflow."""
        try:
            registered_models = self.client.search_model_versions(f"name='{MODEL_REGISTRY_NAME}'")
            if not registered_models:
                logger.info("No champion model found in registry (initial deployment run).")
                return None

            latest_version = sorted(registered_models, key=lambda x: int(x.version), reverse=True)[0]
            run = self.client.get_run(latest_version.run_id)
            champion_score = run.data.metrics.get("test_roc_auc")

            if champion_score is not None:
                logger.info(
                    f"Current Champion Version: v{latest_version.version} | "
                    f"Test ROC-AUC: {champion_score:.5f}"
                )
                return float(champion_score)

        except Exception as e:
            logger.warning(f"Could not retrieve registered champion score: {e}")

        return None

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        logger.info("Starting Model Evaluation Component (Holdout Test Gate)...")
        self.config.model_evaluation_dir.mkdir(parents=True, exist_ok=True)

        # 1. Load the holdout test dataset (already transformed by preprocessor)
        logger.info("Loading holdout test dataset for unbiased evaluation...")
        test_df = pd.read_parquet(self.data_transformation_artifact.processed_test_path)

        cols_to_drop = [c for c in self.drop_cols if c in test_df.columns]
        if cols_to_drop:
            test_df = test_df.drop(columns=cols_to_drop)

        if self.target_col not in test_df.columns:
            raise ValueError(f"Target column '{self.target_col}' missing from test dataset.")

        X_test = test_df.drop(columns=[self.target_col])
        y_test = test_df[self.target_col]

        logger.info(f"Test Holdout shape: {X_test.shape}")

        # 2. Load candidate model directly using joblib from local artifact path
        model_path = Path(self.model_trainer_artifact.trained_model_path)
        logger.info(f"Loading candidate model from: {model_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"Candidate model artifact not found at: {model_path}")

        model = joblib.load(model_path)

        # 3. Read metadata to retrieve MLflow run_id
        run_id = None
        metrics_file = Path(self.model_trainer_artifact.metrics_file_path)
        if metrics_file.exists():
            try:
                with open(metrics_file, "r") as f:
                    metadata = json.load(f)
                    run_id = metadata.get("run_id")
            except Exception as e:
                logger.warning(f"Could not read run_id from metadata: {e}")

        # 4. Evaluate challenger performance on holdout test set
        test_probs = model.predict_proba(X_test)[:, 1]
        test_preds = (test_probs >= 0.5).astype(np.int32)

        challenger_test_roc_auc = float(roc_auc_score(y_test, test_probs))
        challenger_test_logloss = float(log_loss(y_test, test_probs))
        challenger_test_precision = float(precision_score(y_test, test_preds, zero_division=0))
        challenger_test_recall = float(recall_score(y_test, test_preds, zero_division=0))

        logger.info(
            f"Challenger Test Metrics -> ROC-AUC: {challenger_test_roc_auc:.5f} | "
            f"LogLoss: {challenger_test_logloss:.5f} | Precision: {challenger_test_precision:.5f} | "
            f"Recall: {challenger_test_recall:.5f}"
        )

        # 5. Log test metrics back to the active training run in MLflow
        if run_id:
            try:
                with mlflow.start_run(run_id=run_id):
                    mlflow.log_metrics({
                        "test_roc_auc": challenger_test_roc_auc,
                        "test_logloss": challenger_test_logloss,
                        "test_precision": challenger_test_precision,
                        "test_recall": challenger_test_recall,
                    })
            except Exception as e:
                logger.warning(f"Could not attach test metrics to MLflow run {run_id}: {e}")

        # 6. Champion vs. Challenger Decision Logic
        is_model_accepted = False
        decision_reason = ""
        champion_score = self._get_registered_champion_score()

        # Gate 1: Absolute Performance Baseline
        if challenger_test_roc_auc < self.config.threshold:
            decision_reason = (
                f"Challenger rejected: Test ROC-AUC ({challenger_test_roc_auc:.5f}) is below "
                f"the absolute threshold ({self.config.threshold:.5f})."
            )
            logger.warning(decision_reason)

        # Gate 2: Initial Deployment Approval
        elif champion_score is None:
            is_model_accepted = True
            decision_reason = "No existing champion in registry. Challenger approved as initial champion."
            logger.info(decision_reason)

        # Gate 3: Relative Improvement Over Champion
        else:
            required_score = champion_score + self.config.improvement_delta
            if challenger_test_roc_auc >= required_score:
                is_model_accepted = True
                decision_reason = (
                    f"Challenger accepted: Outperformed champion ({challenger_test_roc_auc:.5f} >= "
                    f"required {required_score:.5f} [Champion: {champion_score:.5f} + Delta: {self.config.improvement_delta:.5f}])."
                )
                logger.info(decision_reason)
            else:
                decision_reason = (
                    f"Challenger rejected: Did not outperform champion ({challenger_test_roc_auc:.5f} < "
                    f"required {required_score:.5f})."
                )
                logger.warning(decision_reason)

        # 7. Promotion to MLflow Model Registry
        if is_model_accepted and run_id:
            try:
                mlflow_model_uri = f"runs:/{run_id}/model"
                mlflow.register_model(model_uri=mlflow_model_uri, name=MODEL_REGISTRY_NAME)
                logger.success(f"Model successfully promoted to Registry: '{MODEL_REGISTRY_NAME}'")
            except Exception as e:
                logger.warning(f"Could not register model in MLflow Registry: {e}")

        # 8. Persist Evaluation Audit Report
        evaluation_report = {
            "is_model_accepted": is_model_accepted,
            "cv_roc_auc": self.model_trainer_artifact.metric_value,
            "challenger_test_roc_auc": challenger_test_roc_auc,
            "challenger_test_logloss": challenger_test_logloss,
            "challenger_test_precision": challenger_test_precision,
            "challenger_test_recall": challenger_test_recall,
            "champion_test_roc_auc": champion_score,
            "absolute_threshold": self.config.threshold,
            "improvement_delta": self.config.improvement_delta,
            "decision_reason": decision_reason,
            "evaluated_model_path": str(model_path),
            "run_id": run_id,
        }

        with open(self.config.metric_report_path, "w") as f:
            json.dump(evaluation_report, f, indent=4)

        return ModelEvaluationArtifact(
            is_model_accepted=is_model_accepted,
            evaluated_model_uri=f"runs:/{run_id}/model" if run_id else str(model_path),
            metric_report_path=self.config.metric_report_path,
            improvement_delta=self.config.improvement_delta,
        )