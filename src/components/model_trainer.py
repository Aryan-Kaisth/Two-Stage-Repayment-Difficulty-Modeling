import json
import logging
import time
import warnings
from typing import Tuple

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from configs.main_config import drop_cols, seed, target_col
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from src.entity.config_entity import ModelTrainerConfig
from src.model_factory import MODEL_FACTORY

logging.getLogger("mlflow").setLevel(logging.ERROR)

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="mlflow",
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="mlflow",
)


class ModelTrainer:
    def __init__(
        self,
        config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact,
        n_splits: int = 5,
    ):
        self.config = config
        self.data_transformation_artifact = data_transformation_artifact
        self.model_name = self.config.model_type
        self.n_splits = n_splits
        self.seed = seed

        self.target_col = (
            target_col.lower() if isinstance(target_col, str) else "target"
        )
        self.drop_cols = [
            col.lower()
            for col in (
                [drop_cols] if isinstance(drop_cols, str) else drop_cols
            )
        ]

    def _load_training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load the transformed training dataset only."""
        logger.info("Loading processed training dataset...")

        train_df = pd.read_parquet(
            self.data_transformation_artifact.processed_train_path
        )

        if self.target_col not in train_df.columns:
            raise ValueError(
                f"Target column '{self.target_col}' was not found in the processed training data."
            )

        cols_to_drop = [
            col for col in self.drop_cols if col in train_df.columns
        ]

        if cols_to_drop:
            train_df = train_df.drop(columns=cols_to_drop)

        X = train_df.drop(columns=[self.target_col])
        y = train_df[self.target_col]

        logger.info(
            "Training data loaded | X: {} | y: {}",
            X.shape,
            y.shape,
        )

        return X, y

    def _train_fold(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
    ) -> np.ndarray:
        """Train one fresh model and return validation probabilities."""
        model = MODEL_FACTORY[self.model_name]()
        model.fit(X_train, y_train)
        return model.predict_proba(X_valid)[:, 1]

    def initiate_model_training(self) -> ModelTrainerArtifact:
        logger.info(
            "Starting Model Trainer Component | Model: {}",
            self.model_name,
        )

        self.config.model_trainer_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Deterministic file paths sourced directly from Config
        metadata_path = self.config.metadata_path
        oof_predictions_path = self.config.oof_predictions_path
        uncalibrated_model_path = self.config.trained_model_path

        # Load training features and labels
        X, y = self._load_training_data()

        logger.info(
            "Starting {}-fold stratified cross-validation...",
            self.n_splits,
        )

        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.seed,
        )

        oof_probabilities = np.zeros(
            len(X),
            dtype=np.float64,
        )
        fold_auc_scores: list[float] = []
        fold_loss_scores: list[float] = []

        with mlflow.start_run(run_name=self.model_name) as run:
            run_id = run.info.run_id

            mlflow.set_tags(
                {
                    "model_family": self.model_name,
                    "training_mode": "stratified_cv",
                }
            )

            mlflow.log_params(
                {
                    "n_splits": self.n_splits,
                    "random_state": self.seed,
                }
            )

            logger.info("Executing cross-validation folds...")

            for fold, (train_indices, valid_indices) in enumerate(
                skf.split(X, y),
                start=1,
            ):
                X_train_fold = X.iloc[train_indices]
                y_train_fold = y.iloc[train_indices]
                X_valid_fold = X.iloc[valid_indices]
                y_valid_fold = y.iloc[valid_indices]

                fold_probabilities = self._train_fold(
                    X_train=X_train_fold,
                    y_train=y_train_fold,
                    X_valid=X_valid_fold,
                )

                # Store OOF predictions in original row positions
                oof_probabilities[valid_indices] = fold_probabilities

                fold_auc = float(roc_auc_score(y_valid_fold, fold_probabilities))
                fold_loss = float(log_loss(y_valid_fold, fold_probabilities))

                fold_auc_scores.append(fold_auc)
                fold_loss_scores.append(fold_loss)

                logger.debug(
                    "Fold {}/{} | ROC-AUC: {:.5f} | LogLoss: {:.5f}",
                    fold,
                    self.n_splits,
                    fold_auc,
                    fold_loss,
                )

            # Aggregate cross-validation metrics
            cv_metrics = {
                "oof_roc_auc": float(roc_auc_score(y, oof_probabilities)),
                "oof_logloss": float(log_loss(y, oof_probabilities)),
                "std_roc_auc": float(np.std(fold_auc_scores)),
                "std_logloss": float(np.std(fold_loss_scores)),
            }

            logger.info(
                "OOF Metrics | ROC-AUC: {:.5f} (±{:.5f}) | LogLoss: {:.5f} (±{:.5f})",
                cv_metrics["oof_roc_auc"],
                cv_metrics["std_roc_auc"],
                cv_metrics["oof_logloss"],
                cv_metrics["std_logloss"],
            )

            mlflow.log_metrics(cv_metrics)

            # Save deterministic OOF predictions
            oof_df = pd.DataFrame(
                {
                    "target": y.to_numpy(),
                    "oof_probability": oof_probabilities,
                }
            )
            oof_df.to_parquet(
                oof_predictions_path,
                index=False,
            )
            mlflow.log_artifact(str(oof_predictions_path))

            # Fit the final candidate on ALL training data
            logger.info("Fitting final candidate model on all training data...")
            final_model = MODEL_FACTORY[self.model_name]()
            start_time = time.time()
            final_model.fit(X, y)

            logger.info(
                "Final model fitted in {:.2f}s.",
                time.time() - start_time,
            )

            # Save uncalibrated model locally
            joblib.dump(final_model, uncalibrated_model_path)
            logger.info(
                "Saved local uncalibrated model to: {}",
                uncalibrated_model_path,
            )

            # Log parameters & model to MLflow
            model_params = final_model.get_params()
            mlflow.log_params(model_params)

            mlflow.log_dict(
                {"feature_names": X.columns.tolist()},
                "feature_names.json",
            )

            mlflow.sklearn.log_model(
                sk_model=final_model,
                name="model",
                serialization_format="cloudpickle",
            )

            # Save deterministic metadata JSON
            model_metadata = {
                "run_id": run_id,
                "model_family": self.model_name,
                "n_splits": self.n_splits,
                "random_state": self.seed,
                "parameters": model_params,
                "metrics": cv_metrics,
                "uncalibrated_model_path": str(uncalibrated_model_path),
                "oof_predictions_path": str(oof_predictions_path),
            }

            with open(metadata_path, "w") as file:
                json.dump(
                    model_metadata,
                    file,
                    indent=4,
                )

            mlflow.log_artifact(str(metadata_path))
            mlflow.log_artifact(str(uncalibrated_model_path))

            logger.success(
                "Candidate model training completed and logged under run ID: {}",
                run_id,
            )

        return ModelTrainerArtifact(
            trained_model_path=str(uncalibrated_model_path),
            metrics_file_path=metadata_path,
            metric_value=cv_metrics["oof_roc_auc"],
            oof_predictions_path=oof_predictions_path,
        )