from pathlib import Path
from typing import Optional, Union

import joblib
import numpy as np
import pandas as pd
from loguru import logger

from configs.config import drop_cols, target_col
from src.components.data_transformation import DataTransformation
from src.components.data_validation import DataValidation
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelCalibrationArtifact,
)
from src.entity.config_entity import (
    DataTransformationConfig,
    ModelCalibrationConfig,
)


class PredictionPipeline:
    def __init__(
        self,
        data_transformation_artifact: Optional[DataTransformationArtifact] = None,
        model_calibration_artifact: Optional[ModelCalibrationArtifact] = None,
    ):
        self.preprocessor_path = (
            data_transformation_artifact.preprocessor_path
            if data_transformation_artifact
            else DataTransformationConfig().preprocessor_path
        )
        self.model_path = (
            model_calibration_artifact.calibrated_model_path
            if model_calibration_artifact
            else ModelCalibrationConfig().calibrated_model_path
        )

        self.target_col = target_col.lower() if isinstance(target_col, str) else "target"
        self.drop_cols = [c.lower() for c in ([drop_cols] if isinstance(drop_cols, str) else drop_cols)]

        self.validator = DataValidation()
        self.transformer = DataTransformation()
        self.preprocessor = self._load_artifact(self.preprocessor_path, "Preprocessor")
        self.model = self._load_artifact(self.model_path, "Production Model")

    def _load_artifact(self, path: Union[str, Path], name: str):
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(
                f"{name} artifact not found at: {file_path}. "
                "Ensure the training and calibration pipeline has completed successfully."
            )
        logger.info(f"Loading {name} from: {file_path}")
        return joblib.load(file_path)

    def read_input(self, file_path: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
        """Reads input data and standardizes column names to lowercase."""
        if isinstance(file_path, pd.DataFrame):
            df = file_path.copy()
        else:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"Input file path not found: {path}")
            suffix = path.suffix.lower()
            if suffix == ".csv":
                df = pd.read_csv(path)
            elif suffix in [".parquet", ".pq"]:
                df = pd.read_parquet(path)
            else:
                raise ValueError(f"Unsupported file format '{suffix}'. Provide a .csv or .parquet file.")

        df.columns = df.columns.str.strip().str.lower()
        return df

    def validate_input_schema(self, df: pd.DataFrame) -> None:
        """
        Hard Stop Gatekeeper:
        Validates the incoming dataset against ApplicationTestSchema.
        If validation fails, halts execution immediately with a clear error prompt.
        """
        logger.info("Validating inference dataset against ApplicationTestSchema...")
        validation_status, report = self.validator._validate_schema(test_df=df)

        if not validation_status:
            logger.error("Schema validation rejected the uploaded dataset.")
            raise ValueError(
                "Data validation failed. Please check your input data and provide a valid file matching the required schema."
            )

        logger.info("Inference dataset successfully passed schema validation.")

    def transform_features(self, raw_df: pd.DataFrame) -> Union[pd.DataFrame, np.ndarray]:
        """
        Transforms validated raw data into the final numeric feature matrix.
        Executes anomaly cleaning, feature engineering, and ColumnTransformer transformation.
        """
        df = raw_df.copy()
        df.columns = df.columns.str.strip().str.lower()

        # 1. Feature cleaning & engineering
        df = self.transformer._clean_anomalies(df)
        df = self.transformer._engineer_features(df)

        # 2. Drop target column if present in inference data
        if self.target_col in df.columns:
            df = df.drop(columns=[self.target_col])

        # 3. Apply fitted preprocessor (imputation + ordinal encoding)
        X_transformed = self.preprocessor.transform(df)

        # 4. Standardize output to DataFrame if ColumnTransformer returned ndarray
        if isinstance(X_transformed, np.ndarray):
            feature_names = (
                self.preprocessor.get_feature_names_out()
                if hasattr(self.preprocessor, "get_feature_names_out")
                else (self.preprocessor.feature_names_in_ if hasattr(self.preprocessor, "feature_names_in_") else None)
            )
            if feature_names is not None and len(feature_names) == X_transformed.shape[1]:
                cleaned_names = [f.split("__")[-1] for f in feature_names]
                X_transformed = pd.DataFrame(X_transformed, columns=cleaned_names, index=raw_df.index)
            else:
                X_transformed = pd.DataFrame(X_transformed, index=raw_df.index)

        # 5. Drop non-feature identifier columns if present
        if isinstance(X_transformed, pd.DataFrame):
            cols_to_drop = [col for col in self.drop_cols if col in X_transformed.columns]
            if cols_to_drop:
                X_transformed = X_transformed.drop(columns=cols_to_drop)

        return X_transformed

    def predict(self, file_path: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
        """
        Executes end-to-end prediction:
        1. Ingests raw input.
        2. Validates schema strictly (Hard Stop if validation fails).
        3. Transforms features.
        4. Generates calibrated repayment risk scores.
        """
        logger.info("Executing prediction pipeline...")

        raw_df = self.read_input(file_path)
        logger.info(f"Loaded input data with shape: {raw_df.shape}")

        if raw_df.empty:
            raise ValueError("The uploaded dataset contains 0 rows.")

        # --- GATE 1: SCHEMA VALIDATION (Halts execution if invalid) ---
        self.validate_input_schema(raw_df)

        # --- GATE 2: FEATURE TRANSFORMATION ---
        X_inference = self.transform_features(raw_df)

        # --- GATE 3: MODEL INFERENCE ---
        logger.info("Generating calibrated repayment risk scores...")
        probabilities = self.model.predict_proba(X_inference)[:, 1]

        # --- GATE 4: AUGMENT OUTPUT ---
        results_df = raw_df.copy()
        results_df["repayment_risk"] = np.round(probabilities, 5)

        logger.success(f"Inference completed successfully for {len(results_df)} records.")
        return results_df