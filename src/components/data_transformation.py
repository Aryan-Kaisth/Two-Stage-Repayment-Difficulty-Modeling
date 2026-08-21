from typing import Optional

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from configs.config import drop_cols, target_col
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact,
)
from src.entity.config_entity import DataTransformationConfig


class DataTransformation:
    def __init__(
        self,
        data_validation_artifact: Optional[DataValidationArtifact] = None,
        config: DataTransformationConfig = DataTransformationConfig(),
    ):
        self.data_validation_artifact = data_validation_artifact
        self.config = config

        self.target_col = (
            target_col.lower()
            if isinstance(target_col, str)
            else "target"
        )

        self.drop_cols = [
            col.lower()
            for col in (
                [drop_cols]
                if isinstance(drop_cols, str)
                else drop_cols
            )
        ]

    def _clean_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(
            "Cleaning anomalies (XNA, 365243, Unknowns)..."
        )

        df = df.copy()

        df = df.replace(
            to_replace=["XNA", "XAP"],
            value=np.nan,
        )

        if "days_employed" in df.columns:
            df["days_employed"] = df["days_employed"].replace(
                365243,
                np.nan,
            )

        if "days_last_phone_change" in df.columns:
            df["days_last_phone_change"] = df[
                "days_last_phone_change"
            ].replace(
                0,
                np.nan,
            )

        if "name_family_status" in df.columns:
            df["name_family_status"] = df[
                "name_family_status"
            ].replace(
                "Unknown",
                np.nan,
            )

        if "name_income_type" in df.columns:
            df["name_income_type"] = df[
                "name_income_type"
            ].replace(
                "Maternity leave",
                np.nan,
            )

        if "region_rating_client_w_city" in df.columns:
            df["region_rating_client_w_city"] = df[
                "region_rating_client_w_city"
            ].replace(
                -1,
                np.nan,
            )

        return df

    def _engineer_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        logger.info("Engineering new features...")

        df = df.copy()

        # Binary features
        df["has_children"] = (
            df["cnt_children"] > 0
        ).astype(np.int8)

        df["has_job"] = (
            df["days_employed"] < 0
        ).astype(np.int8)

        df["long_employment"] = (
            df["days_employed"] < -2000
        ).astype(np.int8)

        df["retirement_age"] = (
            df["days_birth"] < -14000
        ).astype(np.int8)

        # Sums
        df["sum_amt_income_total_amt_annuity"] = (
            df["amt_income_total"]
            + df["amt_annuity"]
        )

        if any(
            col.startswith("amt_req_credit_bureau_")
            for col in df.columns
        ):
            df["total_enquiries_credit_bureau"] = df[
                [
                    col
                    for col in df.columns
                    if col.startswith("amt_req_credit_bureau_")
                ]
            ].sum(axis=1)

        # Differences
        df["diff_amt_credit_amt_goods_price"] = (
            df["amt_credit"]
            - df["amt_goods_price"]
        )

        df["diff_amt_annuity_amt_goods_price"] = (
            df["amt_annuity"]
            - df["amt_goods_price"]
        )

        df["diff_amt_income_total_amt_annuity"] = (
            df["amt_income_total"]
            - df["amt_annuity"]
        )

        df["cnt_adult_fam_member"] = (
            df["cnt_fam_members"]
            - df["cnt_children"]
        )

        df[
            "diff_obs_30_cnt_social_circle_obs_60_cnt_social_circle"
        ] = (
            df["obs_30_cnt_social_circle"]
            - df["obs_60_cnt_social_circle"]
        )

        df[
            "diff_def_30_cnt_social_circle_def_60_cnt_social_circle"
        ] = (
            df["def_30_cnt_social_circle"]
            - df["def_60_cnt_social_circle"]
        )

        # External source features
        ext_cols = [
            "ext_source_1",
            "ext_source_2",
            "ext_source_3",
        ]

        if all(col in df.columns for col in ext_cols):
            df["ext_sources_prod"] = (
                df["ext_source_1"]
                * df["ext_source_2"]
                * df["ext_source_3"]
            )

            df["ext_sources_weighted_sum"] = (
                df["ext_source_3"] * 5
                + df["ext_source_1"] * 3
                + df["ext_source_2"]
            )

            df["ext_sources_weighted_avg"] = (
                df["ext_sources_weighted_sum"] / 3
            )

        # Ratios
        df["ratio_amt_credit_to_amt_annuity"] = (
            df["amt_credit"]
            / df["amt_annuity"]
        )

        df["ratio_amt_credit_to_cnt_adult_fam_member"] = (
            df["amt_credit"]
            / df["cnt_adult_fam_member"]
        )

        df["ratio_amt_income_total_to_amt_annuity"] = (
            df["amt_income_total"]
            / df["amt_annuity"]
        )

        df["amt_income_total_per_adult_fam_member"] = (
            df["amt_income_total"]
            / df["cnt_adult_fam_member"]
        )

        df["ratio_amt_goods_price_to_livingarea_avg"] = (
            df["amt_goods_price"]
            / df["livingarea_avg"]
        )

        df["ratio_amt_goods_price_to_landarea_avg"] = (
            df["amt_goods_price"]
            / df["landarea_avg"]
        )

        df["ratio_amt_goods_price_to_floorsmax_avg_avg"] = (
            df["amt_goods_price"]
            / df["floorsmax_avg"]
        )

        df[
            "ratio_amt_goods_price_to_livingapartments_avg"
        ] = (
            df["amt_goods_price"]
            / df["livingapartments_avg"]
        )

        df["ratio_amt_goods_price_to_years_build_avg"] = (
            df["amt_goods_price"]
            / df["years_build_avg"]
        )

        df["ratio_amt_goods_price_to_days_employed"] = (
            df["amt_goods_price"]
            / df["days_employed"]
        )

        df["ratio_amt_goods_price_to_cnt_children"] = (
            df["amt_goods_price"]
            / df["cnt_children"]
        )

        df[
            "ratio_amt_goods_price_to_sum_amt_income_total_amt_annuity"
        ] = (
            df["amt_goods_price"]
            / df["sum_amt_income_total_amt_annuity"]
        )

        df["ratio_amt_annuity_to_livingarea_avg"] = (
            df["amt_annuity"]
            / df["livingarea_avg"]
        )

        df["ratio_amt_annuity_to_days_employed"] = (
            df["amt_annuity"]
            / df["days_employed"]
        )

        df["ratio_amt_annuity_to_cnt_children"] = (
            df["amt_annuity"]
            / df["cnt_children"]
        )

        df["ratio_amt_annuity_to_cnt_adult_fam_member"] = (
            df["amt_annuity"]
            / df["cnt_adult_fam_member"]
        )

        df[
            "ratio_ext_source_3_to_region_population_relative"
        ] = (
            df["ext_source_3"]
            / df["region_population_relative"]
        )

        df[
            "ratio_days_last_phone_change_to_days_registration"
        ] = (
            df["days_last_phone_change"]
            / df["days_registration"]
        )

        df["pctg_fam_children"] = (
            df["cnt_children"]
            / df["cnt_fam_members"]
        )

        df["ratio_amt_credit_to_amt_goods_price"] = (
            df["amt_credit"]
            / df["amt_goods_price"]
        )

        df["ratio_amt_credit_to_amt_income_total"] = (
            df["amt_credit"]
            / df["amt_income_total"]
        )

        df["ratio_amt_credit_to_cnt_fam_members"] = (
            df["amt_credit"]
            / df["cnt_fam_members"]
        )

        df["ratio_amt_credit_to_cnt_children"] = (
            df["amt_credit"]
            / (1 + df["cnt_children"])
        )

        df["ratio_amt_income_total_to_amt_credit"] = (
            df["amt_income_total"]
            / df["amt_credit"]
        )

        df["ratio_amt_income_total_to_cnt_children"] = (
            df["amt_income_total"]
            / (1 + df["cnt_children"])
        )

        df["ratio_amt_annuity_to_amt_income_total"] = (
            df["amt_annuity"]
            / (1 + df["amt_income_total"])
        )

        df["ratio_children_to_adults"] = (
            df["cnt_children"]
            / df["cnt_adult_fam_member"]
        )

        df["ratio_own_car_age_to_days_birth"] = (
            df["own_car_age"]
            / df["days_birth"]
        )

        df["ratio_own_car_age_to_days_employed"] = (
            df["own_car_age"]
            / df["days_employed"]
        )

        df[
            "ratio_days_last_phone_change_to_days_birth"
        ] = (
            df["days_last_phone_change"]
            / df["days_birth"]
        )

        df[
            "ratio_days_last_phone_change_to_days_employed"
        ] = (
            df["days_last_phone_change"]
            / df["days_employed"]
        )

        df["pctg_days_employed"] = (
            df["days_employed"]
            / df["days_birth"]
        )

        # Enquiry percentages
        if "total_enquiries_credit_bureau" in df.columns:
            df["pctg_enquiries_hour"] = (
                df["amt_req_credit_bureau_hour"]
                / df["total_enquiries_credit_bureau"]
            )

            df["pctg_enquiries_day"] = (
                df["amt_req_credit_bureau_day"]
                / df["total_enquiries_credit_bureau"]
            )

            df["pctg_enquiries_week"] = (
                df["amt_req_credit_bureau_week"]
                / df["total_enquiries_credit_bureau"]
            )

            df["pctg_enquiries_mon"] = (
                df["amt_req_credit_bureau_mon"]
                / df["total_enquiries_credit_bureau"]
            )

            df["pctg_enquiries_qrt"] = (
                df["amt_req_credit_bureau_qrt"]
                / df["total_enquiries_credit_bureau"]
            )

            df["pctg_enquiries_year"] = (
                df["amt_req_credit_bureau_year"]
                / df["total_enquiries_credit_bureau"]
            )

        df = df.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        return df

    def _get_preprocessor(
        self,
        cat_cols: list[str],
    ) -> ColumnTransformer:
        """Build the categorical preprocessing pipeline."""

        cat_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="constant",
                        fill_value="missing",
                    ),
                ),
                (
                    "ordinal_encoder",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                        encoded_missing_value=-1,
                    ),
                ),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "cat",
                    cat_pipeline,
                    cat_cols,
                ),
            ],
            remainder="passthrough",
            verbose_feature_names_out=False,
        )

        preprocessor.set_output(transform="pandas")

        return preprocessor

    def _prepare_model_data(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, Optional[pd.Series]]:
        """Separate the target and remove non-model columns."""

        df = df.copy()

        if self.target_col in df.columns:
            y = df[self.target_col].copy()
            df = df.drop(columns=[self.target_col])
        else:
            y = None

        cols_to_drop = [
            col
            for col in self.drop_cols
            if col in df.columns
        ]

        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        return df, y

    def _ensure_numeric_output(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Ensure the model receives numeric features only."""

        non_numeric_cols = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        if non_numeric_cols:
            raise TypeError(
                "Non-numeric columns remain after preprocessing: "
                f"{non_numeric_cols}"
            )

        return df

    def initiate_data_transformation(
        self,
    ) -> DataTransformationArtifact:
        logger.info(
            "Starting Data Transformation Component"
        )

        self.config.transformation_artifacts_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        train_df = pd.read_csv(
            self.data_validation_artifact.train_file_path
        )

        test_df = pd.read_csv(
            self.data_validation_artifact.test_file_path
        )

        train_df.columns = (
            train_df.columns.str.lower()
        )

        test_df.columns = (
            test_df.columns.str.lower()
        )

        logger.info(
            "Cleaning anomalies in train and test datasets..."
        )

        train_df = self._clean_anomalies(train_df)
        test_df = self._clean_anomalies(test_df)

        logger.info(
            "Engineering features in train and test datasets..."
        )

        train_df = self._engineer_features(train_df)
        test_df = self._engineer_features(test_df)

        X_train, y_train = self._prepare_model_data(
            train_df
        )

        X_test, y_test = self._prepare_model_data(
            test_df
        )

        logger.info(
            "Prepared model inputs | "
            f"X_train: {X_train.shape} | "
            f"X_test: {X_test.shape}"
        )

        # Identify categorical columns from training data only.
        cat_cols = X_train.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        logger.info(
            "Identified {} categorical columns.",
            len(cat_cols),
        )

        preprocessor = self._get_preprocessor(
            cat_cols=cat_cols
        )

        logger.info(
            "Fitting preprocessor on training data..."
        )

        X_train_transformed = (
            preprocessor.fit_transform(X_train)
        )

        logger.info(
            "Transforming test data using fitted preprocessor..."
        )

        X_test_transformed = (
            preprocessor.transform(X_test)
        )

        X_train_transformed = self._ensure_numeric_output(
            X_train_transformed
        )

        X_test_transformed = self._ensure_numeric_output(
            X_test_transformed
        )

        if y_train is not None:
            X_train_transformed[
                self.target_col
            ] = y_train.to_numpy()

        if y_test is not None:
            X_test_transformed[
                self.target_col
            ] = y_test.to_numpy()

        preprocessor_path = (
            self.config.preprocessor_path
        )

        joblib.dump(
            preprocessor,
            preprocessor_path,
        )

        logger.info(
            "Saved fitted preprocessor to: {}",
            preprocessor_path,
        )

        train_out = (
            self.config.processed_train_path
        )

        test_out = (
            self.config.processed_test_path
        )

        X_train_transformed.to_parquet(
            train_out,
            index=False,
        )

        X_test_transformed.to_parquet(
            test_out,
            index=False,
        )

        logger.info(
            "Saved transformed datasets to: {}",
            self.config.transformation_artifacts_dir,
        )

        return DataTransformationArtifact(
            processed_train_path=train_out,
            processed_test_path=test_out,
            preprocessor_path=preprocessor_path,
        )