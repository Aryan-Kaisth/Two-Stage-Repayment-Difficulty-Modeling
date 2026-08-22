from pathlib import Path
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

from evidently import Report
from evidently.presets import DataDriftPreset
from evidently.metrics import ValueDrift

from src.constants import SAMPLE_DATA_PATH

from src.plot import (
    plot_shap_waterfall,
    plot_shap_decision,
    plot_shap_beeswarm,
    plot_calibration_curve,
)

from configs.main_config import (
    ingestion_config,
    transformation_config,
    trainer_config,
    target_col,
)
from src.pipelines.prediction_pipeline import PredictionPipeline


st.set_page_config(
    page_title="Home Credit Risk Prediction & Monitoring",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- Premium Styling & UI Theme Customization ---
st.html(
    """
    <style>
    /* ------------------------------
       GLOBAL APPLICATION
    ------------------------------ */
    :root {
        --hc-navy: #0b1220;
        --hc-slate: #111827;
        --hc-blue: #3b82f6;
        --hc-sky: #38bdf8;
        --hc-violet: #8b5cf6;
        --hc-emerald: #10b981;
        --hc-amber: #f59e0b;
        --hc-red: #ef4444;
        --hc-border: rgba(148, 163, 184, 0.18);
        --hc-muted: #94a3b8;
        --hc-text: #e2e8f0;
        --hc-panel: rgba(15, 23, 42, 0.72);
    }

    .stApp {
        background:
            radial-gradient(
                circle at 10% 0%,
                rgba(59, 130, 246, 0.08),
                transparent 28%
            ),
            radial-gradient(
                circle at 90% 5%,
                rgba(139, 92, 246, 0.07),
                transparent 26%
            ),
            #020617;
    }

    [data-testid="stAppViewContainer"] {
        background: transparent;
    }

    [data-testid="stHeader"] {
        background: rgba(2, 6, 23, 0.72);
    }

    [data-testid="stToolbar"] {
        right: 1rem;
    }

    .main .block-container {
        max-width: 1500px;
        padding-top: 1.6rem;
        padding-bottom: 4rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    /* ------------------------------
       HERO
    ------------------------------ */
    .hero-container {
        position: relative;
        overflow: hidden;
        background:
            radial-gradient(
                circle at 82% 30%,
                rgba(56, 189, 248, 0.18),
                transparent 22%
            ),
            radial-gradient(
                circle at 65% 90%,
                rgba(139, 92, 246, 0.18),
                transparent 26%
            ),
            linear-gradient(
                135deg,
                #07111f 0%,
                #0f1c33 48%,
                #172554 100%
            );
        padding: 30px 34px;
        border-radius: 20px;
        color: #ffffff;
        margin-bottom: 25px;
        border: 1px solid rgba(148, 163, 184, 0.18);
        box-shadow:
            0 20px 55px rgba(2, 6, 23, 0.45),
            inset 0 1px 0 rgba(255, 255, 255, 0.06);
    }

    .hero-container::before {
        content: "";
        position: absolute;
        width: 240px;
        height: 240px;
        right: -70px;
        top: -80px;
        border-radius: 50%;
        background: rgba(56, 189, 248, 0.08);
        filter: blur(10px);
    }

    .hero-kicker {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 5px 10px;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.08);
        color: #bae6fd;
        font-size: 0.74rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.9px;
        margin-bottom: 12px;
    }

    .hero-title {
        position: relative;
        font-size: 2.35rem;
        line-height: 1.1;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.9px;
        background: linear-gradient(
            90deg,
            #f8fafc 0%,
            #bfdbfe 42%,
            #67e8f9 72%,
            #c4b5fd 100%
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        position: relative;
        max-width: 860px;
        font-size: 0.97rem;
        line-height: 1.65;
        color: #cbd5e1;
        margin-top: 10px;
        margin-bottom: 0;
    }

    .hero-badges {
        position: relative;
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 19px;
    }

    .hero-badge {
        padding: 5px 10px;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 600;
        color: #dbeafe;
        background: rgba(59, 130, 246, 0.12);
        border: 1px solid rgba(96, 165, 250, 0.19);
    }

    /* ------------------------------
       SECTION HEADERS
    ------------------------------ */
    .section-wrap {
        margin-top: 1.35rem;
        margin-bottom: 0.65rem;
    }

    .section-kicker {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        font-size: 0.72rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #94a3b8;
        margin-bottom: 4px;
    }

    .section-title {
        color: #f8fafc;
        font-size: 1.42rem;
        font-weight: 750;
        letter-spacing: -0.3px;
        margin: 0;
    }

    .section-description {
        color: #94a3b8;
        font-size: 0.84rem;
        margin-top: 5px;
        line-height: 1.55;
    }

    .section-line {
        width: 72px;
        height: 3px;
        border-radius: 999px;
        margin-top: 9px;
        background: linear-gradient(
            90deg,
            #38bdf8,
            #8b5cf6
        );
    }

    /* ------------------------------
       SOURCE / STATUS CARDS
    ------------------------------ */
    .source-card {
        background: linear-gradient(
            145deg,
            rgba(15, 23, 42, 0.9),
            rgba(15, 23, 42, 0.68)
        );
        border: 1px solid var(--hc-border);
        border-radius: 16px;
        padding: 18px 20px;
        box-shadow:
            0 12px 30px rgba(2, 6, 23, 0.22),
            inset 0 1px 0 rgba(255, 255, 255, 0.03);
    }

    .source-label {
        font-size: 0.71rem;
        text-transform: uppercase;
        letter-spacing: 0.85px;
        font-weight: 800;
        color: #64748b;
        margin-bottom: 7px;
    }

    .source-value {
        font-size: 1rem;
        color: #f8fafc;
        font-weight: 700;
    }

    .source-meta {
        margin-top: 5px;
        color: #94a3b8;
        font-size: 0.76rem;
    }

    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 7px;
        box-shadow: 0 0 12px currentColor;
    }

    .status-green {
        color: #34d399;
        background: #34d399;
    }

    .status-blue {
        color: #60a5fa;
        background: #60a5fa;
    }

    .status-amber {
        color: #fbbf24;
        background: #fbbf24;
    }

    /* ------------------------------
       METRIC CARDS
    ------------------------------ */
    .metric-card {
        position: relative;
        overflow: hidden;
        min-height: 112px;
        padding: 17px 18px 16px 18px;
        border-radius: 16px;
        background:
            linear-gradient(
                145deg,
                rgba(15, 23, 42, 0.94),
                rgba(15, 23, 42, 0.70)
            );
        border: 1px solid var(--hc-border);
        box-shadow:
            0 12px 26px rgba(2, 6, 23, 0.18),
            inset 0 1px 0 rgba(255, 255, 255, 0.025);
    }

    .metric-card::after {
        content: "";
        position: absolute;
        width: 90px;
        height: 90px;
        right: -35px;
        top: -35px;
        border-radius: 50%;
        background: var(--metric-glow, rgba(59, 130, 246, 0.08));
        filter: blur(5px);
    }

    .metric-label {
        position: relative;
        z-index: 1;
        color: #94a3b8;
        font-size: 0.73rem;
        text-transform: uppercase;
        letter-spacing: 0.65px;
        font-weight: 700;
    }

    .metric-value {
        position: relative;
        z-index: 1;
        margin-top: 8px;
        color: #f8fafc;
        font-size: 1.55rem;
        line-height: 1;
        font-weight: 800;
        letter-spacing: -0.5px;
    }

    .metric-accent {
        position: relative;
        z-index: 1;
        width: 34px;
        height: 3px;
        margin-top: 12px;
        border-radius: 999px;
        background: var(--metric-accent, #60a5fa);
    }

    /* ------------------------------
       RISK CARDS
    ------------------------------ */
    .risk-summary {
        background:
            linear-gradient(
                145deg,
                rgba(15, 23, 42, 0.97),
                rgba(17, 24, 39, 0.82)
            );
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 18px;
        padding: 20px 22px;
        box-shadow:
            0 15px 38px rgba(2, 6, 23, 0.25),
            inset 0 1px 0 rgba(255, 255, 255, 0.03);
    }

    .risk-label {
        color: #94a3b8;
        font-size: 0.73rem;
        text-transform: uppercase;
        letter-spacing: 0.75px;
        font-weight: 800;
    }

    .risk-score {
        font-size: 2.1rem;
        line-height: 1;
        font-weight: 850;
        color: #f8fafc;
        margin-top: 8px;
    }

    .risk-caption {
        color: #64748b;
        font-size: 0.76rem;
        margin-top: 7px;
    }

    .risk-pill {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        padding: 6px 12px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 750;
        margin-top: 10px;
    }

    .pill-low {
        background: rgba(16, 185, 129, 0.12);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.28);
    }

    .pill-med {
        background: rgba(245, 158, 11, 0.12);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.28);
    }

    .pill-high {
        background: rgba(239, 68, 68, 0.12);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.28);
    }

    /* ------------------------------
       SELECTOR / CONTENT CARDS
    ------------------------------ */
    .content-card {
        background: rgba(15, 23, 42, 0.64);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 16px;
        padding: 16px 18px;
    }

    .mini-heading {
        color: #e2e8f0;
        font-size: 0.93rem;
        font-weight: 750;
        margin-bottom: 4px;
    }

    .mini-caption {
        color: #64748b;
        font-size: 0.76rem;
        line-height: 1.5;
    }

    /* ------------------------------
       STREAMLIT COMPONENT POLISH
    ------------------------------ */
    div[data-testid="stFileUploader"] {
        background: rgba(15, 23, 42, 0.48);
        border: 1px dashed rgba(96, 165, 250, 0.28);
        border-radius: 16px;
        padding: 8px;
    }

    div[data-testid="stFileUploaderDropzone"] {
        background: rgba(15, 23, 42, 0.32);
        border-radius: 12px;
    }

    div[data-testid="stButton"] > button {
        border-radius: 11px;
        font-weight: 700;
        min-height: 43px;
        padding-left: 1.1rem;
        padding-right: 1.1rem;
        transition:
            transform 0.15s ease,
            box-shadow 0.15s ease,
            border-color 0.15s ease;
    }

    div[data-testid="stButton"] > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 9px 22px rgba(59, 130, 246, 0.18);
    }

    div[data-testid="stDownloadButton"] > button {
        border-radius: 10px;
        font-weight: 700;
    }

    div[data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.38);
        border: 1px solid rgba(148, 163, 184, 0.11);
        border-radius: 14px;
        padding: 12px 14px;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;
    }

    button[data-baseweb="tab"] {
        font-weight: 700;
    }

    hr {
        border-color: rgba(148, 163, 184, 0.12);
        margin-top: 1.4rem;
        margin-bottom: 1.4rem;
    }

    .stAlert {
        border-radius: 12px;
    }

    /* Keep radio area compact and clean */
    div[role="radiogroup"] {
        gap: 0.35rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                rgba(7, 17, 31, 0.98),
                rgba(2, 6, 23, 0.98)
            );
        border-right: 1px solid rgba(148, 163, 184, 0.11);
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1.4rem;
    }

    /* Mobile adjustments */
    @media (max-width: 900px) {
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
            padding-top: 1rem;
        }

        .hero-title {
            font-size: 1.75rem;
        }

        .hero-container {
            padding: 23px 22px;
            border-radius: 17px;
        }
    }
    </style>
    """
)


@st.cache_resource
def load_prediction_pipeline():
    """Initializes and caches the prediction pipeline in memory."""
    return PredictionPipeline()


def get_base_tree_estimator(model):
    """
    Safely retrieves the underlying tree booster whether the model
    is a pure LightGBM estimator, wrapped in CalibratedClassifierCV,
    or wrapped in FrozenEstimator.
    """
    if (
        hasattr(model, "calibrated_classifiers_")
        and len(model.calibrated_classifiers_) > 0
    ):
        est = model.calibrated_classifiers_[0].estimator
        if hasattr(est, "estimator"):  # Unwraps FrozenEstimator
            return est.estimator
        return est

    if hasattr(model, "estimator"):
        est = model.estimator
        if hasattr(est, "estimator"):
            return est.estimator
        return est

    if hasattr(model, "named_steps"):
        return list(model.named_steps.values())[-1]

    return model


@st.cache_data
def get_raw_training_reference():
    """Loads baseline raw training data directly from ingestion_config.train_file_path."""
    train_raw_path = Path(ingestion_config.train_file_path)

    if train_raw_path.exists():
        if train_raw_path.suffix == ".parquet":
            return pd.read_parquet(train_raw_path)

        return pd.read_csv(train_raw_path)

    return None


@st.cache_data
def get_oof_predictions_reference():
    """Loads Out-of-Fold (OOF) baseline predictions directly from trainer_config.oof_predictions_path."""
    oof_path = Path(trainer_config.oof_predictions_path)

    if oof_path.exists():
        if oof_path.suffix == ".parquet":
            return pd.read_parquet(oof_path)

        return pd.read_csv(oof_path)

    return None


@st.cache_data
def get_test_calibration_data():
    """Loads holdout test data and scores it directly via the production model."""
    test_path = Path(transformation_config.processed_test_path)

    if not test_path.exists():
        test_path = Path(transformation_config.processed_train_path)

    if not test_path.exists():
        return None, None

    test_df = pd.read_parquet(test_path)
    actual_target = target_col.lower() if isinstance(target_col, str) else "target"

    target_matches = [
        c for c in test_df.columns if c.lower() == actual_target
    ]

    if not target_matches:
        return None, None

    target_name = target_matches[0]
    y_true = test_df[target_name].values
    pipeline = load_prediction_pipeline()

    cols_to_drop = list(
        set(
            [c for c in pipeline.drop_cols if c in test_df.columns]
            + [target_name]
        )
    )

    X_test = test_df.drop(columns=cols_to_drop)

    y_prob = pipeline.model.predict_proba(X_test)[:, 1]

    return y_true, y_prob


def render_evidently_html(eval_snapshot, default_height=800):
    """Helper to safely export and render modern Evidently report HTML in Streamlit."""
    with tempfile.NamedTemporaryFile(
        suffix=".html",
        delete=False,
        mode="w+",
        encoding="utf-8",
    ) as tmp_file:
        eval_snapshot.save_html(tmp_file.name)
        report_path = Path(tmp_file.name)

    st.iframe(report_path, height=default_height)


def render_section_header(kicker, title, description=None):
    """Renders a consistent visual section heading."""
    description_html = (
        f'<div class="section-description">{description}</div>'
        if description
        else ""
    )

    st.markdown(
        f"""
        <div class="section-wrap">
            <div class="section-kicker">✦ {kicker}</div>
            <div class="section-title">{title}</div>
            {description_html}
            <div class="section-line"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label, value, accent, glow):
    """Renders a custom dashboard metric card."""
    st.markdown(
        f"""
        <div
            class="metric-card"
            style="--metric-accent:{accent}; --metric-glow:{glow};"
        >
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-accent"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --- Sidebar ---
with st.sidebar:
    st.markdown(
        """
        <div style="
            padding: 4px 2px 18px 2px;
        ">
            <div style="
                font-size:0.72rem;
                letter-spacing:1px;
                text-transform:uppercase;
                color:#64748b;
                font-weight:800;
                margin-bottom:7px;
            ">
                Home Credit
            </div>
            <div style="
                font-size:1.08rem;
                color:#f8fafc;
                font-weight:800;
            ">
                Repayment Difficulty Prediction
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown(
        """
        <div style="
            font-size:0.75rem;
            color:#94a3b8;
            line-height:1.7;
        ">
            <b style="color:#e2e8f0;">Prediction</b><br>
            Two-stage repayment difficulty prediction
            <br><br>
            <b style="color:#e2e8f0;">Calibration</b><br>
            Post-hoc probability calibration
            <br><br>
            <b style="color:#e2e8f0;">Explainability</b><br>
            SHAP-based applicant explanations
            <br><br>
            <b style="color:#e2e8f0;">Monitoring</b><br>
            Prediction and feature drift monitoring
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.caption("Model environment")
    st.markdown(
        ":green-badge[● Production Ready]",
    )
    st.caption(
        "Reference baselines are loaded from the configured training and OOF artifacts."
    )


# --- Polished Header Banner ---
st.html(
    """
    <div class="hero-container">
        <div class="hero-kicker">
            🏦 Repayment Risk
        </div>

        <div class="hero-title">
            Two-Stage Repayment Difficulty Prediction
        </div>

        <div class="hero-subtitle">
            Explainable two-stage machine learning system with post-hoc probability 
            calibration for Home Credit repayment difficulty prediction.
        </div>
    </div>
    """
)


# --- Data Acquisition ---
render_section_header(
    "Acquisition",
    "Data Intake & Assessment",
    "Upload an unseen applicant batch or use the built-in sample dataset for a ready-to-run demonstration.",
)

source_col1, source_col2 = st.columns(
    [1.65, 1],
    gap="large",
)

with source_col1:
    st.markdown(
        """
        <div class="source-card">
            <div class="source-label">Input source</div>
            <div class="source-value">
                Select how this scoring session should begin
            </div>
            <div class="source-meta">
                Both paths feed the same validation, transformation,
                inference, explainability, and monitoring pipeline.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    input_source = st.radio(
        "Choose how you want to provide applicant data:",
        options=["Upload a file", "Use sample data"],
        horizontal=True,
        index=0,
        label_visibility="collapsed",
    )

with source_col2:
    current_source_label = (
        "Uploaded applicant batch"
        if input_source == "Upload a file"
        else "Built-in sample dataset"
    )

    current_source_meta = (
        "CSV / Parquet"
        if input_source == "Upload a file"
        else "assets/sample_unseen.csv"
    )

    current_dot = (
        "status-blue"
        if input_source == "Upload a file"
        else "status-green"
    )

    st.markdown(
        f"""
        <div class="source-card">
            <div class="source-label">Current mode</div>
            <div class="source-value">
                <span class="status-dot {current_dot}"></span>
                {current_source_label}
            </div>
            <div class="source-meta">
                {current_source_meta}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


uploaded_file = None

if input_source == "Upload a file":
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose a CSV or Parquet file",
        type=["csv", "parquet"],
        help="Upload application data matching the training feature schema.",
        accept_multiple_files=False,
    )

if input_source == "Use sample data":
    sample_file_path = SAMPLE_DATA_PATH

    if sample_file_path.exists():
        st.info(
            "Using the sample applicant dataset (Unseen)."
        )

        current_file_id = (
            f"sample_data_{sample_file_path.stat().st_mtime}"
        )

        if st.session_state.get("last_uploaded_file") != current_file_id:
            st.session_state["last_uploaded_file"] = current_file_id
            st.session_state.pop("results_df", None)
            st.session_state.pop("uploaded_raw_df", None)

        try:
            df = pd.read_csv(sample_file_path)
            df.columns = df.columns.str.strip().str.lower()
            df = df.reset_index(drop=True)
        except Exception as exc:
            st.error(f"Unable to read the sample data: {exc}")
            st.stop()
    else:
        st.error(
            f"Sample data file not found at `{sample_file_path}`."
        )
        st.stop()

elif uploaded_file is not None:
    current_file_id = f"{uploaded_file.name}_{uploaded_file.size}"

    if st.session_state.get("last_uploaded_file") != current_file_id:
        st.session_state["last_uploaded_file"] = current_file_id
        st.session_state.pop("results_df", None)
        st.session_state.pop("uploaded_raw_df", None)

    try:
        if uploaded_file.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_parquet(uploaded_file)

        df.columns = df.columns.str.strip().str.lower()
        df = df.reset_index(drop=True)

        st.success(
            f"Successfully loaded `{uploaded_file.name}`"
        )

    except Exception as exc:
        st.error(f"Unable to read the uploaded file: {exc}")
        st.stop()


if (input_source == "Use sample data") or (uploaded_file is not None):
    st.divider()

    # --- Dataset Profile ---
    render_section_header(
        "Profiling",
        "Dataset health at a glance",
        "A compact view of volume, structure, missingness, and duplicate records before inference begins.",
    )

    total_rows = len(df)
    total_columns = df.shape[1]
    missing_cells = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    numeric_columns = df.select_dtypes(include="number").shape[1]
    categorical_columns = total_columns - numeric_columns

    missing_percentage = (
        missing_cells / (total_rows * total_columns) * 100
        if total_rows > 0 and total_columns > 0
        else 0.0
    )

    profile_col1, profile_col2, profile_col3 = st.columns(
        3,
        gap="medium",
    )

    with profile_col1:
        render_metric_card(
            "Applicant records",
            f"{total_rows:,}",
            "#60a5fa",
            "rgba(59, 130, 246, 0.09)",
        )

    with profile_col2:
        render_metric_card(
            "Feature count",
            f"{total_columns:,}",
            "#8b5cf6",
            "rgba(139, 92, 246, 0.09)",
        )

    with profile_col3:
        render_metric_card(
            "Numeric / categorical",
            f"{numeric_columns} / {categorical_columns}",
            "#22d3ee",
            "rgba(34, 211, 238, 0.08)",
        )

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    profile_col4, profile_col5, profile_col6 = st.columns(
        3,
        gap="medium",
    )

    with profile_col4:
        render_metric_card(
            "Missing cells",
            f"{missing_cells:,}",
            "#f59e0b",
            "rgba(245, 158, 11, 0.09)",
        )

    with profile_col5:
        render_metric_card(
            "Missing rate",
            f"{missing_percentage:.2f}%",
            "#fb7185",
            "rgba(244, 63, 94, 0.08)",
        )

    with profile_col6:
        render_metric_card(
            "Duplicate rows",
            f"{duplicate_rows:,}",
            "#a78bfa",
            "rgba(167, 139, 250, 0.08)",
        )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown(
            "#### 📋 Applicant data preview"
        )
        st.caption(
            "First five records from the active input dataset."
        )
        st.dataframe(
            df.head(5),
            width="stretch",
            height=210,
            hide_index=True,
        )

    st.divider()

    # --- Prediction Section ---
    render_section_header(
        "Inference",
        "Loan Repayment Assessment",
        "Validate the schema, transform the records, and produce calibrated repayment-risk probabilities with the production pipeline.",
    )

    execution_col1, execution_col2 = st.columns(
        [1.55, 1],
        gap="large",
    )

    with execution_col1:
        with st.container(border=True):
            st.markdown(
                "#### 🎯 Ready for inference?"
            )
            st.caption(
                "The active dataset above is passed directly into the existing prediction pipeline."
            )

            if input_source == "Use sample data":
                st.markdown(
                    ":green-badge[● Sample dataset active]"
                )
            elif uploaded_file is not None:
                st.markdown(
                    ":blue-badge[● Uploaded dataset active]"
                )
            else:
                st.markdown(
                    ":gray-badge[● Waiting for input]"
                )

            run_predictions_clicked = st.button(
                "Run Predictions",
                type="primary",
                icon=":material/play_arrow:",
                width="stretch",
            )

    with execution_col2:
        with st.container(border=True):
            st.markdown(
                "#### 🔐 Pipeline stages"
            )
            st.markdown(
                """
                <div style="
                    color:#94a3b8;
                    font-size:0.79rem;
                    line-height:1.85;
                ">
                    <b style="color:#e2e8f0;">01</b> Schema validation<br>
                    <b style="color:#e2e8f0;">02</b> Feature transformation<br>
                    <b style="color:#e2e8f0;">03</b> Calibrated probability inference
                </div>
                """,
                unsafe_allow_html=True,
            )

    if run_predictions_clicked:
        try:
            pipeline = load_prediction_pipeline()
            raw_df = pipeline.read_input(df)

            # Schema Validation
            with st.spinner("🔍 Validating input schema..."):
                pipeline.validate_input_schema(raw_df)

            # Feature Transformation & Engineering
            with st.spinner(
                "⚙️ Transforming features & cleaning anomalies..."
            ):
                X_inference = pipeline.transform_features(raw_df)

            # Calibrated Risk Inference
            with st.spinner(
                "🎯 Generating calibrated repayment risk inferences..."
            ):
                probabilities = pipeline.model.predict_proba(
                    X_inference
                )[:, 1]

                results_df = raw_df.copy()
                results_df["repayment_risk"] = np.round(
                    probabilities,
                    5,
                )

            st.session_state["results_df"] = results_df
            st.session_state["uploaded_raw_df"] = raw_df
            st.session_state["transformed_inference_df"] = X_inference

            st.success(
                "✅ Inference completed successfully!"
            )

        except ValueError as ve:
            st.error(str(ve))

        except Exception as e:
            st.error(
                f"Prediction failed during execution: {e}"
            )

    # Display Results, SHAP, and Drift Monitoring
    if "results_df" in st.session_state:
        results_df = st.session_state["results_df"]
        raw_input_df = st.session_state["uploaded_raw_df"]
        pipeline = load_prediction_pipeline()

        prob_col = (
            "repayment_risk"
            if "repayment_risk" in results_df.columns
            else (
                "default_probability"
                if "default_probability" in results_df.columns
                else [
                    c
                    for c in results_df.columns
                    if "prob" in c.lower()
                    or "risk" in c.lower()
                ][0]
            )
        )

        avg_prob = float(results_df[prob_col].mean())
        max_prob = float(results_df[prob_col].max())
        min_prob = float(results_df[prob_col].min())

        st.divider()

        # --- Portfolio Risk Snapshot ---
        render_section_header(
            "Portfolio Risk",
            "Scored cohort snapshot",
            "The active batch has been scored successfully. Use the summary below to understand overall repayment-risk concentration.",
        )

        risk_col1, risk_col2, risk_col3 = st.columns(
            3,
            gap="medium",
        )

        with risk_col1:
            with st.container(border=True):
                st.markdown(
                    f"""
                    <div class="risk-summary">
                        <div class="risk-label">
                            Average repayment risk
                        </div>
                        <div class="risk-score">
                            {avg_prob:.4f}
                        </div>
                        <div class="risk-caption">
                            Mean probability across the scored cohort
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with risk_col2:
            with st.container(border=True):
                st.markdown(
                    f"""
                    <div class="risk-summary">
                        <div class="risk-label">
                            Highest repayment risk
                        </div>
                        <div class="risk-score">
                            {max_prob:.4f}
                        </div>
                        <div class="risk-caption">
                            Maximum predicted probability in the batch
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with risk_col3:
            with st.container(border=True):
                st.markdown(
                    f"""
                    <div class="risk-summary">
                        <div class="risk-label">
                            Lowest repayment risk
                        </div>
                        <div class="risk-score">
                            {min_prob:.4f}
                        </div>
                        <div class="risk-caption">
                            Minimum predicted probability in the batch
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Results & Calibration Layout
        res_col, plot_col = st.columns(
            [1.20, 1.30],
            gap="large",
        )

        with res_col:
            with st.container(border=True):
                st.markdown(
                    "### 🎯 Prediction Probabilities"
                )
                st.caption(
                    "A compact preview of the production model's output."
                )

                display_cols = [
                    c
                    for c in [
                        "sk_id_curr",
                        "SK_ID_CURR",
                        prob_col,
                    ]
                    if c in results_df.columns
                ]

                if not display_cols or len(display_cols) == 1:
                    display_cols = [prob_col] + [
                        c
                        for c in results_df.columns
                        if c != prob_col
                    ][:5]

                st.dataframe(
                    results_df[display_cols].head(8),
                    width="stretch",
                    hide_index=True,
                )

        with plot_col:
            with st.container(border=True):
                st.markdown(
                    "### 🔧 Probability Calibration"
                )
                st.caption(
                    "Holdout calibration against observed outcomes."
                )

                y_test_true, y_test_prob = (
                    get_test_calibration_data()
                )

                if (
                    y_test_true is not None
                    and y_test_prob is not None
                ):
                    fig_cal, _ = plot_calibration_curve(
                        y_true=y_test_true,
                        y_prob=y_test_prob,
                        title="Holdout Calibration Curve",
                    )

                    fig_cal.set_size_inches(
                        7.2,
                        5.1,
                    )

                    st.pyplot(
                        fig_cal,
                        width="stretch",
                    )

                    st.markdown(
                        """
                        <div style="
                            margin-top: 8px;
                            padding: 12px 14px;
                            border: 1px solid rgba(251, 191, 36, 0.28);
                            border-left: 3px solid #FBBF24;
                            border-radius: 10px;
                            background: linear-gradient(
                                135deg,
                                rgba(120, 83, 7, 0.34),
                                rgba(69, 48, 5, 0.24)
                            );
                            color: #cbd5e1;
                            font-size: 0.74rem;
                            line-height: 1.55;
                            box-shadow:
                                inset 0 1px 0 rgba(255, 255, 255, 0.025),
                                0 4px 14px rgba(2, 6, 23, 0.12);
                        ">
                            <strong style="color:#FBBF24;">
                                Calibration:
                            </strong>
                            The prediction pipeline applies either
                            <strong style="color:#93C5FD;">
                                Sigmoid
                            </strong>
                            or
                            <strong style="color:#C4B5FD;">
                                Isotonic Regression
                            </strong>
                            on top of the selected candidate model to calibrate predicted probabilities.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    plt.close(fig_cal)
                else:
                    st.warning(
                        "Holdout test dataset not found for calibration visualization."
                    )

        # ID extraction & CSV Export
        id_candidates = [
            c
            for c in results_df.columns
            if c.lower() == "sk_id_curr"
        ]

        id_col = id_candidates[0] if id_candidates else None

        if id_col:
            export_df = pd.DataFrame(
                {
                    "sk_id_curr": results_df[id_col],
                    "repayment_risk": results_df[prob_col],
                }
            )
        else:
            export_df = pd.DataFrame(
                {
                    "id": results_df.index,
                    "repayment_risk": results_df[prob_col],
                }
            )

        csv_buffer = export_df.to_csv(
            index=False
        ).encode("utf-8")

        download_col1, download_col2 = st.columns(
            [1, 3],
            gap="medium",
        )

        with download_col1:
            st.download_button(
                label="Download predictions",
                data=csv_buffer,
                file_name=(
                    "home_credit_repayment_risk_predictions.csv"
                ),
                mime="text/csv",
                icon=":material/download:",
                type="primary",
                width="stretch",
            )

        with download_col2:
            st.caption(
                f"{len(export_df):,} labeled applicants ready for export."
            )

        st.divider()

        # Model Explainability
        render_section_header(
            "Explainability",
            "Understand how the model scored this applicant",
            "Inspect one applicant's risk drivers and compare those explanations with the broader scored cohort.",
        )

        st.markdown(
            "#### Select an applicant"
        )

        if id_col:
            all_ids = (
                results_df[id_col]
                .astype(str)
                .tolist()
            )

            selected_id_str = st.selectbox(
                f"Choose an Applicant ({id_col}):",
                options=all_ids,
                index=0,
                help=(
                    "Select any applicant ID to inspect "
                    "their scorecard and SHAP explanation."
                ),
            )

            selected_idx = results_df[
                results_df[id_col].astype(str)
                == selected_id_str
            ].index[0]

            applicant_score = float(
                results_df.loc[
                    selected_idx,
                    prob_col,
                ]
            )

        else:
            selected_idx = st.number_input(
                "Select row index:",
                min_value=0,
                max_value=len(results_df) - 1,
                value=0,
            )

            selected_id_str = (
                f"Row #{selected_idx}"
            )

            applicant_score = float(
                results_df.loc[
                    selected_idx,
                    prob_col,
                ]
            )

        # Dynamic Risk Tier Badge
        if applicant_score >= 0.5:
            risk_pill_html = (
                '<span class="risk-pill pill-high">'
                "⚠️ High Risk"
                "</span>"
            )
        elif applicant_score >= 0.2:
            risk_pill_html = (
                '<span class="risk-pill pill-med">'
                "🟡 Medium Risk"
                "</span>"
            )
        else:
            risk_pill_html = (
                '<span class="risk-pill pill-low">'
                "🟢 Low Risk"
                "</span>"
            )

        with st.container(border=True):
            card_c1, card_c2, card_c3 = st.columns(
                3,
                gap="large",
            )

            with card_c1:
                st.markdown(
                    f"""
                    <div class="risk-summary">
                        <div class="risk-label">
                            Selected applicant
                        </div>
                        <div class="risk-score"
                             style="font-size:1.55rem;">
                            {selected_id_str}
                        </div>
                        <div class="risk-caption">
                            Active record for individual analysis
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with card_c2:
                st.markdown(
                    f"""
                    <div class="risk-summary">
                        <div class="risk-label">
                            Repayment risk score
                        </div>
                        <div class="risk-score">
                            {applicant_score * 100:.2f}%
                        </div>
                        <div class="risk-caption">
                            Calibrated model probability
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with card_c3:
                st.markdown(
                    f"""
                    <div class="risk-summary">
                        <div class="risk-label">
                            Repayment risk tier
                        </div>
                        {risk_pill_html}
                        <div class="risk-caption">
                            Based on the application's predicted probability
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with st.spinner(
            "Generating SHAP explanations..."
        ):
            sample_size = min(
                len(raw_input_df),
                3_000,
            )

            sample_df = raw_input_df.sample(
                n=sample_size,
                random_state=42,
            ).reset_index(drop=True)

            transformed_cohort = (
                pipeline.transform_features(sample_df)
            )

            single_applicant_raw = (
                raw_input_df.iloc[[selected_idx]]
            )

            transformed_single = (
                pipeline.transform_features(
                    single_applicant_raw
                )
            )

            base_model = get_base_tree_estimator(
                pipeline.model
            )

            explainer = shap.TreeExplainer(
                base_model
            )

            single_shap_exp = explainer(
                transformed_single
            )

            if (
                hasattr(single_shap_exp, "values")
                and len(single_shap_exp.values.shape)
                == 3
            ):
                single_shap_slice = (
                    single_shap_exp[:, :, 1][0]
                )

                single_shap_values = (
                    single_shap_exp.values[:, :, 1]
                )
            else:
                single_shap_slice = (
                    single_shap_exp[0]
                )

                single_shap_values = (
                    single_shap_exp.values
                    if hasattr(
                        single_shap_exp,
                        "values",
                    )
                    else single_shap_exp
                )

            raw_expected_value = (
                explainer.expected_value
            )

            if isinstance(
                raw_expected_value,
                (list, np.ndarray),
            ):
                base_val = (
                    float(raw_expected_value[1])
                    if len(raw_expected_value) > 1
                    else float(raw_expected_value[0])
                )
            else:
                base_val = float(
                    raw_expected_value
                )

            cohort_shap_exp = explainer(
                transformed_cohort
            )

            if (
                hasattr(cohort_shap_exp, "values")
                and len(cohort_shap_exp.values.shape)
                == 3
            ):
                cohort_shap_slice = (
                    cohort_shap_exp[:, :, 1]
                )
            else:
                cohort_shap_slice = (
                    cohort_shap_exp
                )

        shap_tab1, shap_tab2, shap_tab3 = st.tabs(
            [
                f"📈 Waterfall · {selected_id_str}",
                f"⚡️ Decision · {selected_id_str}",
                "🌐 Global importance",
            ]
        )

        with shap_tab1:
            st.caption(
                f"Individual feature contributions to prediction for applicant **`{selected_id_str}`**."
            )

            col_plot1, _ = st.columns(
                [1.3, 1]
            )

            with col_plot1:
                fig_water, _ = (
                    plot_shap_waterfall(
                        single_shap_slice,
                        sample_id=selected_id_str,
                        max_display=8,
                        title=(
                            "Applicant Risk Breakdown"
                        ),
                    )
                )

                st.pyplot(
                    fig_water,
                    width="stretch",
                )

                plt.close(fig_water)

                st.markdown(
                    """
                    <div style="
                        margin-top: 8px;
                        padding: 12px 14px;
                        border: 1px solid rgba(251, 191, 36, 0.28);
                        border-left: 3px solid #FBBF24;
                        border-radius: 10px;
                        background: linear-gradient(
                            135deg,
                            rgba(120, 83, 7, 0.34),
                            rgba(69, 48, 5, 0.24)
                        );
                        color: #cbd5e1;
                        font-size: 0.74rem;
                        line-height: 1.55;
                        box-shadow:
                            inset 0 1px 0 rgba(255, 255, 255, 0.025),
                            0 4px 14px rgba(2, 6, 23, 0.12);
                    ">
                        <strong style="color:#FBBF24;">
                            SHAP Waterfall · Local Explanation
                        </strong><br>
                        Displays how individual feature contributions move the model output
                        from the baseline prediction to this applicant's final prediction.
                        Features are ordered by the magnitude of their contribution.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with shap_tab2:
            st.caption(
                f"Cumulative feature contributions to prediction for applicant **`{selected_id_str}`**."
            )

            col_plot2, _ = st.columns(
                [1.3, 1]
            )

            with col_plot2:
                fig_dec, _ = plot_shap_decision(
                    base_value=base_val,
                    shap_values=single_shap_values,
                    features_df=transformed_single,
                    sample_id=selected_id_str,
                    max_display=10,
                    title="Decision Trajectory",
                )

                st.pyplot(
                    fig_dec,
                    width="stretch",
                )

                plt.close(fig_dec)

                st.markdown(
                    """
                    <div style="
                        margin-top: 8px;
                        padding: 12px 14px;
                        border: 1px solid rgba(251, 191, 36, 0.28);
                        border-left: 3px solid #FBBF24;
                        border-radius: 10px;
                        background: linear-gradient(
                            135deg,
                            rgba(120, 83, 7, 0.34),
                            rgba(69, 48, 5, 0.24)
                        );
                        color: #cbd5e1;
                        font-size: 0.74rem;
                        line-height: 1.55;
                        box-shadow:
                            inset 0 1px 0 rgba(255, 255, 255, 0.025),
                            0 4px 14px rgba(2, 6, 23, 0.12);
                    ">
                        <strong style="color:#FBBF24;">
                            SHAP Decision Plot · Local Explanation
                        </strong><br>
                        Shows the cumulative effect of feature contributions as they move
                        the prediction from the expected model output toward this applicant's
                        final predicted value.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with shap_tab3:
            st.caption(
                "Feature impact distribution across applicants in the selected cohort."
            )

            col_plot3, _ = st.columns(
                [1.3, 1]
            )

            with col_plot3:
                fig_bee, _ = plot_shap_beeswarm(
                    cohort_shap_slice,
                    max_display=10,
                    title=(
                        "Global Risk Drivers (SHAP Beeswarm)"
                    ),
                )

                st.pyplot(
                    fig_bee,
                    width="stretch",
                )

                plt.close(fig_bee)

                st.markdown(
                    """
                    <div style="
                        margin-top: 8px;
                        padding: 12px 14px;
                        border: 1px solid rgba(251, 191, 36, 0.28);
                        border-left: 3px solid #FBBF24;
                        border-radius: 10px;
                        background: linear-gradient(
                            135deg,
                            rgba(120, 83, 7, 0.34),
                            rgba(69, 48, 5, 0.24)
                        );
                        color: #cbd5e1;
                        font-size: 0.74rem;
                        line-height: 1.55;
                        box-shadow:
                            inset 0 1px 0 rgba(255, 255, 255, 0.025),
                            0 4px 14px rgba(2, 6, 23, 0.12);
                    ">
                        <strong style="color:#FBBF24;">
                            SHAP Beeswarm · Global Explanation
                        </strong><br>
                        Shows the distribution and magnitude of feature contributions across
                        the selected applicant cohort, highlighting features with the greatest
                        influence on model predictions.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.divider()

        # Production Monitoring & Drift Detection
        render_section_header(
            "Governance",
            "Monitor production behavior and distribution shift",
            "Compare the current scored cohort against established probability and raw-feature reference baselines.",
        )

        monitoring_tab1, monitoring_tab2 = st.tabs(
            [
                "📈 Prediction drift",
                "🧪 Feature drift",
            ]
        )

        # Prediction Drift
        with monitoring_tab1:
            with st.container(border=True):
                st.markdown(
                    "### Probability distribution drift"
                )

                st.caption(
                    "Evaluates Population Stability Index (PSI) between Out-of-Fold (OOF) baseline predictions and currently scored applicants."
                )

                oof_reference_df = (
                    get_oof_predictions_reference()
                )

                if oof_reference_df is None:
                    st.warning(
                        "⚠️ OOF reference predictions not found at "
                        f"`{trainer_config.oof_predictions_path}`. "
                        "Ensure model training has executed."
                    )
                else:
                    oof_cols = [
                        c
                        for c in oof_reference_df.columns
                        if "prob" in c.lower()
                        or "risk" in c.lower()
                        or "oof" in c.lower()
                        or "pred" in c.lower()
                    ]

                    oof_score_col = (
                        oof_cols[0]
                        if oof_cols
                        else oof_reference_df.columns[-1]
                    )

                    ref_pred_df = pd.DataFrame(
                        {
                            "repayment_risk":
                                oof_reference_df[
                                    oof_score_col
                                ].values
                        }
                    )

                    curr_pred_df = pd.DataFrame(
                        {
                            "repayment_risk":
                                results_df[
                                    prob_col
                                ].values
                        }
                    )

                    with st.spinner(
                        "Computing Prediction Drift report (PSI)..."
                    ):
                        pred_report = Report(
                            [
                                ValueDrift(
                                    column="repayment_risk",
                                    method="psi",
                                )
                            ]
                        )

                        pred_eval = pred_report.run(
                            current_data=curr_pred_df,
                            reference_data=ref_pred_df,
                        )

                        render_evidently_html(
                            pred_eval,
                            default_height=500,
                        )

        # Raw Feature Data Drift
        with monitoring_tab2:
            with st.container(border=True):
                st.markdown(
                    "### Raw feature distribution drift"
                )

                st.caption(
                    "Evaluates multi-column statistical distribution shifts between baseline `train.csv` and the active applicant batch."
                )

                raw_train_ref_df = (
                    get_raw_training_reference()
                )

                if raw_train_ref_df is None:
                    st.warning(
                        "⚠️ Baseline raw training data not found at "
                        f"`{ingestion_config.train_file_path}`."
                    )
                else:
                    raw_train_ref_df.columns = (
                        raw_train_ref_df.columns
                        .str.strip()
                        .str.lower()
                    )

                    # Match common raw features (excluding target and ID keys)
                    excluded_cols = {
                        "target",
                        "sk_id_curr",
                        "index",
                        "id",
                    }

                    common_raw_features = [
                        c
                        for c in raw_input_df.columns
                        if c in raw_train_ref_df.columns
                        and c not in excluded_cols
                    ]

                    if not common_raw_features:
                        st.info(
                            "No matching raw features found between uploaded data and baseline training data."
                        )
                    else:
                        sample_size_ref = min(
                            len(raw_train_ref_df),
                            5_000,
                        )

                        ref_raw_sample = (
                            raw_train_ref_df[
                                common_raw_features
                            ].sample(
                                n=sample_size_ref,
                                random_state=42,
                            )
                        )

                        curr_raw_sample = (
                            raw_input_df[
                                common_raw_features
                            ]
                        )

                        with st.spinner(
                            "Evaluating Raw Data Drift across input features..."
                        ):
                            feature_report = Report(
                                [DataDriftPreset()]
                            )

                            feature_eval = (
                                feature_report.run(
                                    current_data=curr_raw_sample,
                                    reference_data=ref_raw_sample,
                                )
                            )

                            render_evidently_html(
                                feature_eval,
                                default_height=800,
                            )