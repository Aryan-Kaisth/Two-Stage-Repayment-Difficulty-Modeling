from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import CalibrationDisplay


def plot_calibration_curve(
    y_true,
    y_prob,
    n_bins=10,
    strategy="quantile",
    title="Calibration Reliability Curve",
):
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    CalibrationDisplay.from_predictions(
        y_true,
        y_prob,
        n_bins=n_bins,
        strategy=strategy,
        ax=ax,
    )
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
    )
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("Mean Predicted Probability", fontsize=8)
    ax.set_ylabel("Fraction of Positives", fontsize=8)
    ax.tick_params(axis="both", labelsize=8)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig, ax


def plot_shap_beeswarm(shap_exp, max_display=10, title="Global Feature Impact"):
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    shap.plots.beeswarm(
        shap_exp,
        max_display=max_display,
        show=False,
        plot_size=None,
        ax=ax,
    )
    ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
    ax.tick_params(axis="both", labelsize=8)
    plt.tight_layout()
    return fig, ax


def plot_shap_waterfall(
    shap_exp_sample,
    sample_id="Selected Applicant",
    max_display=8,
    title="Applicant Risk Breakdown",
):
    fig = plt.figure(figsize=(5.5, 3.6))
    shap.plots.waterfall(
        shap_exp_sample,
        max_display=max_display,
        show=False,
    )
    plt.title(f"{title} (ID: {sample_id})", fontsize=10, fontweight="bold", pad=8)
    plt.tick_params(axis="both", labelsize=8)
    plt.tight_layout()
    return fig, plt.gca()


def plot_shap_decision(
    base_value: float,
    shap_values: np.ndarray,
    features_df: pd.DataFrame,
    sample_id: str = "Selected Applicant",
    max_display: int = 10,
    title: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=(5.5, 3.8))

    # Single-instance decision trajectory
    shap.decision_plot(
        base_value=base_value,
        shap_values=shap_values,
        features=features_df.iloc[0],
        feature_names=features_df.columns.tolist(),
        feature_display_range=slice(None, -max_display - 1, -1),
        show=False,
        highlight=None,
    )

    plot_title = title if title else f"Decision Trajectory (ID: {sample_id})"
    plt.title(plot_title, fontsize=10, fontweight="bold", pad=8)
    plt.tick_params(axis="both", labelsize=8)
    plt.tight_layout()
    return fig, plt.gca()
