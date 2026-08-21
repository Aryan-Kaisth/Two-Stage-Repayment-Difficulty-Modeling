from pathlib import Path
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import CalibrationDisplay


def plot_calibration_curve(
    y_true: Union[np.ndarray, pd.Series],
    y_prob: Union[np.ndarray, pd.Series],
    n_bins: int = 10,
    strategy: str = "quantile",
    title: Optional[str] = "Calibration Reliability Curve",
    save_path: Optional[Union[str, Path]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Renders a calibration curve to evaluate model reliability.
    """
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
        label="Perfect Calibration",
        color="gray",
    )
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("Mean Predicted Probability", fontsize=8)
    ax.set_ylabel("Fraction of Positives", fontsize=8)
    ax.tick_params(axis="both", labelsize=8)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")

    return fig, ax


def plot_shap_beeswarm(
    shap_exp: shap.Explanation,
    max_display: int = 10,
    title: Optional[str] = "Global Feature Impact",
    save_path: Optional[Union[str, Path]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Renders a SHAP Beeswarm plot showing global feature impact and distribution.
    """
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

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")

    return fig, ax


def plot_shap_waterfall(
    shap_exp_sample: shap.Explanation,
    sample_idx: int = 0,
    max_display: int = 8,
    title: Optional[str] = "Applicant Risk Breakdown",
    save_path: Optional[Union[str, Path]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Renders a SHAP Waterfall plot showing individual feature contributions.
    """
    fig = plt.figure(figsize=(5.5, 3.6))
    shap.plots.waterfall(
        shap_exp_sample,
        max_display=max_display,
        show=False,
    )
    plot_title = f"{title} (Applicant #{sample_idx})" if title else f"Applicant #{sample_idx}"
    plt.title(plot_title, fontsize=10, fontweight="bold", pad=8)
    plt.tick_params(axis="both", labelsize=8)
    plt.tight_layout()

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")

    return fig, plt.gca()


def plot_shap_decision(
    base_value: float,
    shap_values: np.ndarray,
    features_df: pd.DataFrame,
    sample_idx: Optional[int] = None,
    max_display: int = 10,
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Renders a SHAP Decision plot showing cumulative feature contribution
    trajectories from base expectation to the final model prediction score.
    """
    fig = plt.figure(figsize=(5.5, 3.8))

    if sample_idx is not None:
        sample_shap = shap_values[sample_idx]
        sample_features = features_df.iloc[sample_idx]
        highlight = None
    else:
        sample_shap = shap_values[:15]
        sample_features = features_df.iloc[:15]
        highlight = 0

    shap.decision_plot(
        base_value=base_value,
        shap_values=sample_shap,
        features=sample_features,
        feature_names=features_df.columns.tolist(),
        feature_display_range=slice(None, -max_display - 1, -1),
        show=False,
        highlight=highlight,
    )

    plot_title = title if title else "SHAP Decision Trajectory"
    plt.title(plot_title, fontsize=10, fontweight="bold", pad=8)
    plt.tick_params(axis="both", labelsize=8)
    plt.tight_layout()

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")

    return fig, plt.gca()