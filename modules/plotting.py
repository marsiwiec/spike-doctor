# ==============================================================================
# Plotting Functions
# ==============================================================================

import warnings
from typing import Optional, Sequence, Dict, Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyabf

from modules import constants, helper


def _plot_error_message(ax: plt.Axes, message: str, title: Optional[str] = None):
    """Helper to display an error message on a plot axes."""
    ax.text(
        0.5, 0.5, message, color="red", ha="center", va="center",
        fontsize=9, transform=ax.transAxes,
    )
    if title:
        ax.set_title(title, fontsize=9)
    ax.tick_params(
        axis="both", labelbottom=False, labelleft=False, bottom=False, left=False
    )
    for spine in ("top", "right", "bottom", "left"):
        ax.spines[spine].set_visible(False)


def plot_raw_traces(
    abf: Optional[pyabf.ABF],
    filename: str,
    ax: plt.Axes,
    channel: int = 0,
    load_error: Optional[str] = None,
    title_prefix: str = "Raw Traces",
) -> None:
    """Plots raw voltage traces onto a given axes."""
    plot_title = f"{title_prefix}: {filename}"

    if load_error:
        _plot_error_message(ax, f"Load Error:\n{load_error}", plot_title)
        return

    if abf is None:
        _plot_error_message(ax, "Plot Error:\nABF Data Not Available", plot_title)
        return

    try:
        num_sweeps = getattr(abf, "sweepCount", 0)
        if num_sweeps == 0:
            _plot_error_message(ax, "No sweeps in file", plot_title)
            return

        num_to_plot = min(num_sweeps, constants.MAX_RAW_PLOT_SWEEPS)
        for i in range(num_to_plot):
            abf.setSweep(i, channel=channel)
            ax.plot(abf.sweepX, abf.sweepY, lw=0.5, alpha=0.7)
        if num_sweeps > num_to_plot:
            plot_title += f" (First {num_to_plot})"

        plot_title += " (CC)" if helper.is_current_clamp(abf) else " (VC?)"
        ax.set_title(plot_title, fontsize=9)
        ax.set_xlabel(f"{getattr(abf, 'sweepLabelX', 'Time')}", fontsize=8)
        ax.set_ylabel(f"{getattr(abf, 'sweepLabelY', 'Signal')}", fontsize=8)
        ax.tick_params(axis="both", which="major", labelsize=8)
        ax.grid(True, linestyle=":", alpha=0.6)

    except Exception as e:
        helper._log_message("ERROR", filename, None, f"Raw trace plotting failed: {e}")
        ax.cla()
        _plot_error_message(ax, f"Plotting Error:\n{e}", f"Error: {filename}")


def plot_feature_vs_current(
    analysis_df: Optional[pd.DataFrame],
    feature_name: str,
    current_col: str,
    filename: str,
    abf: Optional[pyabf.ABF],
    ax: plt.Axes,
) -> None:
    """Plots a specified feature against current onto given axes."""
    feature_title = feature_name.replace("_", " ").title()
    plot_title = f"{feature_title} vs Current"
    ax.set_title(plot_title, fontsize=9)

    if not helper.is_valid_analysis_df(analysis_df):
        _plot_error_message(ax, "No analysis data.", plot_title)
        return
    if feature_name not in analysis_df.columns:
        _plot_error_message(ax, f"Feature '{feature_name}'\nnot found.", plot_title)
        return
    if current_col not in analysis_df.columns:
        _plot_error_message(ax, f"Current col '{current_col}'\nnot found.", plot_title)
        return
    if not pd.api.types.is_numeric_dtype(analysis_df[current_col]) or not pd.api.types.is_numeric_dtype(analysis_df[feature_name]):
        _plot_error_message(ax, "Data columns contain\nnon-numeric values.", plot_title)
        return

    plot_data = analysis_df[[current_col, feature_name]].dropna()
    if plot_data.empty:
        _plot_error_message(ax, "No valid data points\nfor plotting.", plot_title)
        return

    y_label = feature_title
    feature_units = helper.get_feature_units(feature_name, abf)
    if feature_units:
        y_label += f" ({feature_units})"

    try:
        ax.plot(
            plot_data[current_col], plot_data[feature_name],
            marker="o", linestyle="-", markersize=4,
        )
        ax.set_xlabel("Current step (pA)", fontsize=8)
        ax.set_ylabel(y_label, fontsize=8)
        ax.tick_params(axis="both", which="major", labelsize=8)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.axhline(0, color="grey", lw=0.5, linestyle="--")
        ax.axvline(0, color="grey", lw=0.5, linestyle="--")
    except Exception as e:
        helper._log_message("ERROR", filename, None, f"Plotting '{feature_name}' failed: {e}")
        ax.cla()
        _plot_error_message(ax, f"Plotting Error:\n{e}", f"Error: {filename}")


def plot_phase_plane(
    voltage: Optional[np.ndarray],
    dvdt: Optional[np.ndarray],
    filename: str,
    sweep_num: Optional[int] = None,
    current_pA: Optional[float] = None,
    title_suffix: str = "",
    ax: plt.Axes = None,
) -> None:
    """Generates a phase-plane plot onto given axes."""
    plot_title = "Phase Plane Plot at 2x Rheobase\n"
    file_sweep_info = f"{filename}"
    if sweep_num is not None:
        file_sweep_info += f" (Sw {sweep_num}"
        if current_pA is not None:
            file_sweep_info += f", ~{current_pA:.1f} pA"
        file_sweep_info += ")"

    if title_suffix:
        plot_title += f": {title_suffix}"
    else:
        plot_title += f" ({file_sweep_info})"

    ax.set_title(plot_title, fontsize=9)

    show_error_text = False
    err_msg = title_suffix.replace(": ", ":\n")
    if (
        voltage is None or dvdt is None
        or not isinstance(voltage, np.ndarray) or not isinstance(dvdt, np.ndarray)
        or len(voltage) != len(dvdt) or len(voltage) == 0
    ):
        if not title_suffix or "N/A" in title_suffix:
            err_msg = "Invalid or mismatched\nVoltage/dVdt data"
        show_error_text = True
    elif title_suffix and any(
        s in title_suffix.lower()
        for s in ["error", "invalid", "missing", "failed", "not found", "no abf"]
    ):
        show_error_text = True

    if show_error_text:
        ax.text(
            0.5, 0.5, err_msg, ha="center", va="center",
            color="gray", fontsize=9, transform=ax.transAxes,
        )
        if "error" not in title_suffix.lower():
            helper._log_message(
                "WARN", filename, sweep_num,
                f"Phase plane plot skipped/failed: {err_msg.replace(chr(10),' ')}",
            )
        ax.tick_params(
            axis="both", labelbottom=False, labelleft=False, bottom=False, left=False
        )
        return

    try:
        ax.plot(voltage, dvdt, color="black", lw=0.5)
        ax.set_xlabel("Membrane Potential (mV)", fontsize=8)
        ax.set_ylabel("dV/dt (mV/ms)", fontsize=8)
        ax.tick_params(axis="both", which="major", labelsize=8)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.axhline(0, color="grey", lw=0.5, linestyle="--")
    except Exception as e:
        helper._log_message("ERROR", filename, sweep_num, f"Phase plane plotting failed: {e}")
        ax.cla()
        _plot_error_message(ax, f"Plotting Error:\n{e}", f"Phase Plot Error: {filename}")


def _prepare_phase_plot_data(
    analysis_df: Optional[pd.DataFrame],
    abf_obj: Optional[pyabf.ABF],
    filename: str,
    current_col: str,
    channel: int = 0,
) -> dict:
    """
    Calculates data needed for the phase plot (V, dV/dt) for a sweep near 2x rheobase.

    Returns a dict with keys:
        voltage, dvdt, sweep, current, status
    On failure, voltage/dvdt/sweep/current are None and status is the error message.
    """
    if not isinstance(abf_obj, pyabf.ABF):
        return {"voltage": None, "dvdt": None, "sweep": None, "current": None, "status": "ABF Not Loaded"}
    if not helper.is_valid_analysis_df(analysis_df):
        return {"voltage": None, "dvdt": None, "sweep": None, "current": None, "status": "Analysis Failed"}
    if current_col not in analysis_df.columns or "spike_count" not in analysis_df.columns:
        return {"voltage": None, "dvdt": None, "sweep": None, "current": None, "status": "Required Columns Missing"}
    if not pd.api.types.is_numeric_dtype(analysis_df[current_col]) or not pd.api.types.is_numeric_dtype(analysis_df["spike_count"]):
        return {"voltage": None, "dvdt": None, "sweep": None, "current": None, "status": "Non-numeric Data"}

    spiking = analysis_df[
        (analysis_df["spike_count"].fillna(0) >= 1)
        & (analysis_df[current_col].fillna(-np.inf) > 0)
        & np.isfinite(analysis_df[current_col])
    ]
    if spiking.empty:
        return {"voltage": None, "dvdt": None, "sweep": None, "current": None, "status": "Rheobase Not Found"}

    rheobase = spiking[current_col].min()
    target_current = 2 * rheobase

    valid = analysis_df.dropna(subset=[current_col])
    valid = valid[np.isfinite(valid[current_col])]
    if valid.empty:
        return {"voltage": None, "dvdt": None, "sweep": None, "current": None, "status": "No Valid Current Data"}

    closest_idx = (valid[current_col] - target_current).abs().idxmin()
    closest = valid.loc[closest_idx]
    if "sweep" not in closest:
        return {"voltage": None, "dvdt": None, "sweep": None, "current": None, "status": "Sweep Column Missing"}

    target_sweep = int(closest["sweep"])
    target_current_pA = closest[current_col]

    try:
        abf_obj.setSweep(target_sweep, channel=channel)
        phase_v = abf_obj.sweepY
        time_s = abf_obj.sweepX

        if not (isinstance(phase_v, np.ndarray) and isinstance(time_s, np.ndarray)
                and len(phase_v) > 1 and len(time_s) > 1 and len(phase_v) == len(time_s)):
            return {"voltage": phase_v, "dvdt": None, "sweep": target_sweep, "current": target_current_pA, "status": "Sweep Data Invalid"}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            phase_dvdt = np.gradient(phase_v, time_s * 1000.0)

        return {"voltage": phase_v, "dvdt": phase_dvdt, "sweep": target_sweep, "current": target_current_pA, "status": ""}
    except IndexError:
        return {"voltage": None, "dvdt": None, "sweep": target_sweep, "current": target_current_pA, "status": f"Sweep Index Error: {target_sweep}"}
    except Exception as e:
        return {"voltage": None, "dvdt": None, "sweep": target_sweep, "current": target_current_pA, "status": f"Sweep Load Error: {e}"}


def _generate_summary_plots_for_file(
    result_data: Dict[str, Any],
    axes: Sequence[plt.Axes],
    current_col: str = constants.CURRENT_COL_NAME,
) -> None:
    """
    Generates the standard set of summary plots (raw, SC, phase) onto provided axes.
    Expects 3 axes: [raw_ax, sc_ax, phase_ax].
    """
    if len(axes) != 3:
        helper._log_message(
            "ERROR", result_data.get("original_filename", "UnknownFile"), None,
            "_generate_summary_plots_for_file expects 3 axes.",
        )
        return

    filename = result_data.get("original_filename", "Unknown Filename")
    abf_obj = result_data.get("abf_object")
    load_err = result_data.get("load_error")
    analysis_df = result_data.get("analysis_df")
    channel = result_data.get("channel_selection", 0)

    raw_ax, sc_ax, phase_ax = axes

    # 1. Raw Trace Plot
    try:
        plot_raw_traces(
            abf_obj, filename, ax=raw_ax, channel=channel,
            load_error=load_err, title_prefix="Raw",
        )
    except Exception as e_plot:
        helper._log_message("ERROR", filename, None, f"Summary Raw Plot Error: {e_plot}")
        _plot_error_message(raw_ax, f"Raw Plot Error:\n{e_plot}", f"Raw Error: {filename}")

    # 2. Spike Count vs Current Plot
    try:
        if load_err:
            _plot_error_message(sc_ax, f"Load Error:\n{load_err}", "Spike Count vs Current")
        elif not helper.is_valid_analysis_df(analysis_df):
            _plot_error_message(sc_ax, "Analysis skipped\nor failed.", "Spike Count vs Current")
        else:
            plot_feature_vs_current(
                analysis_df, "spike_count", current_col, filename, abf_obj, ax=sc_ax,
            )
    except Exception as e_plot:
        helper._log_message("ERROR", filename, None, f"Summary SC Plot Error: {e_plot}")
        _plot_error_message(sc_ax, f"SC Plot Error:\n{e_plot}", "Spike Count vs Current")

    # 3. Phase Plane Plot
    try:
        data = _prepare_phase_plot_data(analysis_df, abf_obj, filename, current_col, channel=channel)
        plot_phase_plane(
            data["voltage"], data["dvdt"], filename,
            data["sweep"], data["current"], data["status"], ax=phase_ax,
        )
    except Exception as e_plot:
        helper._log_message("ERROR", filename, None, f"Summary Phase Plot Error: {e_plot}")
        _plot_error_message(phase_ax, f"Phase Plot Error:\n{e_plot}", "Phase Plane")
