# ==============================================================================
# Plotting Functions
# ==============================================================================

import warnings
from typing import Optional, Sequence, Dict, Any, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyabf

from modules import constants, helper


def _plot_error_message(ax: plt.Axes, message: str, title: Optional[str] = None):
    """Helper to display an error message on a plot axes."""
    ax.text(
        0.5,
        0.5,
        message,
        color="red",
        ha="center",
        va="center",
        fontsize=9,
        transform=ax.transAxes,  # Use axes coordinates
    )
    if title:
        ax.set_title(title, fontsize=9)
    ax.tick_params(
        axis="both", labelbottom=False, labelleft=False, bottom=False, left=False
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)


def plot_raw_traces(
    abf: Optional[pyabf.ABF],
    filename: str,
    ax: plt.Axes,
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
            ax.text(
                0.5,
                0.5,
                "No sweeps in file",
                ha="center",
                va="center",
                fontsize=9,
                transform=ax.transAxes,
            )
            ax.set_title(plot_title, fontsize=9)
            ax.tick_params(
                axis="both",
                labelbottom=False,
                labelleft=False,
                bottom=False,
                left=False,
            )
        else:
            num_to_plot = min(num_sweeps, constants.MAX_RAW_PLOT_SWEEPS)
            for i in range(num_to_plot):
                abf.setSweep(i)
                ax.plot(abf.sweepX, abf.sweepY, lw=0.5, alpha=0.7)
            if num_sweeps > num_to_plot:
                plot_title += f" (First {num_to_plot})"

        plot_title += " (CC)" if helper.is_current_clamp(abf) else " (VC?)"
        ax.set_title(plot_title, fontsize=9)
        ax.set_xlabel(
            f"{getattr(abf, 'sweepLabelX', 'Time')}",
            fontsize=8,
        )
        ax.set_ylabel(
            f"{getattr(abf, 'sweepLabelY', 'Signal')}",
            fontsize=8,
        )
        ax.tick_params(axis="both", which="major", labelsize=8)
        ax.grid(True, linestyle=":", alpha=0.6)

    except Exception as e:
        helper._log_message("ERROR", filename, None, f"Raw trace plotting failed: {e}")
        ax.cla()  # Clear axes before plotting error
        _plot_error_message(ax, f"Plotting Error:\n{e}", f"Error: {filename}")


def plot_feature_vs_current(
    analysis_df: Optional[pd.DataFrame],
    feature_name,
    current_col: str,
    filename: str,
    abf: Optional[pyabf.ABF],  # Used for units
    ax: plt.Axes,
) -> None:
    """Plots a specified feature against current onto given axes."""
    feature_title = feature_name.replace("_", " ").title()
    plot_title = f"{feature_title} vs Current"
    ax.set_title(plot_title, fontsize=9)

    error_message = None
    plot_data = pd.DataFrame()

    if not isinstance(analysis_df, pd.DataFrame) or analysis_df.empty:
        error_message = "No analysis data."
    elif feature_name not in analysis_df.columns:
        error_message = f"Feature '{feature_name}'\nnot found."
    elif current_col not in analysis_df.columns:
        error_message = f"Current col '{current_col}'\nnot found."
    else:
        # Check for non-numeric types before dropping NA
        if not pd.api.types.is_numeric_dtype(
            analysis_df[current_col]
        ) or not pd.api.types.is_numeric_dtype(analysis_df[feature_name]):
            error_message = "Data columns contain\nnon-numeric values."
        else:
            plot_data = analysis_df[[current_col, feature_name]].dropna()
            if plot_data.empty:
                error_message = "No valid data points\nfor plotting."

    if error_message:
        ax.text(
            0.5,
            0.5,
            error_message,
            ha="center",
            va="center",
            fontsize=9,
            color="gray",
            transform=ax.transAxes,
        )
        helper._log_message(
            "WARN",
            filename,
            None,
            f"Plotting '{feature_name}': {error_message.replace(chr(10),' ')}",
        )
        ax.tick_params(
            axis="both", labelbottom=False, labelleft=False, bottom=False, left=False
        )
        return

    y_label = feature_title
    current_units = "pA"  # Default assumption based on constants.CURRENT_COL_NAME
    feature_units = ""
    if "resistance" in feature_name.lower():
        feature_units = "MΩ"
    elif "constant" in feature_name.lower():
        feature_units = "ms"
    elif "capacitance" in feature_name.lower():
        feature_units = "pF"
    elif "frequency" in feature_name.lower() or "isi" in feature_name.lower():
        feature_units = "Hz"
    elif "time_to_" in feature_name.lower() or "latency" in feature_name.lower():
        feature_units = "ms"
    elif "voltage" in feature_name.lower() or "potential" in feature_name.lower():
        feature_units = getattr(abf, "sweepUnitsY", "mV") if abf else "mV"
    if feature_units:
        y_label += f" ({feature_units})"

    current_label = current_col.replace("_", " ").title()

    current_label += f" ({current_units})"

    try:
        ax.plot(
            plot_data[current_col],
            plot_data[feature_name],
            marker="o",
            linestyle="-",
            markersize=4,
        )
        ax.set_xlabel(current_label, fontsize=8)
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
    plot_title = f"Phase Plane Plot at 2x Rheobase\n"
    file_sweep_info = f"{filename}"
    if sweep_num is not None:
        file_sweep_info += f" (Sw {sweep_num}"
        if current_pA is not None:
            file_sweep_info += f", ~{current_pA:.1f} pA"
        file_sweep_info += ")"

    if title_suffix:
        plot_title += f": {title_suffix}"
    else:
        plot_title += f" ({file_sweep_info})"  # Add file/sweep if no error suffix

    ax.set_title(plot_title, fontsize=9)

    # Check data validity or if suffix indicates an issue
    show_error_text = False
    err_msg = title_suffix.replace(": ", ":\n")  # Default error message is the suffix
    if (
        voltage is None
        or dvdt is None
        or not isinstance(voltage, np.ndarray)
        or not isinstance(dvdt, np.ndarray)
        or len(voltage) != len(dvdt)
        or len(voltage) == 0
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
            0.5,
            0.5,
            err_msg,
            ha="center",
            va="center",
            color="gray",
            fontsize=9,
            transform=ax.transAxes,
        )
        # Log only if it wasn't logged during data prep
        if "error" not in title_suffix.lower():
            helper._log_message(
                "WARN",
                filename,
                sweep_num,
                f"Phase plane plot skipped/failed: {err_msg.replace(chr(10),' ')}",
            )
        ax.tick_params(
            axis="both", labelbottom=False, labelleft=False, bottom=False, left=False
        )
        return

    try:
        ax.plot(voltage, dvdt, color="black", lw=0.5)
        ax.set_xlabel(f"Membrane Potential (mV)", fontsize=8)
        ax.set_ylabel("dV/dt (mV/ms)", fontsize=8)
        ax.tick_params(axis="both", which="major", labelsize=8)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.axhline(0, color="grey", lw=0.5, linestyle="--")

    except Exception as e:
        helper._log_message("ERROR", filename, sweep_num, f"Phase plane plotting failed: {e}")
        ax.cla()
        _plot_error_message(
            ax, f"Plotting Error:\n{e}", f"Phase Plot Error: {filename}"
        )


def _prepare_phase_plot_data(
    analysis_df: Optional[pd.DataFrame],
    abf_obj: Optional[pyabf.ABF],
    filename: str,
    current_col: str,
) -> Tuple[
    Optional[np.ndarray], Optional[np.ndarray], Optional[int], Optional[float], str
]:
    """
    Calculates data needed for the phase plot (V, dV/dt) for a sweep near 2x rheobase.

    Returns:
        Tuple: (voltage_mV, dvdt_mV_ms, target_sweep_num, target_current_pA, status_suffix)
               Returns (None, None, None, None, error_message) on failure.
    """
    target_sweep_num: Optional[int] = None
    target_current_pA: Optional[float] = None
    phase_v: Optional[np.ndarray] = None
    phase_dvdt: Optional[np.ndarray] = None
    phase_title_suffix: str = "N/A"  # Default suffix

    if not isinstance(abf_obj, pyabf.ABF):
        return None, None, None, None, "ABF Not Loaded"
    if not isinstance(analysis_df, pd.DataFrame) or analysis_df.empty:
        return None, None, None, None, "Analysis Failed"
    if (
        current_col not in analysis_df.columns
        or "spike_count" not in analysis_df.columns
    ):
        return None, None, None, None, "Required Columns Missing"
    if not pd.api.types.is_numeric_dtype(
        analysis_df[current_col]
    ) or not pd.api.types.is_numeric_dtype(analysis_df["spike_count"]):
        return None, None, None, None, "Non-numeric Data"

    try:
        # Find Rheobase (first sweep with >= 1 spike and positive current)
        # Ensure we handle potential NaN/inf in current/spike_count gracefully
        spiking_sweeps = analysis_df[
            (analysis_df["spike_count"].fillna(0) >= 1)
            & (analysis_df[current_col].fillna(-np.inf) > 0)
            & np.isfinite(analysis_df[current_col])
        ]

        if not spiking_sweeps.empty:
            rheobase_row = spiking_sweeps.loc[spiking_sweeps[current_col].idxmin()]
            rheobase_current = rheobase_row[current_col]
            target_current = 2 * rheobase_current
            helper._log_message(
                "DEBUG",
                filename,
                None,
                f"Rheobase found: {rheobase_current:.2f} pA. Target 2x: {target_current:.2f} pA",
            )

            # Find sweep closest to 2x rheobase
            # Ensure we only consider sweeps with valid finite current values
            valid_current_df = analysis_df.dropna(subset=[current_col])
            valid_current_df = valid_current_df[
                np.isfinite(valid_current_df[current_col])
            ]

            if not valid_current_df.empty:
                closest_idx = (
                    (valid_current_df[current_col] - target_current).abs().idxmin()
                )
                closest_sweep_row = valid_current_df.loc[closest_idx]
                if "sweep" not in closest_sweep_row:
                    return None, None, None, None, "Sweep Column Missing"

                target_sweep_num = int(closest_sweep_row["sweep"])
                target_current_pA = closest_sweep_row[current_col]
                helper._log_message(
                    "DEBUG",
                    filename,
                    None,
                    f"Closest sweep to 2x rheo: {target_sweep_num} ({target_current_pA:.2f} pA)",
                )

                # Get data for the target sweep
                try:
                    abf_obj.setSweep(target_sweep_num)
                    phase_v = abf_obj.sweepY
                    time_s = abf_obj.sweepX
                    if (
                        isinstance(phase_v, np.ndarray)
                        and len(phase_v) > 1
                        and isinstance(time_s, np.ndarray)
                        and len(time_s) > 1
                        and len(phase_v) == len(time_s)
                    ):
                        # Calculate dV/dt (mV/ms)
                        with warnings.catch_warnings():  # Suppress potential gradient warnings
                            warnings.simplefilter("ignore")
                            phase_dvdt = np.gradient(phase_v, time_s * 1000.0)
                        phase_title_suffix = ""  # Success! Clear the suffix
                    else:
                        phase_title_suffix = "Sweep Data Invalid"
                        helper._log_message(
                            "WARN",
                            filename,
                            target_sweep_num,
                            "Invalid V/T data for phase plot gradient.",
                        )
                except IndexError:
                    phase_title_suffix = f"Sweep Index Error: {target_sweep_num}"
                    helper._log_message("ERROR", filename, None, phase_title_suffix)
                except Exception as e_sweep:
                    phase_title_suffix = f"Sweep Load Error: {e_sweep}"
                    helper._log_message(
                        "ERROR", filename, target_sweep_num, phase_title_suffix
                    )

            else:
                phase_title_suffix = "No Valid Current Data"
                helper._log_message(
                    "WARN",
                    filename,
                    None,
                    "No sweeps with valid current found for phase plot.",
                )

        else:
            phase_title_suffix = "Rheobase Not Found"
            helper._log_message(
                "WARN",
                filename,
                None,
                "Rheobase not found (no spiking sweeps with positive current).",
            )

    except Exception as e:
        phase_title_suffix = f"Phase Calc Error: {e}"
        helper._log_message(
            "ERROR",
            filename,
            target_sweep_num,  # Might be None if error happened early
            f"Error calculating phase plot data: {e}",
        )
        # Reset data in case of error
        phase_v, phase_dvdt, target_sweep_num, target_current_pA = (
            None,
            None,
            None,
            None,
        )

    return phase_v, phase_dvdt, target_sweep_num, target_current_pA, phase_title_suffix


def _generate_summary_plots_for_file(
    result_data: Dict[str, Any],
    axes: Sequence[plt.Axes],  # Expects 3 axes: [raw_ax, sc_ax, phase_ax]
    current_col: str = constants.CURRENT_COL_NAME,
) -> None:
    """
    Generates the standard set of summary plots (raw, SC, phase) onto provided axes.

    Args:
        result_data: Dictionary containing analysis results for one file
                     (must include 'original_filename', 'abf_object', 'analysis_df', 'load_error').
        axes: A sequence (list or tuple) of 3 Matplotlib Axes objects.
        current_col: Name of the column containing current step values.
    """
    if len(axes) != 3:
        helper._log_message(
            "ERROR",
            result_data.get("original_filename", "UnknownFile"),
            None,
            "_generate_summary_plots_for_file expects 3 axes.",
        )
        return

    filename = result_data.get("original_filename", "Unknown Filename")
    abf_obj = result_data.get("abf_object")
    load_err = result_data.get("load_error")
    analysis_df = result_data.get("analysis_df")  # Can be None or empty DataFrame

    raw_ax, sc_ax, phase_ax = axes

    # 1. Raw Trace Plot
    try:
        plot_raw_traces(
            abf_obj,
            filename,
            ax=raw_ax,
            load_error=load_err,
            title_prefix="Raw",  # Shorter title for multi-plots
        )
    except Exception as e_plot:
        helper._log_message("ERROR", filename, None, f"Summary Raw Plot Error: {e_plot}")
        _plot_error_message(
            raw_ax, f"Raw Plot Error:\n{e_plot}", f"Raw Error: {filename}"
        )

    # 2. Spike Count vs Current Plot
    try:
        # Handle case where analysis failed before calling plot function
        if load_err:
            _plot_error_message(
                sc_ax, f"Load Error:\n{load_err}", "Spike Count vs Current"
            )
        elif not isinstance(analysis_df, pd.DataFrame) or analysis_df.empty:
            _plot_error_message(
                sc_ax, "Analysis skipped\nor failed.", "Spike Count vs Current"
            )
        else:
            plot_feature_vs_current(
                analysis_df,
                "spike_count",
                current_col,
                filename,
                abf_obj,  # Pass for units
                ax=sc_ax,
            )
    except Exception as e_plot:
        helper._log_message("ERROR", filename, None, f"Summary SC Plot Error: {e_plot}")
        _plot_error_message(
            sc_ax, f"SC Plot Error:\n{e_plot}", "Spike Count vs Current"
        )

    # 3. Phase Plane Plot
    phase_v, phase_dvdt, target_sweep, target_current, suffix = (
        None,
        None,
        None,
        None,
        "Prep Error",
    )
    try:
        # Prepare data first (handles load errors, analysis failure internally)
        phase_v, phase_dvdt, target_sweep, target_current, suffix = (
            _prepare_phase_plot_data(analysis_df, abf_obj, filename, current_col)
        )

        # Plot the phase plane using prepared data
        plot_phase_plane(
            phase_v,
            phase_dvdt,
            filename,
            target_sweep,
            target_current,
            suffix,  # Contains status/error message from prep function
            ax=phase_ax,
        )
    except Exception as e_plot:
        helper._log_message(
            "ERROR", filename, target_sweep, f"Summary Phase Plot Error: {e_plot}"
        )
        _plot_error_message(phase_ax, f"Phase Plot Error:\n{e_plot}", "Phase Plane")
