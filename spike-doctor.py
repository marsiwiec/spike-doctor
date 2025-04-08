import io
import base64
import traceback
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any, Sequence
from io import BytesIO
import shiny
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import pyabf
import efel
from shiny import App, render, ui, reactive, req

# ==============================================================================
# Configuration Constants
# ==============================================================================

# --- Analysis Settings ---
# IMPORTANT: Set this index based on your experimental protocol's stimulus epoch.
# This index refers to the epoch within abf.sweepEpochs that defines the main
# stimulus start and end times used for eFEL and other calculations.
# Check your ABF file's header in Clampfit or pyABF to determine the correct index.
# Common indices: 0 (first epoch), 1, 2, etc.

# --- eFEL Specific Settings ---
DEFAULT_EFEL_FEATURES: List[str] = [
    "spike_count",
    "voltage_base",
    "steady_state_voltage_stimend",
    "voltage_deflection",
    "ohmic_input_resistance",
    "mean_frequency",
    "inv_last_ISI",  # Inverse of the last inter-spike interval (instantaneous freq)
    "time_to_first_spike",
    "decay_time_constant_after_stim",
]
# Features needed internally even if not selected by the user
REQUIRED_INTERNAL_EFEL_FEATURES: List[str] = [
    "spike_count",
    "voltage_base",
    "decay_time_constant_after_stim",
    "ohmic_input_resistance",
]

# --- Window Definitions (in milliseconds from start of sweep) ---
# Window for calculating the AVERAGE stimulus current injected (used by eFEL trace dict)
EFEL_AVG_CURRENT_WIN_START_MS: float = 400.0
EFEL_AVG_CURRENT_WIN_END_MS: float = 500.0

# --- Plotting & Debugging ---
MAX_RAW_PLOT_SWEEPS: int = 100  # Max sweeps to overlay on the raw trace plot
CURRENT_COL_NAME = "current_step_pA"

# ==============================================================================
# Helper Functions
# ==============================================================================


def _log_message(level: str, abf_id: str, sweep_num: Optional[int], message: str):
    """Standardized logging output."""
    prefix = f"{level.upper()}({abf_id}"
    if sweep_num is not None:
        prefix += f", Sw {sweep_num}"
    prefix += ")"
    print(f"{prefix}: {message}")


def fig_to_src(fig: Optional[plt.Figure]) -> Optional[str]:
    """Converts a Matplotlib figure to a base64 encoded image source."""
    if fig is None:
        return None
    try:
        with io.BytesIO() as buf:
            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            base64_str = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)  # Close the figure to free memory
        return f"data:image/png;base64,{base64_str}"
    except Exception as e:
        _log_message("ERROR", "FigConv", None, f"Figure conversion failed: {e}")
        plt.close(fig)
        return None


def get_abf_info_text(abf: Optional[pyabf.ABF], filename: str) -> str:
    """Generates a formatted string with ABF file metadata."""
    if abf is None:
        return f"File: {filename}\nError: Could not load ABF object."
    try:
        info_lines = [
            f"File: {filename}",
            f"Protocol: {getattr(abf, 'protocol', 'N/A')}",
            f"ABF Version: {getattr(abf, 'abfVersionString', 'N/A')}",
        ]
        # Calculate duration robustly
        duration = getattr(abf, "abfLengthSec", None)
        rate = getattr(abf, "dataRate", 0)
        points = getattr(abf, "dataPointCount", 0)
        if duration is None and rate > 0 and points > 0:
            duration = points / rate
            info_lines.append(f"Duration (Calculated): {duration:.2f} s")
        elif duration is not None:
            info_lines.append(f"Duration: {duration:.2f} s")
        else:
            info_lines.append("Duration: N/A")

        info_lines.extend(
            [
                f"Sample Rate: {rate} Hz",
                f"Channels: {getattr(abf, 'channelCount', 'N/A')}",
                f"Sweeps: {getattr(abf, 'sweepCount', 'N/A')}",
                f"Voltage Units: {getattr(abf, 'sweepUnitsY', '?')}",
                f"Current Units: {getattr(abf, 'sweepUnitsC', '?')}",
            ]
        )

        sweep_points = getattr(abf, "sweepPointCount", 0)
        if getattr(abf, "sweepCount", 0) > 0 and rate > 0 and sweep_points > 0:
            info_lines.append(
                f"Sweep Duration: {sweep_points / rate:.3f} s ({sweep_points} points)"
            )
        else:
            info_lines.append("Sweep Duration: N/A")

        return "\n".join(info_lines)
    except Exception as e:
        return f"Error retrieving ABF info for {filename}: {e}"


def is_current_clamp(abf: Optional[pyabf.ABF]) -> bool:
    """Checks if the ABF file is likely a current clamp recording."""
    if abf is None:
        return False
    try:
        y_units = str(getattr(abf, "sweepUnitsY", "")).lower()
        c_units = str(getattr(abf, "sweepUnitsC", "")).lower()
        # Primary check: Voltage recorded (Y) and Current commanded (C)
        is_volt_y = "v" in y_units
        is_curr_c = "a" in c_units
        # Fallback: Sometimes command channel units might be missing/weird in CC
        return (is_volt_y and is_curr_c) or (is_volt_y and not c_units)
    except Exception:
        return False


def parse_efel_value(
    raw_efel_result: Optional[Dict[str, Any]], feature_key: str
) -> float:
    """
    Safely extracts and converts a single float value from eFEL results.
    Handles None results, empty lists, NaN, and conversion errors.
    """
    if raw_efel_result is None:
        return np.nan
    value = raw_efel_result.get(feature_key)
    if value is None:
        return np.nan
    # eFEL often returns lists, even for single values
    if isinstance(value, (list, np.ndarray)):
        if len(value) == 0:
            return np.nan
        first_val = value[0]
    else:
        first_val = value

    if pd.isna(first_val):
        return np.nan
    try:
        return float(first_val)
    except (TypeError, ValueError):
        return np.nan


def _validate_abf_for_analysis(abf: pyabf.ABF, abf_id_str: str) -> bool:
    """Performs initial checks on the ABF object properties."""
    if not hasattr(abf, "sweepCount") or abf.sweepCount <= 0:
        _log_message("WARN", abf_id_str, None, "No sweeps found.")
        return False
    if not is_current_clamp(abf):
        _log_message(
            "WARN",
            abf_id_str,
            None,
            "File may not be current clamp. Attempting analysis anyway.",
        )

    if not hasattr(abf, "dataRate") or abf.dataRate <= 0:
        _log_message("ERROR", abf_id_str, None, "Invalid data rate.")
        return False
    if not hasattr(abf, "sweepPointCount") or abf.sweepPointCount <= 0:
        _log_message("ERROR", abf_id_str, None, "Invalid sweep point count.")
        return False
    if not hasattr(abf, "dataPointsPerMs") or abf.dataPointsPerMs <= 0:
        _log_message("ERROR", abf_id_str, None, "Invalid data points per ms.")
        return False
    return True


# ==============================================================================
# Plotting Functions
# ==============================================================================


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
            num_to_plot = min(num_sweeps, MAX_RAW_PLOT_SWEEPS)
            for i in range(num_to_plot):
                abf.setSweep(i)
                ax.plot(abf.sweepX, abf.sweepY, lw=0.5, alpha=0.7)
            if num_sweeps > num_to_plot:
                plot_title += f" (First {num_to_plot})"

        plot_title += " (CC)" if is_current_clamp(abf) else " (VC?)"
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
        _log_message("ERROR", filename, None, f"Raw trace plotting failed: {e}")
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
        _log_message(
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
    current_units = "pA"  # Default assumption based on CURRENT_COL_NAME
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
        _log_message("ERROR", filename, None, f"Plotting '{feature_name}' failed: {e}")
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
            _log_message(
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
        _log_message("ERROR", filename, sweep_num, f"Phase plane plotting failed: {e}")
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
            _log_message(
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
                _log_message(
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
                        _log_message(
                            "WARN",
                            filename,
                            target_sweep_num,
                            "Invalid V/T data for phase plot gradient.",
                        )
                except IndexError:
                    phase_title_suffix = f"Sweep Index Error: {target_sweep_num}"
                    _log_message("ERROR", filename, None, phase_title_suffix)
                except Exception as e_sweep:
                    phase_title_suffix = f"Sweep Load Error: {e_sweep}"
                    _log_message(
                        "ERROR", filename, target_sweep_num, phase_title_suffix
                    )

            else:
                phase_title_suffix = "No Valid Current Data"
                _log_message(
                    "WARN",
                    filename,
                    None,
                    "No sweeps with valid current found for phase plot.",
                )

        else:
            phase_title_suffix = "Rheobase Not Found"
            _log_message(
                "WARN",
                filename,
                None,
                "Rheobase not found (no spiking sweeps with positive current).",
            )

    except Exception as e:
        phase_title_suffix = f"Phase Calc Error: {e}"
        _log_message(
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
    current_col: str = CURRENT_COL_NAME,
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
        _log_message(
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
        _log_message("ERROR", filename, None, f"Summary Raw Plot Error: {e_plot}")
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
        _log_message("ERROR", filename, None, f"Summary SC Plot Error: {e_plot}")
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
        _log_message(
            "ERROR", filename, target_sweep, f"Summary Phase Plot Error: {e_plot}"
        )
        _plot_error_message(phase_ax, f"Phase Plot Error:\n{e_plot}", "Phase Plane")


# ==============================================================================
# Core Analysis Function
# ==============================================================================


def run_analysis_on_abf(
    abf: Optional[pyabf.ABF],
    original_filename: str,
    user_selected_features: List[str],
    stimulus_epoch_index: int,
    detection_threshold: float,
    derivative_threshold: float,
    debug_plot: bool = True,  # Renamed from CREATE_DEBUG_PLOTS for clarity
    current_col_name: str = CURRENT_COL_NAME,
) -> Dict[str, Union[Optional[pd.DataFrame], Optional[plt.Figure]]]:
    """
    Analyzes a single ABF file using eFEL and manual calculations.

    Args:
        abf: The loaded pyabf.ABF object.
        original_filename: The original name of the file (for reporting).
        user_selected_features: List of eFEL features requested by the user.
        stimulus_epoch_index: Index for sweepEpochs to define stimulus time.
        detection_threshold: eFEL spike detection voltage threshold (mV).
        derivative_threshold: eFEL spike detection derivative threshold (mV/ms).
        debug_plot: Whether to generate the detailed debug plot figure.
        current_col_name: The name to use for the calculated current column in the output df.


    Returns:
        A dictionary containing:
            'analysis_df': A pandas DataFrame with results per sweep, or None on failure.
            'debug_plot_fig': A Matplotlib Figure object for debugging (if requested), or None.
    """
    analysis_output = {"analysis_df": None, "debug_plot_fig": None}
    abf_id_str = original_filename  # Use filename for logging if ABF object is bad

    if not isinstance(abf, pyabf.ABF):
        _log_message(
            "ERROR",
            abf_id_str,
            None,
            "Invalid or missing ABF object passed to analysis.",
        )
        return analysis_output
    abf_id_str = getattr(abf, "abfID", original_filename)

    # 1. Initial Validation
    if not _validate_abf_for_analysis(abf, abf_id_str):
        return analysis_output


    # 4. Determine Current Unit Conversion Factor (raw units -> pA)
    current_unit_factor = 1.0  # Assume pA by default
    current_unit_raw = getattr(abf, "sweepUnitsC", "").lower()
    output_plot_unit = "pA"  # For debug plot label
    if "na" in current_unit_raw:
        current_unit_factor = 1000.0  # nA to pA
        _log_message(
            "DEBUG", abf_id_str, None, "Detected nA command units. Factor = 1000."
        )
    elif not current_unit_raw or "pa" not in current_unit_raw:
        output_plot_unit = f"{current_unit_raw or '?'} (Assumed pA)"
        _log_message(
            "WARN",
            abf_id_str,
            None,
            f"Command units are '{current_unit_raw}'. Assuming pA.",
        )

    # Prepare eFEL
    all_efel_features_needed = list(
        set(user_selected_features) | set(REQUIRED_INTERNAL_EFEL_FEATURES)
    )
    try:
        efel.reset()
        efel.set_setting("strict_stiminterval", True)
        efel.set_setting("Threshold", detection_threshold)
        efel.set_setting("DerivativeThreshold", derivative_threshold)
        _log_message(
            "DEBUG",
            abf_id_str,
            None,
            f"Configured eFEL. Requesting {len(all_efel_features_needed)} features.",
        )
    except Exception as e:
        _log_message(
            "ERROR",
            abf_id_str,
            None,
            f"Failed to configure eFEL settings: {e}. Aborting.",
        )
        return analysis_output

    # Sweep Loop and Analysis
    sweep_results_list = []
    debug_plot_generated = False  # Flag to ensure only one debug plot per file
    # Choose a sweep for debug plot: prefer middle, but ensure it's a valid index
    middle_sweep_idx_in_list = abf.sweepCount // 2
    middle_sweep_index_for_debug = (
        abf.sweepList[middle_sweep_idx_in_list] if abf.sweepList else 0
    )

    try:
        for sweep_num in abf.sweepList:
            abf.setSweep(sweep_num)


            # --- Calculate Average Stimulus Current (for eFEL trace dict, in nA) ---
            stimulus_current_for_efel_nA = abf.sweepEpochs.levels[stimulus_epoch_index] / 1000

            trace_for_efel = {
                "T": abf.sweepX * 1000.0,
                "V": abf.sweepY,
                "stim_start": [abf.sweepEpochs.p1s[stimulus_epoch_index] * abf.dataSecPerPoint * 1000.0],
                "stim_end": [abf.sweepEpochs.p2s[stimulus_epoch_index] * abf.dataSecPerPoint * 1000.0],
                "stimulus_current": [float(stimulus_current_for_efel_nA) * current_unit_factor],
            }

            # --- Run eFEL ---
            efel_results_raw = None
            efel_error = None
            efel_warnings = []
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter(
                    "always", RuntimeWarning
                )  # Catch runtime warnings
                warnings.simplefilter(
                    "ignore", DeprecationWarning
                )  # Ignore numpy/efel deprecations
                try:
                    # Wrap eFEL call to catch potential exceptions more broadly
                    efel_results_list = efel.get_feature_values(
                        [trace_for_efel], all_efel_features_needed, raise_warnings=True
                    )
                    if efel_results_list:  # Should return a list containing one dict
                        efel_results_raw = efel_results_list[0]
                    else:  # Should not happen if raise_error=True, but check anyway
                        efel_error = ValueError(
                            "eFEL returned empty list unexpectedly."
                        )

                except Exception as e:
                    efel_error = e  # Store error to handle after warnings

                # Store relevant warnings (filter out known safe ones if needed)
                efel_warnings = [
                    str(w.message)
                    for w in caught_warnings
                    if issubclass(w.category, RuntimeWarning)
                ]

            if efel_error:
                # Log error traceback for more details if possible
                tb_str = traceback.format_exc()
                _log_message(
                    "ERROR",
                    abf_id_str,
                    sweep_num,
                    f"eFEL feature extraction failed: {efel_error}\nTraceback:\n{tb_str}",
                )
                # Set raw results to None so parsing yields NaNs
                efel_results_raw = None
            if efel_warnings:
                # Log only the first warning to avoid spamming console
                summary_warning = (
                    f"{efel_warnings[0]}{'...' if len(efel_warnings) > 1 else ''}"
                )
                _log_message(
                    "WARN", abf_id_str, sweep_num, f"eFEL warnings: {summary_warning}"
                )

            # --- Parse eFEL Results Safely ---
            efel_results_parsed = {
                feat: parse_efel_value(efel_results_raw, feat)
                for feat in all_efel_features_needed
            }

            # --- Manual Calculations (Cm) ---

            Cm_manual_pF = np.nan
            tau_efel_ms = efel_results_parsed.get("decay_time_constant_after_stim", np.nan)
            R_in_MOhm = efel_results_parsed.get("ohmic_input_resistance", np.nan)

            
            # Ensure spike_count is a valid integer (0 if NaN/None)
            spike_count = efel_results_parsed.get("spike_count", np.nan)
            spike_count = 0 if pd.isna(spike_count) else int(spike_count)

            V_base_efel = efel_results_parsed.get("voltage_base", np.nan)

            # Calculate Cm only if Tau and Rin are valid and positive
            if (
                pd.notna(tau_efel_ms)
                and tau_efel_ms > 0
                and pd.notna(R_in_MOhm)
                and R_in_MOhm > 0
                and np.isfinite(tau_efel_ms)
                and np.isfinite(R_in_MOhm)
            ):
                try:
                    # Cm (pF) = Tau (ms) / Rin (MOhm) * 1000
                    Cm_manual_pF = (tau_efel_ms / R_in_MOhm) * 1000.0
                    if not np.isfinite(Cm_manual_pF):
                        Cm_manual_pF = np.nan
                except (ZeroDivisionError, FloatingPointError) as div_err:
                    # Should be caught by Rin>0 check, but handle defensively
                    _log_message(
                        "WARN",
                        abf_id_str,
                        sweep_num,
                        f"Division error during Cm calc: {div_err}",
                    )
                    Cm_manual_pF = np.nan
                except Exception as cm_err:
                    _log_message(
                        "WARN",
                        abf_id_str,
                        sweep_num,
                        f"Manual Cm calculation failed: {cm_err}",
                    )
                    Cm_manual_pF = np.nan

            # --- Assemble Sweep Results ---
            sweep_data = {
                "filename": original_filename,
                "sweep": sweep_num,
                CURRENT_COL_NAME: trace_for_efel["stimulus_current"][0] * 1000,
                "capacitance_pF": Cm_manual_pF,
                **{ 
                    feat: efel_results_parsed.get(feat, np.nan)
                    for feat in user_selected_features
                },
            }

            # --- Sanitize results based on spike count ---
            if spike_count <= 1:
                for feature in sweep_data:
                    # Invalidate freq/ISI features if 0 or 1 spike
                    if "frequency" in feature.lower() or "isi" in feature.lower():
                        sweep_data[feature] = np.nan
            if spike_count == 0:
                # Invalidate time_to features if no spikes
                for feature in sweep_data:
                    if "time_to_" in feature.lower() or "latency" in feature.lower():
                        sweep_data[feature] = np.nan
            if spike_count > 0:
                sweep_data["capacitance_pF"] = np.nan


            sweep_results_list.append(sweep_data)

            # --- Generate Debug Plot (only once per file, for a designated sweep) ---
            # Check flag and if current sweep is the target debug sweep
            if (
                debug_plot
                and sweep_num == middle_sweep_index_for_debug
                and not debug_plot_generated
            ):
                fig_debug = None  # Initialize figure variable
                try:
                    fig_debug, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
                    fig_debug.set_layout_engine("tight")  # Use tight layout

                    # --- Voltage Trace ---
                    axs[0].plot(
                        abf.sweepX,
                        abf.sweepY,
                        color="black",
                        lw=0.7,
                        label=f"Sweep {sweep_num}",
                    )
                    # Highlight eFEL stimulus window
                    axs[0].axvspan(
                        xmin = trace_for_efel["stim_start"][0] / 1000.0,
                        xmax = trace_for_efel["stim_end"][0] / 1000.0,
                        color="salmon",  # Use a different color
                        alpha=0.2,
                        zorder=-10,  # Place behind data
                    )
                    # Plot V_base from eFEL (if valid)
                    if pd.notna(V_base_efel):
                        axs[0].axhline(
                            V_base_efel,
                            color="darkorchid",  # Different color
                            linestyle=":",
                            lw=1.5,
                            label=f"V_base (eFEL): {V_base_efel:.1f} mV",
                            zorder=0,
                        )

                    axs[0].set_ylabel(f"Voltage ({getattr(abf, 'sweepUnitsY', 'mV')})")
                    axs[0].set_title(
                        f"Debug Plot: {abf_id_str} - Sweep {sweep_num}", fontsize=10
                    )
                    axs[0].legend(fontsize=7, loc="best")
                    axs[0].grid(True, linestyle=":", alpha=0.5)

                    # --- Current Trace ---
                    if hasattr(abf, "sweepC") and isinstance(abf.sweepC, np.ndarray):
                        axs[1].plot(
                            abf.sweepX,
                            abf.sweepC * current_unit_factor,  # Plot in pA
                            color="royalblue",  # Different blue
                            lw=0.7,
                        )
                    else:
                        axs[1].text(
                            0.5,
                            0.5,
                            "Sweep Command (sweepC)\nNot Available",
                            ha="center",
                            va="center",
                            color="red",
                            transform=axs[1].transAxes,
                        )

                    # Highlight eFEL stimulus window
                    axs[1].axvspan(
                        xmin = trace_for_efel["stim_start"][0] / 1000.0,
                        xmax = trace_for_efel["stim_end"][0] / 1000.0,
                        color="salmon",
                        alpha=0.2,
                        zorder=-10,
                    )
                    axs[1].grid(True, linestyle=":", alpha=0.5)
                    axs[1].set_ylabel(f"Current ({getattr(abf, 'sweepUnitsC', 'pA')})")
                    axs[1].set_xlabel("Time (s)")

                    # fig_debug.tight_layout() # Use set_layout_engine instead
                    analysis_output["debug_plot_fig"] = fig_debug
                    debug_plot_generated = True  # Set flag
                    _log_message(
                        "DEBUG", abf_id_str, sweep_num, "Debug plot generated."
                    )

                except Exception as plot_err:
                    # Log error but don't stop analysis
                    _log_message(
                        "ERROR",
                        abf_id_str,
                        sweep_num,
                        f"Debug plot generation failed: {plot_err}\n{traceback.format_exc()}",
                    )
                    analysis_output["debug_plot_fig"] = (
                        None  # Ensure it's None on error
                    )
                    if fig_debug is not None:
                        plt.close(fig_debug)  # Close figure if created but failed later

        # --- End of Sweep Loop ---

    except Exception as loop_err:
        _log_message(
            "ERROR",
            abf_id_str,
            None,
            f"Critical error during sweep processing: {loop_err}",
        )
        traceback.print_exc()
        # Return partial results if any were collected before the error
        if sweep_results_list:
            _log_message(
                "WARN",
                abf_id_str,
                None,
                "Returning partial results due to error in loop.",
            )
        else:
            # If loop failed early, ensure output df is None
            analysis_output["analysis_df"] = None
            return analysis_output  # Return immediately, no further processing

    # 7. Final DataFrame Assembly
    if not sweep_results_list:
        _log_message(
            "WARN",
            abf_id_str,
            None,
            "Analysis finished, but no sweep results were generated.",
        )
        # Return an empty DataFrame instead of None if analysis ran but yielded no rows
        analysis_output["analysis_df"] = pd.DataFrame()
        return analysis_output

    try:
        analysis_df = pd.DataFrame(sweep_results_list)

        # Define standard columns (ensure current column name matches)
        standard_cols = [
            "filename",
            "sweep",
            current_col_name,
            "input_resistance_Mohm",
            "time_constant_ms",
            "capacitance_pF",
        ]

        # Get list of eFEL features actually present in the results (requested by user)
        efel_cols_present = sorted(
            [f for f in user_selected_features if f in analysis_df.columns]
        )

        # Combine and order columns: standard first, then selected eFEL features
        final_ordered_cols = standard_cols + efel_cols_present
        # Ensure only existing columns are used for reindexing to avoid KeyErrors
        final_ordered_cols = [
            col for col in final_ordered_cols if col in analysis_df.columns
        ]

        analysis_df = analysis_df.reindex(columns=final_ordered_cols)

        # Sort by filename and sweep number
        analysis_df = analysis_df.sort_values(by=["filename", "sweep"]).reset_index(
            drop=True
        )

        analysis_output["analysis_df"] = analysis_df
        _log_message(
            "DEBUG",
            abf_id_str,
            None,
            f"Analysis complete. DataFrame shape: {analysis_df.shape}",
        )

    except Exception as df_err:
        _log_message(
            "ERROR", abf_id_str, None, f"Failed to assemble final DataFrame: {df_err}"
        )
        # Set df to None if assembly fails
        analysis_output["analysis_df"] = None

    return analysis_output


# ==============================================================================
# Shiny UI Definition
# ==============================================================================
try:
    AVAILABLE_EFEL_FEATURES = sorted(efel.get_feature_names())
except Exception as e:
    print(f"Warning: Could not dynamically get eFEL features: {e}. Using defaults.")
    AVAILABLE_EFEL_FEATURES = sorted(
        DEFAULT_EFEL_FEATURES + REQUIRED_INTERNAL_EFEL_FEATURES
    )

VALID_DEFAULT_FEATURES = [
    f for f in DEFAULT_EFEL_FEATURES if f in AVAILABLE_EFEL_FEATURES
]

app_ui = ui.page_fluid(
    ui.tags.head(ui.tags.title("Spike Doctor")),
    ui.layout_sidebar(
        ui.sidebar(
            ui.h4("Spike Doctor"),
            ui.h5("I'm gonna diagnose your spikes!"),
            ui.input_file(
                "abf_files", "Select ABF File(s):", accept=[".abf"], multiple=True
            ),
            ui.hr(),
            ui.download_button("download_analysis_csv", "Download Results (CSV)"),
            ui.download_button("download_analysis_excel", "Download Results (Excel)"),
            ui.download_button("download_plots_pdf", "Download Summary Plots (PDF)"),
            ui.hr(),
            ui.h5("Analysis Parameters"),
            ui.tags.b("Stimulus Definition:"),
            ui.input_numeric(
                "stimulus_epoch_index",
                "Stimulus Epoch Index (0-based):",
                value=2,
                min=0,
                step=1,
            ),
            # ui.tags.ul(
            #     ui.tags.li("Method: From ABF Header (SweepEpochs) using specified index."),
            #     ui.tags.li(
            #         "Important: Verify the correct index for your protocol in Clampfit/pyABF."
            #     ),
            # ),
            ui.tags.b("Spike Detection:"),
            ui.input_numeric(
                "detection_threshold",
                "Detection Threshold (mV):",
                value=-20,
                step=1,
            ),
            ui.input_numeric(
                "derivative_threshold",
                "Derivative Threshold (mV/ms):",
                value=10,
                step=1,
            ),
            # ui.tags.b("Current Input (for eFEL):"),
            # ui.tags.ul(
            #     ui.tags.li(
            #         f"Average current during {EFEL_AVG_CURRENT_WIN_START_MS:.0f}-{EFEL_AVG_CURRENT_WIN_END_MS:.0f} ms"
            #     ),
            #     ui.tags.li(f"(Passed as 'stimulus_current' in nA to eFEL)"),
            # ),
            # ui.tags.b("Manual Calculations (Rin, Cm):"),
            # ui.tags.ul(
            #     ui.tags.li(
            #         f"Analysis Window: {MANUAL_CALC_WIN_START_MS:.0f}-{MANUAL_CALC_WIN_END_MS:.0f} ms"
            #     ),
            #     ui.tags.li("Performed only if eFEL spike_count = 0"),
            #     ui.tags.li("Uses eFEL V_base and Tau_decay"),
            #     ui.tags.li("Outputs: input_resistance_Mohm, capacitance_pF"),
            # ),
            ui.tags.b("Options:"),
            ui.input_checkbox(
                "debug_plots",
                "Generate Debug Plots",
                True,
            ),
            ui.h5("eFEL Features to Calculate:"),
            ui.input_checkbox_group(
                "selected_efel_features",
                label=None,
                choices=AVAILABLE_EFEL_FEATURES,
                selected=VALID_DEFAULT_FEATURES,
            ),
            width=380,
        ),
        ui.navset_tab(
            ui.nav_panel(
                "Summary Plots",
                ui.h3("Analysis Summary"),
                ui.output_text_verbatim("analysis_summary_text"),
                ui.hr(),
                ui.h4("File Plots"),
                ui.output_ui("dynamic_summary_plots_ui"),
            ),
            ui.nav_panel(
                "Detailed Results Table",
                ui.h4("Combined Analysis Data"),
                ui.output_data_frame("analysis_data_table"),
            ),
            ui.nav_panel(
                "Debug Plots",
                ui.h4("Debug Plots (Middle Sweep)"),
                ui.help_text(
                    "Generated if 'Generate Debug Plots' is enabled. Shows details of analysis steps for one sweep per file."
                ),
                ui.output_ui("dynamic_debug_plots_ui"),
            ),
        ),
    ),
)


# ==============================================================================
# Shiny Server Logic
# ==============================================================================
def server(input: shiny.Inputs, output: shiny.Outputs, session: shiny.Session):

    loaded_abf_data = reactive.Value([])

    @reactive.Effect
    @reactive.event(input.abf_files)
    def _load_abf_files():
        """Loads ABF files selected by the user."""
        file_infos = input.abf_files()
        if not file_infos:
            loaded_abf_data.set([])
            _log_message("INFO", "App", None, "File selection cleared.")
            return

        data_list = []
        num_files = len(file_infos)
        _log_message("INFO", "App", None, f"Loading {num_files} ABF file(s)...")
        with ui.Progress(min=0, max=num_files) as p:
            p.set(message="Loading ABF files", detail="Starting...")
            for i, file_info in enumerate(file_infos):
                filename = file_info["name"]
                filepath = Path(file_info["datapath"]).resolve()
                p.set(i, detail=f"Loading {filename}...")
                abf_obj, error_msg = None, None
                try:
                    # Ensure warnings during loading are not treated as errors here
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        abf_obj = pyabf.ABF(str(filepath), loadData=True)
                except FileNotFoundError:
                    error_msg = "File not found at temporary path."
                    _log_message(
                        "ERROR", filename, None, f"{error_msg} Path: {filepath}"
                    )
                except Exception as e:
                    error_msg = f"Failed to load: {e}"
                    _log_message(
                        "ERROR",
                        filename,
                        None,
                        f"{error_msg}\n{traceback.format_exc()}",
                    )

                data_list.append(
                    {
                        "original_filename": filename,
                        "filepath": str(filepath),
                        "abf_object": abf_obj,
                        "load_error": error_msg,
                    }
                )
            p.set(num_files, detail="Loading complete.")

        loaded_abf_data.set(data_list)
        _log_message("INFO", "App", None, f"Finished loading {len(data_list)} files.")

    analysis_results_list = reactive.Calc(
        lambda: [
            {
                **file_data,
                **run_analysis_on_abf(
                    abf=file_data.get("abf_object"),
                    original_filename=file_data.get("original_filename"),
                    user_selected_features=input.selected_efel_features(),
                    stimulus_epoch_index=input.stimulus_epoch_index(),
                    detection_threshold=input.detection_threshold(),
                    derivative_threshold=input.derivative_threshold(),
                    debug_plot=input.debug_plots(),
                    current_col_name=CURRENT_COL_NAME,
                ),
            }
            for file_data in loaded_abf_data()
        ]
    )

    @reactive.Calc
    def combined_analysis_df() -> pd.DataFrame:
        """
        Combines analysis results from individual files into a single DataFrame.
        Returns an empty DataFrame if no valid results exist or on error.
        """
        results_list = analysis_results_list()  # Get the list of result dicts
        _log_message(
            "DEBUG",
            "App",
            None,
            f"Attempting to combine results from {len(results_list)} files.",
        )

        valid_dfs = [
            r.get("analysis_df")
            for r in results_list
            if isinstance(r.get("analysis_df"), pd.DataFrame)
            and not r["analysis_df"].empty
        ]

        if not valid_dfs:
            _log_message(
                "WARN", "App", None, "No valid analysis DataFrames to combine."
            )
            return pd.DataFrame()
        try:
            combined_df = pd.concat(valid_dfs, ignore_index=True, sort=False)
            _log_message(
                "DEBUG", "App", None, f"Combined DataFrame shape: {combined_df.shape}"
            )
            return combined_df
        except Exception as e:
            _log_message("ERROR", "App", None, f"Failed to concatenate DataFrames: {e}")
            traceback.print_exc()
            return pd.DataFrame()

    @output
    @render.text
    def analysis_summary_text():
        """Displays a summary of loaded files and analysis status."""
        results = analysis_results_list()
        num_total = len(results)
        if num_total == 0:
            return "1. Upload one or more ABF files.\n2. Adjust parameters if needed.\n3. View results in tabs."

        num_load_ok = sum(
            1 for r in results if r.get("abf_object") and not r.get("load_error")
        )
        num_load_err = sum(1 for r in results if r.get("load_error"))
        num_analyzed_ok = sum(
            1
            for r in results
            if isinstance(r.get("analysis_df"), pd.DataFrame)
            and not r["analysis_df"].empty
        )
        num_analysis_failed = num_total - num_analyzed_ok - num_load_err

        summary = (
            f"--- File Status ---\n"
            f"Total Files Attempted: {num_total}\n"
            f"Successfully Loaded: {num_load_ok}\n"
            f"Load Errors: {num_load_err}\n"
            f"Successfully Analyzed: {num_analyzed_ok}\n"
            f"Analysis Skipped/Failed: {num_analysis_failed}\n"
            f"---\n"
            f"--- Current Settings ---\n"
            f"Stimulus Epoch Index Used: {input.stimulus_epoch_index()}\n"  # Display used index
            f"Spike V Threshold: {input.detection_threshold()} mV\n"
            f"Spike dV/dt Threshold: {input.derivative_threshold()} mV/ms\n"
            f"Debug Plots Enabled: {'Yes' if input.debug_plots() else 'No'}\n"
            f"# eFEL Features Selected: {len(input.selected_efel_features())}\n"
            f"---\n"
        )

        # Show info for the first successfully loaded file
        first_ok_file_data = next((r for r in results if r.get("abf_object")), None)
        if first_ok_file_data:
            summary += f"First File Info ({first_ok_file_data['original_filename']}):\n"
            summary += get_abf_info_text(
                first_ok_file_data["abf_object"],
                first_ok_file_data["original_filename"],
            )
            summary += "\n---\n"

        first_analyzed_data = next(
            (
                r
                for r in results
                if isinstance(r.get("analysis_df"), pd.DataFrame)
                and not r["analysis_df"].empty
            ),
            None,
        )
        if first_analyzed_data:
            cols = list(first_analyzed_data["analysis_df"].columns)
            col_str = ", ".join(cols)
            max_line_len = 70
            wrapped_cols = "\n".join(
                col_str[i : i + max_line_len]
                for i in range(0, len(col_str), max_line_len)
            )
            summary += f"--- Output Columns ({first_analyzed_data['original_filename']}) ---\n{wrapped_cols}\n"
        elif num_load_ok > 0:
            summary += "--- Output Columns ---\n(Waiting for successful analysis...)\n"
        else:
            summary += "--- Output Columns ---\n(Waiting for files to load...)\n"

        return summary

    @output
    @render.ui
    def dynamic_summary_plots_ui():
        """Dynamically generates UI for raw trace, spike count, and phase-plane plots."""
        results = analysis_results_list()
        req(results)

        ui_elements = []
        if not results:
            return ui.p("No files loaded or analyzed yet.")

        _log_message(
            "DEBUG", "UI", None, f"Generating summary plot UI for {len(results)} files."
        )

        for i, result_data in enumerate(results):
            filename = result_data.get("original_filename", f"File {i+1}")
            plot_fig = None
            try:
                plot_fig, axes = plt.subplots(
                    1, 3, figsize=(12, 3.5)
                )
                plot_fig.set_layout_engine("tight")

                # Use the centralized plotting function to draw on these axes
                _generate_summary_plots_for_file(
                    result_data, axes, current_col=CURRENT_COL_NAME
                )

                plot_src = fig_to_src(plot_fig)

                if plot_src:
                    file_ui = ui.div(
                        ui.hr() if i > 0 else None,
                        ui.h5(filename),
                        ui.row(
                            ui.column(
                                12,
                                ui.img(
                                    src=plot_src,
                                    style="width: 100%; height: auto; max-width: 1400px; border: 1px solid #ddd;",
                                ),
                            )
                        ),
                    )
                    ui_elements.append(file_ui)
                else:
                    # Handle case where figure conversion failed
                    _log_message(
                        "WARN",
                        filename,
                        None,
                        "Figure conversion to src failed for UI summary plot.",
                    )
                    ui_elements.append(
                        ui.div(
                            ui.hr() if i > 0 else None,
                            ui.h5(filename),
                            ui.p(
                                f"Could not generate summary plot image for {filename}."
                            ),
                        )
                    )

            except Exception as e_ui_plot:
                _log_message(
                    "ERROR",
                    filename,
                    None,
                    f"Failed to generate UI summary plot figure for {filename}: {e_ui_plot}",
                )
                traceback.print_exc()
                if plot_fig:  # Ensure figure is closed on error
                    plt.close(plot_fig)
                # Add an error message to the UI for this file
                ui_elements.append(
                    ui.div(
                        ui.hr() if i > 0 else None,
                        ui.h5(filename),
                        ui.p(
                            f"Error generating plots: {e_ui_plot}", style="color: red;"
                        ),
                    )
                )

        return (
            ui.TagList(*ui_elements)
            if ui_elements
            else ui.p("Processing files or no plots generated.")
        )

    @output
    @render.ui
    def dynamic_debug_plots_ui():
        """Dynamically generates UI for the debug plots."""
        if not input.debug_plots():
            return ui.tags.p("Debug plots are disabled in the configuration.")

        results = analysis_results_list()
        req(results)

        ui_elements = []
        for result_data in results:
            filename = result_data["original_filename"]
            debug_fig = result_data.get("debug_plot_fig")

            if isinstance(debug_fig, plt.Figure):
                plots_found = True
                debug_plot_src = fig_to_src(debug_fig)
                if debug_plot_src:
                    file_ui = ui.div(
                        ui.h5(f"Debug Details: {filename}"),
                        ui.row(
                            ui.column(
                                12,
                                ui.img(
                                    src=debug_plot_src,
                                    style="width: 100%; height: auto;",
                                ),
                            )
                        ),
                        ui.hr(),
                    )
                    ui_elements.append(file_ui)
                else:
                    _log_message(
                        "WARN",
                        filename,
                        None,
                        "Figure conversion to src failed for UI debug plot.",
                    )
                    ui_elements.append(
                        ui.div(
                            ui.h5(f"Debug Details: {filename}"),
                            ui.p(
                                f"Could not generate debug plot image for {filename}."
                            ),
                            ui.hr(),
                        )
                    )
        if not plots_found and results:  # If analysis ran but no plots generated
            return ui.tags.div(
                # ui.h4("Debug Plots (Middle Sweep)"),
                ui.help_text(
                    "Debug plots are enabled, but none were generated. This might happen if analysis failed early for all files, or if the middle sweep processing encountered an error."
                )
            )
        elif not results:  # No files loaded yet
            return ui.tags.div(
                # ui.h4("Debug Plots (Middle Sweep)"),
                ui.help_text("Load ABF files to generate debug plots (if enabled).")
            )
        else:  # Plots were found and added
            return ui.TagList(*ui_elements)

    @output
    @render.data_frame
    def analysis_data_table():
        """Renders the combined analysis DataFrame."""
        df = combined_analysis_df()
        if df.empty:
            # Return an empty DataFrame structure to avoid errors if Shiny expects one
            # You could potentially define expected columns here if known beforehand
            _log_message("DEBUG", "UI", None, "Rendering empty DataFrame.")
            return pd.DataFrame()
        return render.DataGrid(
            df.round(3), selection_mode="none", width="100%", height="600px"
        )

        _log_message("DEBUG", "UI", None, f"Rendering DataFrame with shape {df.shape}")
        return render.DataGrid(
            df.round(4),  # Slightly more precision in table view
            row_selection_mode="none",  # Disable row selection
            width="100%",
            height="600px",
            filters=True,  # Enable column filters
        )

    @render.download(
        filename=lambda: f"ABF_analysis_{len(loaded_abf_data())}files_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    def download_analysis_csv():
        """Provides the combined analysis results as a CSV download."""
        df_to_download = combined_analysis_df()
        req(
            df_to_download is not None and not df_to_download.empty,
            cancel_output=ValueError("No analysis data available to download."),
        )  # Provide user feedback
        _log_message(
            "INFO",
            "Download",
            None,
            f"Generating CSV download for {df_to_download.shape[0]} rows.",
        )
        with io.StringIO() as buf:
            df_to_download.to_csv(buf, index=False, float_format="%.6g")
            yield buf.getvalue()

    @render.download(
        filename=lambda: f"ABF_analysis_{len(loaded_abf_data())}files_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    )
    def download_analysis_excel():
        """Downloads analysis results. Each sheet represents one dependent variable.
        Within each sheet, rows are indexed by sweep and CURRENT_COL_NAME,
        and columns correspond to unique filenames. Yields the binary
        content of the Excel file.
        """
        df_before_pivot = combined_analysis_df()
        req(
            df_before_pivot is not None and not df_before_pivot.empty,
            cancel_output=ValueError("No analysis data available to download."),
        )
        _log_message(
            "INFO",
            "Download",
            None,
            f"Generating Excel download for {df_before_pivot.shape[0]} rows.",
        )

        # --- Prepare the Base DataFrame with MultiIndex ---
        index_cols = ["filename", "sweep", CURRENT_COL_NAME]
        if not all(col in df_before_pivot.columns for col in index_cols):
            missing_cols = [
                col for col in index_cols if col not in df_before_pivot.columns
            ]
            err_msg = f"Error: One or more index columns ({missing_cols}) not found in DataFrame for Excel export."
            _log_message("ERROR", "Download", None, err_msg)
            # Raise error to potentially display in UI or logs, prevents download proceed
            raise ValueError(err_msg)
        dependent_vars = [
            col for col in df_before_pivot.columns if col not in index_cols
        ]
        if not dependent_vars:
            err_msg = "Error: No dependent variable columns found for Excel export."
            _log_message("ERROR", "Download", None, err_msg)
            raise ValueError(err_msg)

        # --- Create Excel file in memory ---
        output_buffer = io.BytesIO()

        try:

            # Use ExcelWriter to manage writing multiple sheets to the buffer
            with pd.ExcelWriter(output_buffer, engine="openpyxl") as writer:
                for var_name in dependent_vars:
                    _log_message(
                        "DEBUG", "Download", None, f"Processing sheet for: {var_name}"
                    )
                    # Select necessary columns for this variable's pivot
                    df_subset = df_before_pivot[index_cols + [var_name]].copy()

                    # Handle potential duplicate index entries before pivoting
                    if df_subset.duplicated(subset=index_cols).any():
                        num_duplicates = df_subset.duplicated(subset=index_cols).sum()
                        _log_message(
                            "WARN",
                            "Download",
                            None,
                            f"Found {num_duplicates} duplicate index entries for var '{var_name}'. Keeping first occurrence for Excel sheet.",
                        )
                        df_subset = df_subset.drop_duplicates(
                            subset=index_cols, keep="first"
                        )

                    try:

                        df_pivot = df_subset.pivot_table(
                            index=["sweep", CURRENT_COL_NAME],
                            columns="filename",
                            values=var_name,
                        )

                    except Exception as e_pivot:
                        _log_message(
                            "ERROR",
                            "Download",
                            None,
                            f"Error pivoting data for variable '{var_name}': {e_pivot}",
                        )
                        # Optionally write an error message to the sheet instead of failing?
                        # For now, re-raise to indicate failure for this sheet/download
                        raise RuntimeError(
                            f"Failed to pivot data for {var_name}"
                        ) from e_pivot

                    # Clean sheet name (max 31 chars, no invalid chars)
                    clean_sheet_name = (
                        str(var_name).replace("_", " ").title()
                    )  # Nicer starting point
                    clean_sheet_name = "".join(
                        c for c in clean_sheet_name if c.isalnum() or c in (" ", "-")
                    ).rstrip()
                    clean_sheet_name = clean_sheet_name[:31]

                    df_pivot.to_excel(
                        writer,
                        sheet_name=clean_sheet_name,
                        index=True,
                        float_format="%.6g",
                        na_rep="NaN",
                    )

            # --- Finalize and Yield ---
            output_buffer.seek(0)  # Rewind buffer to the beginning
            excel_data = output_buffer.getvalue()  # Get binary content
            _log_message("INFO", "Download", None, "Excel file generated successfully.")
            yield excel_data

        except Exception as e_excel:
            _log_message(
                "ERROR",
                "Download",
                None,
                f"Critical error during Excel generation: {e_excel}",
            )
            traceback.print_exc()
            # Yield an error message string? Shiny download might handle exceptions directly.
            # For robustness, could yield a simple error file content.
            yield f"Error generating Excel file: {e_excel}".encode("utf-8")
        finally:
            output_buffer.close()

    @render.download(
        filename=lambda: f"ABF_Summary_Plots_{len(loaded_abf_data())}files_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    )
    def download_plots_pdf():
        """
        Generates and downloads a multi-page PDF with summary plots for two files
        per A4 landscape page, using the refactored plotting logic.
        """
        results = analysis_results_list()
        req(
            results,
            cancel_output=ValueError(
                "No analysis results available to generate PDF plots."
            ),
        )

        num_files = len(results)
        _log_message(
            "INFO",
            "PDF Export",
            None,
            f"Starting PDF generation for {num_files} files (2 per page).",
        )

        A4_LANDSCAPE_WIDTH_IN = 11.69
        A4_LANDSCAPE_HEIGHT_IN = 8.27

        try:
            with BytesIO() as pdf_buffer:
                # Create a PdfPages object to write multiple pages to the buffer
                with PdfPages(pdf_buffer) as pdf:
                    # Iterate through the results, taking two files at a time for each page
                    for i in range(0, num_files, 2):
                        fig_pdf_page = None  # Initialize figure for this page
                        try:
                            # Create a figure with a 2x3 grid for two files
                            fig_pdf_page, axes = plt.subplots(
                                2,  # Rows (one per file)
                                3,  # Columns (raw, sc, phase)
                                figsize=(A4_LANDSCAPE_WIDTH_IN, A4_LANDSCAPE_HEIGHT_IN),
                                squeeze=False,  # Always return 2D array for axes
                            )
                            fig_pdf_page.set_layout_engine(
                                "tight", pad=1.5
                            )  # Good padding

                            # --- Process First File (Top Row) ---
                            result_data_1 = results[i]
                            filename_1 = result_data_1["original_filename"]
                            _log_message(
                                "DEBUG",
                                "PDF Export",
                                None,
                                f"Processing File {i+1} for PDF (Top Row): {filename_1}",
                            )
                            # Generate plots onto the top row axes (axes[0, 0], axes[0, 1], axes[0, 2])
                            _generate_summary_plots_for_file(
                                result_data_1,
                                axes=axes[0, :],  # Pass the first row of axes
                                current_col=CURRENT_COL_NAME,
                            )

                            # --- Process Second File (Bottom Row) if it exists ---
                            if i + 1 < num_files:
                                result_data_2 = results[i + 1]
                                filename_2 = result_data_2["original_filename"]
                                _log_message(
                                    "DEBUG",
                                    "PDF Export",
                                    None,
                                    f"Processing File {i+2} for PDF (Bottom Row): {filename_2}",
                                )
                                # Generate plots onto the bottom row axes (axes[1, 0], axes[1, 1], axes[1, 2])
                                _generate_summary_plots_for_file(
                                    result_data_2,
                                    axes=axes[1, :],  # Pass the second row of axes
                                    current_col=CURRENT_COL_NAME,
                                )
                            else:
                                # If odd number of files, turn off axes for the empty second row
                                _log_message(
                                    "DEBUG",
                                    "PDF Export",
                                    None,
                                    f"Odd number of files. Page {pdf.get_pagecount()+1} has only one file.",
                                )
                                for ax_empty in axes[1, :]:
                                    ax_empty.axis("off")

                            # Save the completed figure (page) to the PDF
                            pdf.savefig(fig_pdf_page)

                        except Exception as e_page:
                            # Log error for this specific page, but continue if possible
                            _log_message(
                                "ERROR",
                                "PDF Export",
                                None,
                                f"Failed to create PDF page starting with file {i+1} ({results[i]['original_filename']}): {e_page}",
                            )
                            traceback.print_exc()
                            # Optionally add an error page to the PDF?
                            # fig_err, ax_err = plt.subplots(figsize=(6,4))
                            # _plot_error_message(ax_err, f"Error generating PDF page for:\n{results[i]['original_filename']}\n{e_page}", "PDF Page Error")
                            # pdf.savefig(fig_err)
                            # plt.close(fig_err)

                        finally:
                            # IMPORTANT: Close the figure associated with the page to free memory
                            if fig_pdf_page is not None:
                                plt.close(fig_pdf_page)

                    # --- Set PDF Metadata ---
                    # Must be done *before* closing PdfPages context
                    d = pdf.infodict()
                    d["Title"] = f"ABF Analysis Summary Plots ({num_files} Files)"
                    d["Author"] = "Spike Doctor Analysis App"
                    # Set creation/modification date (UTC recommended for consistency)
                    now_utc = pd.Timestamp.now(tz="UTC")
                    d["CreationDate"] = now_utc
                    d["ModDate"] = now_utc
                    d["Keywords"] = "Electrophysiology ABF Analysis Summary"
                    d["Subject"] = (
                        f"Summary plots for {num_files} ABF files analyzed on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}"
                    )

                    pdf_page_count = pdf.get_pagecount()

                # After PdfPages context closes, get the buffer's content
                pdf_content = pdf_buffer.getvalue()

        except Exception as outer_e:
            _log_message(
                "ERROR",
                "PDF Export",
                None,
                f"Critical error during PDF setup/generation: {outer_e}",
            )
            traceback.print_exc()
            yield f"Error: Failed to generate PDF. Check logs. ({outer_e})".encode(
                "utf-8"
            )
            return  # Stop generation

        _log_message(
            "INFO",
            "PDF Export",
            None,
            f"PDF generation complete ({pdf_page_count} pages).",
        )

        if pdf_content:
            yield pdf_content
        else:
            _log_message(
                "WARN",
                "PDF Export",
                None,
                "PDF content was empty after generation attempt.",
            )
            yield "Error: Generated PDF was empty.".encode("utf-8")


# ==============================================================================
# App Instantiation
# ==============================================================================
app = App(app_ui, server)
