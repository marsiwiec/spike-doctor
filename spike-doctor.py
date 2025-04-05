import io
import base64
import traceback
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any
from io import BytesIO
import shiny
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import pyabf
import efel
from shiny import App, render, ui, reactive, req
from shiny.types import FileInfo

# ==============================================================================
# Configuration Constants
# ==============================================================================

# --- Analysis Settings ---
# IMPORTANT: Set this index based on your experimental protocol's stimulus epoch.
# This index refers to the epoch within abf.sweepEpochs that defines the main
# stimulus start and end times used for eFEL and other calculations.
# Check your ABF file's header in Clampfit or pyABF to determine the correct index.
# Common indices: 0 (first epoch), 1, 2, etc.
SWEEP_EPOCHS_STIMULUS_INDEX: int = 2

# --- eFEL Specific Settings ---
EFEL_DERIVATIVE_THRESHOLD: float = 10.0
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
    "voltage_base",
    "decay_time_constant_after_stim",
]

# --- Window Definitions (in milliseconds from start of sweep) ---
# Window for calculating the AVERAGE stimulus current injected (used by eFEL trace dict)
EFEL_AVG_CURRENT_WIN_START_MS: float = 400.0
EFEL_AVG_CURRENT_WIN_END_MS: float = 500.0

# Window for calculating Vm_mean and current_mean_pA for MANUAL Rin/Cm calculations
MANUAL_CALC_WIN_START_MS: float = 400.0
MANUAL_CALC_WIN_END_MS: float = 500.0

# --- Plotting & Debugging ---
CREATE_DEBUG_PLOTS: bool = True  # Generate a detailed plot for one sweep per file
MAX_RAW_PLOT_SWEEPS: int = 100  # Max sweeps to overlay on the raw trace plot

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


def _get_stimulus_timing_from_epochs(
    abf: pyabf.ABF, stimulus_epoch_index: int, abf_id_str: str
) -> Tuple[Optional[int], Optional[int], Optional[float], Optional[float], str]:
    """
    Extracts stimulus start/end indices and times using abf.sweepEpochs.
    Returns (start_idx, end_idx, start_ms, end_ms, source_description).
    """
    try:
        if not hasattr(abf, "sweepEpochs"):
            raise AttributeError("ABF object missing 'sweepEpochs' attribute.")

        epoch_table = abf.sweepEpochs
        if not hasattr(epoch_table, "p1s") or not hasattr(epoch_table, "p2s"):
            raise AttributeError("SweepEpochs missing 'p1s' or 'p2s' lists.")
        if epoch_table.p1s is None or epoch_table.p2s is None:
            raise ValueError("SweepEpochs 'p1s' or 'p2s' is None.")

        if not (
            0 <= stimulus_epoch_index < len(epoch_table.p1s)
            and 0 <= stimulus_epoch_index < len(epoch_table.p2s)
        ):
            raise IndexError(
                f"Stimulus index {stimulus_epoch_index} is out of bounds "
                f"for {len(epoch_table.p1s)} epochs."
            )

        start_idx = epoch_table.p1s[stimulus_epoch_index]
        end_idx = epoch_table.p2s[stimulus_epoch_index]

        # Validate indices
        if not isinstance(start_idx, (int, np.integer)) or not isinstance(
            end_idx, (int, np.integer)
        ):
            raise TypeError("Epoch indices are not integers.")

        # Clamp end_idx to actual sweep length
        end_idx = min(end_idx, abf.sweepPointCount)

        if not (0 <= start_idx < end_idx <= abf.sweepPointCount):
            raise ValueError(
                f"Invalid epoch indices derived: start={start_idx}, end={end_idx} "
                f"(Sweep points={abf.sweepPointCount})."
            )

        stim_start_ms = (start_idx / abf.dataRate) * 1000.0
        stim_end_ms = (end_idx / abf.dataRate) * 1000.0
        stim_source = f"SweepEpochs(Idx {stimulus_epoch_index})"

        _log_message(
            "DEBUG",
            abf_id_str,
            None,
            f"Stim timing from {stim_source}: {start_idx}-{end_idx} pts "
            f"({stim_start_ms:.2f}-{stim_end_ms:.2f} ms)",
        )
        return start_idx, end_idx, stim_start_ms, stim_end_ms, stim_source

    except Exception as e:
        _log_message(
            "ERROR",
            abf_id_str,
            None,
            f"Failed to get stimulus timing via SweepEpochs: {e}",
        )
        return None, None, None, None, "Error"


def _calculate_time_window_indices(
    abf: pyabf.ABF, start_ms: float, end_ms: float, window_name: str, abf_id_str: str
) -> Tuple[Optional[int], Optional[int]]:
    """Converts time window in ms to sample indices, clamping to sweep bounds."""
    try:
        idx_start = int(start_ms * abf.dataPointsPerMs)
        idx_end = int(end_ms * abf.dataPointsPerMs)

        # Clamp indices to valid range within a sweep
        idx_start = max(0, idx_start)
        idx_end = min(abf.sweepPointCount, idx_end)

        if idx_start >= idx_end:
            _log_message(
                "WARN",
                abf_id_str,
                None,
                f"{window_name} window invalid or zero-width "
                f"({start_ms:.1f}-{end_ms:.1f} ms -> {idx_start}-{idx_end} pts).",
            )
            return None, None

        return idx_start, idx_end
    except Exception as e:
        _log_message(
            "ERROR",
            abf_id_str,
            None,
            f"Failed to calculate {window_name} window indices: {e}",
        )
        return None, None


# ==============================================================================
# Plotting Functions
# ==============================================================================


def plot_raw_traces(
    abf: Optional[pyabf.ABF],
    filename: str,
    ax: Optional[plt.Axes] = None,
    load_error: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Plots raw voltage traces onto a given axes or creates a new figure."""
    if ax is None:
        fig, ax_new = plt.subplots(figsize=(6, 4))
        ax = ax_new
        return_fig = fig
    else:
        return_fig = None
        fig = ax.get_figure()

    plot_title = f"Raw Traces: {filename}"

    if load_error:
        ax.text(
            0.5,
            0.5,
            f"Load Error:\n{load_error}",
            color="red",
            ha="center",
            va="center",
            fontsize=9,
        )
        ax.set_title(plot_title, fontsize=9)
        ax.tick_params(
            axis="both", labelbottom=False, labelleft=False, bottom=False, left=False
        )
        return return_fig

    if abf is None:
        ax.text(
            0.5,
            0.5,
            "Plot Error:\nABF Data Not Available",
            color="red",
            ha="center",
            va="center",
            fontsize=9,
        )
        ax.set_title(plot_title, fontsize=9)
        ax.tick_params(
            axis="both", labelbottom=False, labelleft=False, bottom=False, left=False
        )
        return return_fig

    try:
        num_sweeps = getattr(abf, "sweepCount", 0)
        if num_sweeps == 0:
            ax.text(0.5, 0.5, "No sweeps in file", ha="center", va="center", fontsize=9)
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
            f"{getattr(abf, 'sweepLabelX', 'Time')} ({getattr(abf, 'sweepUnitsX', 's')})",
            fontsize=8,
        )
        ax.set_ylabel(
            f"{getattr(abf, 'sweepLabelY', 'Signal')} ({getattr(abf, 'sweepUnitsY', 'mV')})",
            fontsize=8,
        )
        ax.tick_params(axis="both", which="major", labelsize=8)
        ax.grid(True, linestyle=":", alpha=0.6)

    except Exception as e:
        _log_message("ERROR", filename, None, f"Raw trace plotting failed: {e}")
        ax.cla()
        ax.text(
            0.5,
            0.5,
            f"Plotting Error:\n{e}",
            color="red",
            ha="center",
            va="center",
            fontsize=9,
        )
        ax.set_title(f"Error: {filename}", fontsize=9)
        ax.tick_params(
            axis="both", labelbottom=False, labelleft=False, bottom=False, left=False
        )

    return return_fig


def plot_feature_vs_current(
    analysis_df: Optional[pd.DataFrame],
    feature_name: str,
    current_col: str,
    filename: str,
    abf: Optional[pyabf.ABF],  # Used for units
    ax: Optional[plt.Axes] = None,
) -> Optional[plt.Figure]:
    """Plots a specified feature against current onto given axes or new figure."""
    if ax is None:
        fig, ax_new = plt.subplots(figsize=(6, 4))
        ax = ax_new
        return_fig = fig
    else:
        return_fig = None
        fig = ax.get_figure()

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
        plot_data = analysis_df[[current_col, feature_name]].dropna()
        if plot_data.empty:
            error_message = "No valid data points\nfor plotting."

    if error_message:
        ax.text(
            0.5, 0.5, error_message, ha="center", va="center", fontsize=9, color="gray"
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
        return return_fig

    y_label = feature_title
    current_units = "pA"
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
    if current_col == "current_step_pA":
        current_units = "pA"
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
        ax.text(
            0.5,
            0.5,
            f"Plotting Error:\n{e}",
            color="red",
            ha="center",
            va="center",
            fontsize=9,
        )
        ax.set_title(f"Error: {filename}", fontsize=9)
        ax.tick_params(
            axis="both", labelbottom=False, labelleft=False, bottom=False, left=False
        )

    return return_fig


def plot_phase_plane(
    voltage: Optional[np.ndarray],
    dvdt: Optional[np.ndarray],
    filename: str,
    sweep_num: Optional[int] = None,
    current_pA: Optional[float] = None,
    title_suffix: str = "",
    ax: Optional[plt.Axes] = None,
) -> Optional[plt.Figure]:
    """Generates a phase-plane plot onto given axes or new figure."""
    if ax is None:
        fig, ax_new = plt.subplots(figsize=(6, 4))
        ax = ax_new
        return_fig = fig
    else:
        return_fig = None
        fig = ax.get_figure()

    plot_title = f"Phase Plane"
    if sweep_num is not None:
        plot_title += f" (Sweep {sweep_num}"
        if current_pA is not None:
            plot_title += f", ~{current_pA:.1f} pA"
        plot_title += ")"

    if title_suffix:
        plot_title += f" - {title_suffix}"

    ax.set_title(plot_title, fontsize=9)

    # Check data validity or if suffix indicates an issue
    show_error_text = False
    err_msg = title_suffix  # Default error message is the suffix
    if (
        voltage is None
        or dvdt is None
        or len(voltage) != len(dvdt)
        or len(voltage) == 0
    ):
        if not title_suffix or "N/A" in title_suffix:
            err_msg = "Invalid or mismatched\nVoltage/dVdt data"
        show_error_text = True
    elif title_suffix and any(
        s in title_suffix for s in ["Not Found", "Error", "Invalid", "Missing"]
    ):
        show_error_text = True

    if show_error_text:
        ax.text(
            0.5,
            0.5,
            err_msg.replace(": ", ":\n"),
            ha="center",
            va="center",
            color="gray",
            fontsize=9,
        )
        _log_message(
            "WARN",
            filename,
            sweep_num,
            f"Phase plane plot skipped/failed: {err_msg.replace(chr(10),' ')}",
        )
        ax.tick_params(
            axis="both", labelbottom=False, labelleft=False, bottom=False, left=False
        )
        return return_fig

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
        ax.text(
            0.5,
            0.5,
            f"Plotting Error:\n{e}",
            color="red",
            ha="center",
            va="center",
            fontsize=9,
        )
        ax.set_title(f"Phase Plot Error: {filename}", fontsize=9)
        ax.tick_params(
            axis="both", labelbottom=False, labelleft=False, bottom=False, left=False
        )

    return return_fig


# ==============================================================================
# Core Analysis Function
# ==============================================================================


def run_analysis_on_abf(
    abf: Optional[pyabf.ABF],
    original_filename: str,
    user_selected_features: List[str],
) -> Dict[str, Union[Optional[pd.DataFrame], Optional[plt.Figure]]]:
    """
    Analyzes a single ABF file using eFEL and manual calculations.

    Args:
        abf: The loaded pyabf.ABF object.
        original_filename: The original name of the file (for reporting).
        user_selected_features: List of eFEL features requested by the user.

    Returns:
        A dictionary containing:
            'analysis_df': A pandas DataFrame with results per sweep, or None on failure.
            'debug_plot_fig': A Matplotlib Figure object for debugging, or None.
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

    # 2. Determine Stimulus Timing
    stim_idx_start, stim_idx_end, stim_start_ms, stim_end_ms, stim_source = (
        _get_stimulus_timing_from_epochs(abf, SWEEP_EPOCHS_STIMULUS_INDEX, abf_id_str)
    )
    if stim_start_ms is None or stim_end_ms is None:
        _log_message(
            "ERROR",
            abf_id_str,
            None,
            "Could not determine stimulus timing. Aborting analysis.",
        )
        return analysis_output  # Error logged in timing function

    # 3. Determine Calculation Window Indices
    idx_manual_start, idx_manual_end = _calculate_time_window_indices(
        abf, MANUAL_CALC_WIN_START_MS, MANUAL_CALC_WIN_END_MS, "Manual Calc", abf_id_str
    )
    idx_efel_curr_start, idx_efel_curr_end = _calculate_time_window_indices(
        abf,
        EFEL_AVG_CURRENT_WIN_START_MS,
        EFEL_AVG_CURRENT_WIN_END_MS,
        "eFEL Current Avg",
        abf_id_str,
    )
    if idx_efel_curr_start is None or idx_efel_curr_end is None:
        _log_message(
            "ERROR",
            abf_id_str,
            None,
            "Failed to define eFEL current averaging window. Aborting.",
        )
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

    # 5. Prepare eFEL
    all_efel_features_needed = list(
        set(user_selected_features) | set(REQUIRED_INTERNAL_EFEL_FEATURES)
    )
    try:
        efel.reset()
        efel.setDoubleSetting("strict_stiminterval", True)
        efel.setDoubleSetting("DerivativeThreshold", EFEL_DERIVATIVE_THRESHOLD)
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

    # 6. Sweep Loop and Analysis
    sweep_results_list = []
    debug_plot_generated = False  # Flag to ensure only one debug plot per file
    middle_sweep_index_for_debug = abf.sweepList[abf.sweepCount // 2]

    try:
        for sweep_num in abf.sweepList:
            abf.setSweep(sweep_num)

            # --- Calculate Average Stimulus Current (for eFEL trace dict, in nA) ---
            stimulus_current_for_efel_nA = 0.0
            try:
                if not hasattr(abf, "sweepC") or not isinstance(abf.sweepC, np.ndarray):
                    raise TypeError(
                        "Sweep command data (sweepC) is missing or not a numpy array."
                    )
                # Average over the defined eFEL current window
                avg_sweepC_raw = np.nanmean(
                    abf.sweepC[idx_efel_curr_start:idx_efel_curr_end]
                )
                if np.isnan(avg_sweepC_raw):
                    raise ValueError("Current data in window contains only NaNs.")
                # Convert raw average current to nA for eFEL standard
                stimulus_current_for_efel_nA = avg_sweepC_raw * (
                    current_unit_factor / 1000.0
                )
            except Exception as avg_err:
                # Warn but continue, assuming 0 current for eFEL trace
                _log_message(
                    "WARN",
                    abf_id_str,
                    sweep_num,
                    f"Failed to calculate average stimulus current: {avg_err}. Using 0 nA for eFEL trace.",
                )

            # --- Prepare eFEL Trace Input ---
            # Times must be in ms, Voltage in mV, Current in nA
            trace_for_efel = {
                "T": abf.sweepX * 1000.0,
                "V": abf.sweepY,
                "stim_start": [stim_start_ms],
                "stim_end": [stim_end_ms],
                "stimulus_current": [float(stimulus_current_for_efel_nA)],
            }

            # --- Run eFEL ---
            efel_results_raw = None
            efel_error = None
            efel_warnings = []
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always", RuntimeWarning)
                try:
                    efel_results_raw = efel.get_feature_values(
                        [trace_for_efel], all_efel_features_needed
                    )[0]
                except Exception as e:
                    efel_error = e  # Store error to handle after warnings
                # Store relevant warnings
                efel_warnings = [
                    str(w.message)
                    for w in caught_warnings
                    if issubclass(w.category, RuntimeWarning)
                ]

            if efel_error:
                _log_message(
                    "ERROR",
                    abf_id_str,
                    sweep_num,
                    f"eFEL feature extraction failed: {efel_error}",
                )
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

            # --- Manual Calculations (Rin, Cm) ---
            current_mean_pA = np.nan  # Current during manual window (pA)
            Vm_mean = np.nan  # Voltage during manual window (mV)
            R_in_manual_Mohm = np.nan
            tau_m_efel_ms = np.nan
            Cm_manual_pF = np.nan

            if idx_manual_start is not None and idx_manual_end is not None:
                try:
                    # Ignore warnings during nanmean (e.g., if window has NaNs)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        current_mean_pA = (
                            np.nanmean(abf.sweepC[idx_manual_start:idx_manual_end])
                            * current_unit_factor
                        )
                        Vm_mean = np.nanmean(
                            abf.sweepY[idx_manual_start:idx_manual_end]
                        )
                except Exception as calc_err:
                    _log_message(
                        "WARN",
                        abf_id_str,
                        sweep_num,
                        f"Manual Vm/I calculation failed: {calc_err}",
                    )

            spike_count = efel_results_parsed.get("spike_count", 0)
            spike_count = int(spike_count) if pd.notna(spike_count) else 0
            V_base_efel = efel_results_parsed.get("voltage_base", np.nan)

            if (
                spike_count == 0
                and pd.notna(current_mean_pA)
                and not np.isclose(current_mean_pA, 0)
                and pd.notna(Vm_mean)
                and pd.notna(V_base_efel)
            ):
                try:
                    delta_V = Vm_mean - V_base_efel
                    R_in_manual_Mohm = (delta_V / current_mean_pA) * 1000.0
                    if not np.isfinite(R_in_manual_Mohm):
                        R_in_manual_Mohm = (
                            np.inf if np.isclose(current_mean_pA, 0) else np.nan
                        )
                except (ZeroDivisionError, FloatingPointError):
                    R_in_manual_Mohm = np.inf
                except Exception as rin_err:
                    _log_message(
                        "WARN",
                        abf_id_str,
                        sweep_num,
                        f"Manual Rin calculation failed: {rin_err}",
                    )
                    R_in_manual_Mohm = np.nan

            tau_m_efel_ms = efel_results_parsed.get(
                "decay_time_constant_after_stim", np.nan
            )

            if (
                pd.notna(tau_m_efel_ms)
                and pd.notna(R_in_manual_Mohm)
                and np.isfinite(R_in_manual_Mohm)
                and R_in_manual_Mohm > 0
            ):
                try:
                    Cm_manual_pF = (tau_m_efel_ms / R_in_manual_Mohm) * 1000.0
                    if not np.isfinite(Cm_manual_pF):
                        Cm_manual_pF = np.nan
                except (ZeroDivisionError, FloatingPointError):
                    Cm_manual_pF = np.inf
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
                "current_step_pA": current_mean_pA,
                "input_resistance_Mohm": R_in_manual_Mohm,
                "time_constant_ms": tau_m_efel_ms,
                "capacitance_pF": Cm_manual_pF,
                **{
                    feat: efel_results_parsed.get(feat, np.nan)
                    for feat in user_selected_features
                },
            }

            if spike_count <= 1:
                for feature in sweep_data:
                    if "frequency" in feature.lower() or "isi" in feature.lower():
                        sweep_data[feature] = np.nan
            if spike_count == 0:
                for feature in sweep_data:
                    if "time_to_" in feature.lower() or "latency" in feature.lower():
                        sweep_data[feature] = np.nan

            sweep_results_list.append(sweep_data)

            # --- Generate Debug Plot (only once per file, for a middle sweep) ---
            if (
                CREATE_DEBUG_PLOTS
                and sweep_num == middle_sweep_index_for_debug
                and not debug_plot_generated
            ):
                try:
                    fig_debug, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
                    axs[0].plot(
                        abf.sweepX,
                        abf.sweepY,
                        color="black",
                        lw=0.7,
                        label=f"Sweep {sweep_num}",
                    )
                    axs[0].axvspan(
                        stim_start_ms / 1000.0,
                        stim_end_ms / 1000.0,
                        color="red",
                        alpha=0.15,
                        label=f"eFEL Stim ({stim_source})",
                    )
                    if idx_manual_start is not None and idx_manual_end is not None:
                        axs[0].axvspan(
                            abf.sweepX[idx_manual_start],
                            abf.sweepX[idx_manual_end - 1],
                            color="cyan",
                            alpha=0.2,
                            label="Manual Calc Win",
                        )
                    if pd.notna(V_base_efel):
                        axs[0].axhline(
                            V_base_efel,
                            color="purple",
                            linestyle=":",
                            lw=1.5,
                            label=f"V_base (eFEL): {V_base_efel:.1f} mV",
                        )
                    if pd.notna(Vm_mean):
                        axs[0].hlines(
                            Vm_mean,
                            abf.sweepX[idx_manual_start],
                            abf.sweepX[idx_manual_end - 1],
                            color="cyan",
                            linestyle="--",
                            lw=1.5,
                            label=f"Vm Mean (Manual): {Vm_mean:.1f} mV",
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
                            abf.sweepC * current_unit_factor,
                            color="blue",
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

                    axs[1].axvspan(
                        stim_start_ms / 1000.0,
                        stim_end_ms / 1000.0,
                        color="red",
                        alpha=0.15,
                    )
                    if idx_manual_start is not None and idx_manual_end is not None:
                        axs[1].axvspan(
                            abf.sweepX[idx_manual_start],
                            abf.sweepX[idx_manual_end - 1],
                            color="cyan",
                            alpha=0.2,
                        )
                    if (
                        idx_efel_curr_start is not None
                        and idx_efel_curr_end is not None
                    ):
                        axs[1].axvspan(
                            abf.sweepX[idx_efel_curr_start],
                            abf.sweepX[idx_efel_curr_end - 1],
                            color="lime",
                            alpha=0.15,
                            label="eFEL I Avg Win",
                        )

                    if pd.notna(current_mean_pA):
                        axs[1].hlines(
                            current_mean_pA,
                            abf.sweepX[idx_manual_start],
                            abf.sweepX[idx_manual_end - 1],
                            color="cyan",
                            linestyle="--",
                            lw=1.5,
                            label=f"I Mean (Manual): {current_mean_pA:.1f} {output_plot_unit}",
                        )
                    if pd.notna(stimulus_current_for_efel_nA):
                        # Convert back to pA for plotting consistency if units assumed pA originally
                        current_for_plot = stimulus_current_for_efel_nA * 1000.0
                        axs[1].hlines(
                            current_for_plot,
                            abf.sweepX[idx_efel_curr_start],
                            abf.sweepX[idx_efel_curr_end - 1],
                            color="lime",
                            linestyle="--",
                            lw=1.5,
                            label=f"I Avg (eFEL): {current_for_plot:.1f} {output_plot_unit}",
                        )

                    axs[1].set_ylabel(f"Current ({output_plot_unit})")
                    axs[1].set_xlabel(f"Time ({getattr(abf, 'sweepUnitsX', 's')})")
                    axs[1].legend(fontsize=7, loc="best")
                    axs[1].grid(True, linestyle=":", alpha=0.5)

                    fig_debug.tight_layout()
                    analysis_output["debug_plot_fig"] = fig_debug
                    debug_plot_generated = True
                except Exception as plot_err:
                    # Log error but don't stop analysis
                    _log_message(
                        "ERROR",
                        abf_id_str,
                        sweep_num,
                        f"Debug plot generation failed: {plot_err}",
                    )
                    analysis_output["debug_plot_fig"] = None
                    plt.close(fig_debug)

        # --- End of Sweep Loop ---

    except Exception as loop_err:
        _log_message(
            "ERROR",
            abf_id_str,
            None,
            f"Critical error during sweep processing: {loop_err}",
        )
        traceback.print_exc()
        if sweep_results_list:
            _log_message(
                "WARN",
                abf_id_str,
                None,
                "Returning partial results due to error in loop.",
            )
        else:
            analysis_output["analysis_df"] = None
            return analysis_output

    # 7. Final DataFrame Assembly
    if not sweep_results_list:
        _log_message(
            "WARN",
            abf_id_str,
            None,
            "Analysis finished, but no sweep results were generated.",
        )
        analysis_output["analysis_df"] = pd.DataFrame()
        return analysis_output

    try:
        analysis_df = pd.DataFrame(sweep_results_list)

        standard_cols = [
            "filename",
            "sweep",
            "current_step_pA",
            "input_resistance_Mohm",
            "time_constant_ms",
            "capacitance_pF",
        ]

        efel_cols_present = sorted(
            [f for f in user_selected_features if f in analysis_df.columns]
        )

        final_ordered_cols = standard_cols + efel_cols_present
        analysis_df = analysis_df.reindex(columns=final_ordered_cols)

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
            ui.tags.ul(
                ui.tags.li(f"Method: From ABF Header (SweepEpochs)"),
                ui.tags.li(f"Epoch Index Used: {SWEEP_EPOCHS_STIMULUS_INDEX}"),
            ),
            ui.tags.b("Current Input (for eFEL):"),
            ui.tags.ul(
                ui.tags.li(
                    f"Average current during {EFEL_AVG_CURRENT_WIN_START_MS:.0f}-{EFEL_AVG_CURRENT_WIN_END_MS:.0f} ms"
                ),
                ui.tags.li(f"(Passed as 'stimulus_current' in nA to eFEL)"),
            ),
            ui.tags.b("Manual Calculations (Rin, Cm):"),
            ui.tags.ul(
                ui.tags.li(
                    f"Analysis Window: {MANUAL_CALC_WIN_START_MS:.0f}-{MANUAL_CALC_WIN_END_MS:.0f} ms"
                ),
                ui.tags.li("Performed only if eFEL spike_count = 0"),
                ui.tags.li("Uses eFEL V_base and Tau_decay"),
                ui.tags.li("Outputs: input_resistance_Mohm, capacitance_pF"),
            ),
            ui.tags.b("eFEL Spike Detection:"),
            ui.tags.ul(
                ui.tags.li(f"dV/dt Threshold: {EFEL_DERIVATIVE_THRESHOLD} mV/ms")
            ),
            ui.tags.b("Options:"),
            ui.tags.ul(
                ui.tags.li(
                    f"Generate Debug Plots: {'Yes' if CREATE_DEBUG_PLOTS else 'No'}"
                )
            ),
            ui.hr(),
            ui.input_checkbox_group(
                "selected_efel_features",
                "Select eFEL Features to Calculate:",
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
            return

        data_list = []
        num_files = len(file_infos)
        print(f"Loading {num_files} ABF file(s)...")
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
                        abf_obj = pyabf.ABF(str(filepath))
                except Exception as e:
                    error_msg = f"Failed to load: {e}"
                    _log_message("ERROR", filename, None, error_msg)

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
        print(f"Finished loading {len(data_list)} files.")

    analysis_results_list = reactive.Calc(
        lambda: [
            {
                **file_data,
                **run_analysis_on_abf(
                    file_data["abf_object"],
                    file_data["original_filename"],
                    input.selected_efel_features(),
                ),
            }
            for file_data in loaded_abf_data()
        ]
    )

    @reactive.Calc
    def combined_analysis_df() -> pd.DataFrame:
        """
        Combines analysis results from individual files into a single DataFrame.
        Returns an empty DataFrame if no valid results are found or on error.
        """
        _log_message("DEBUG", "App", None, "Combining analysis results...")
        all_results = analysis_results_list()
        valid_dfs = [
            r.get("analysis_df")
            for r in all_results
            if isinstance(r.get("analysis_df"), pd.DataFrame)
            and not r["analysis_df"].empty
        ]
        if not valid_dfs:
            _log_message("WARN", "App", None, "No valid DataFrames to combine.")
            return pd.DataFrame()
        try:
            combined = pd.concat(valid_dfs, ignore_index=True, sort=False)
            _log_message(
                "DEBUG", "App", None, f"Combined DataFrame shape: {combined.shape}"
            )
            return combined
        except Exception as e:
            _log_message("ERROR", "App", None, f"Failed to concatenate DataFrames: {e}")
            return pd.DataFrame()

    @output
    @render.text
    def analysis_summary_text():
        """Displays a summary of loaded files and analysis status."""
        results = analysis_results_list()
        num_total = len(results)
        if num_total == 0:
            return "Upload one or more ABF files to begin analysis."

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
            f"Total Files Attempted: {num_total}\n"
            f"Successfully Loaded: {num_load_ok}\n"
            f"Load Errors: {num_load_err}\n"
            f"Successfully Analyzed: {num_analyzed_ok}\n"
            f"Analysis Skipped/Failed: {num_analysis_failed}\n"
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
            summary += f"Output Columns ({first_analyzed_data['original_filename']}):\n{', '.join(cols)}\n"

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

        for i, result_data in enumerate(results):
            filename = result_data["original_filename"]
            abf_obj = result_data.get("abf_object")
            load_err = result_data.get("load_error")
            analysis_df = result_data.get("analysis_df")  # Might be None or empty DF

            raw_plot_src = None
            if load_err:
                fig, ax = plt.subplots(figsize=(7, 3))
                ax.text(
                    0.5,
                    0.5,
                    f"Load Error:\n{load_err}",
                    color="red",
                    ha="center",
                    va="center",
                )
                ax.set_title(f"Raw Traces: {filename}")
                raw_plot_src = fig_to_src(fig)
            elif abf_obj:
                raw_plot_src = fig_to_src(plot_raw_traces(abf_obj, filename))

            raw_plot_ui = ui.column(
                4,
                (
                    ui.img(src=raw_plot_src, style="width: 100%; height: auto;")
                    if raw_plot_src
                    else ui.p(f"Could not generate raw plot for {filename}.")
                ),
            )

            sc_plot_src = None
            plot_feature = "spike_count"
            current_column = "current_step_pA"
            if isinstance(analysis_df, pd.DataFrame) and not analysis_df.empty:
                sc_plot_src = fig_to_src(
                    plot_feature_vs_current(
                        analysis_df, plot_feature, current_column, filename, abf_obj
                    )
                )
            elif abf_obj and not load_err:
                fig, ax = plt.subplots(figsize=(7, 3))
                ax.text(
                    0.5,
                    0.5,
                    "Analysis skipped or failed.\nCannot plot spike count.",
                    ha="center",
                    va="center",
                )
                ax.set_title(f"Spike Count vs Current: {filename}")
                sc_plot_src = fig_to_src(fig)

            sc_plot_ui = ui.column(
                4,
                (
                    ui.img(src=sc_plot_src, style="width: 100%; height: auto;")
                    if sc_plot_src
                    else ui.p(f"Could not generate SC plot for {filename}.")
                ),
            )

            phase_plot_src = None
            target_sweep_num = None
            target_current_pA = None
            phase_v = None
            phase_dvdt = None
            phase_title_suffix = "N/A"

            if (
                isinstance(analysis_df, pd.DataFrame)
                and not analysis_df.empty
                and current_column in analysis_df.columns
                and "spike_count" in analysis_df.columns
                and abf_obj
            ):

                try:
                    # Find Rheobase
                    spiking_sweeps = analysis_df[
                        (analysis_df["spike_count"] >= 1)
                        & (analysis_df[current_column] > 0)
                    ]
                    if not spiking_sweeps.empty:
                        rheobase_row = spiking_sweeps.loc[
                            spiking_sweeps[current_column].idxmin()
                        ]
                        rheobase_current = rheobase_row[current_column]
                        target_current = 2 * rheobase_current
                        _log_message(
                            "DEBUG",
                            filename,
                            None,
                            f"Rheobase found: {rheobase_current:.2f} pA. Target 2x: {target_current:.2f} pA",
                        )

                        # Find sweep closest to 2x rheobase
                        # Ensure we only consider sweeps with valid current values for finding the minimum distance
                        valid_current_df = analysis_df.dropna(subset=[current_column])
                        if not valid_current_df.empty:
                            closest_idx = (
                                (valid_current_df[current_column] - target_current)
                                .abs()
                                .idxmin()
                            )
                            closest_sweep_row = valid_current_df.loc[closest_idx]
                            target_sweep_num = int(closest_sweep_row["sweep"])
                            target_current_pA = closest_sweep_row[current_column]
                            _log_message(
                                "DEBUG",
                                filename,
                                None,
                                f"Closest sweep to 2x rheo: {target_sweep_num} ({target_current_pA:.2f} pA)",
                            )

                            abf_obj.setSweep(target_sweep_num)
                            phase_v = abf_obj.sweepY
                            time_s = abf_obj.sweepX
                            if len(phase_v) > 1 and len(time_s) > 1:
                                phase_dvdt = np.gradient(phase_v, time_s * 1000.0)
                                phase_title_suffix = ""
                            else:
                                phase_title_suffix = "Sweep Data Invalid"
                        else:
                            phase_title_suffix = "No Valid Current Data"

                    else:
                        phase_title_suffix = "Rheobase Not Found\n(No Spikes Detected)"
                        _log_message(
                            "WARN", filename, None, "Rheobase not found for phase plot."
                        )

                except Exception as e:
                    phase_title_suffix = f"Phase Calc Error:\n{e}"
                    _log_message(
                        "ERROR",
                        filename,
                        None,
                        f"Error calculating phase plot data: {e}",
                    )

            elif load_err:
                phase_title_suffix = "Load Error"
            elif not abf_obj:
                phase_title_suffix = "ABF Not Loaded"
            elif not isinstance(analysis_df, pd.DataFrame) or analysis_df.empty:
                phase_title_suffix = "Analysis Failed"
            else:
                phase_title_suffix = "Req. Columns Missing"

            phase_fig = plot_phase_plane(
                phase_v,
                phase_dvdt,
                filename,
                target_sweep_num,
                target_current_pA,
                phase_title_suffix,
            )
            phase_plot_src = fig_to_src(phase_fig)

            phase_plot_ui = ui.column(
                4,
                (
                    ui.img(src=phase_plot_src, style="width: 100%; height: auto;")
                    if phase_plot_src
                    else ui.p(f"Could not generate phase plot for {filename}.")
                ),
            )

            file_ui = ui.div(
                ui.hr() if i > 0 else None,  # Add separator between files
                ui.h5(filename),
                ui.row(raw_plot_ui, sc_plot_ui, phase_plot_ui),
            )
            ui_elements.append(file_ui)

        return (
            ui.TagList(*ui_elements)
            if ui_elements
            else ui.p("Processing files or no plots generated.")
        )

    @output
    @render.ui
    def dynamic_debug_plots_ui():
        """Dynamically generates UI for the debug plots."""
        if not CREATE_DEBUG_PLOTS:
            return ui.tags.p("Debug plots are disabled in the configuration.")

        results = analysis_results_list()
        req(results)

        ui_elements = []
        for result_data in results:
            filename = result_data["original_filename"]
            debug_fig = result_data.get("debug_plot_fig")

            if isinstance(debug_fig, plt.Figure):
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

        return (
            ui.TagList(*ui_elements)
            if ui_elements
            else ui.p(
                "No debug plots were generated (check analysis logs or enable debug plots)."
            )
        )

    @output
    @render.data_frame
    def analysis_data_table():
        """Renders the combined analysis DataFrame."""
        df = combined_analysis_df()
        if df.empty:
            # Return an empty DataFrame structure to avoid errors if Shiny expects one
            # You could potentially define expected columns here if known beforehand
            return pd.DataFrame()
        return render.DataGrid(
            df.round(3),
            row_selection_mode="none",
            width="100%",
            height="600px",
        )

    @session.download(
        filename=lambda: f"ABF_analysis_{len(loaded_abf_data())}files_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    def download_analysis_csv():
        """Provides the combined analysis results as a CSV download."""
        df_to_download = combined_analysis_df()
        req(df_to_download is not None and not df_to_download.empty)

        with io.StringIO() as buf:
            df_to_download.to_csv(
                buf, index=False, float_format="%.6f"
            )  # Use more precision in CSV
            yield buf.getvalue()

    @session.download(
        filename=lambda: f"ABF_analysis_{len(loaded_abf_data())}files_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    )
    def download_analysis_excel():
        """Downloads analysis results. Each sheet represents one dependent variable.
        Within each sheet, rows are indexed by sweep and current_step_pA,
        and columns correspond to unique filenames. Yields the binary
        content of the Excel file.
        """
        df_before_pivot = combined_analysis_df()
        req(df_before_pivot is not None and not df_before_pivot.empty)

        # --- Prepare the Base DataFrame with MultiIndex ---
        index_cols = ["filename", "sweep", "current_step_pA"]
        if not all(col in df_before_pivot.columns for col in index_cols):
            print(
                f"Error: One or more index columns {index_cols} not found in DataFrame."
            )
            raise ValueError(f"Missing index columns: {index_cols}")

        try:
            # Set the multi-level index. This df still has dependent vars as columns.
            # Ensure the combination of index_cols uniquely identifies rows for this step.
            df_multi_indexed = df_before_pivot.set_index(index_cols)
        except KeyError as e:
            print(f"Error setting index, column not found: {e}")
            raise
        except Exception as e:
            print(f"Error setting index (potential duplicates?): {e}")
            raise

        dependent_vars = df_multi_indexed.columns

        # --- Create Excel file in memory ---
        output_buffer = io.BytesIO()

        # Use ExcelWriter to manage writing multiple sheets to the buffer
        with pd.ExcelWriter(output_buffer, engine="openpyxl") as writer:
            for var_name in dependent_vars:
                series_for_var = df_multi_indexed[var_name]

                try:
                    df_sheet = series_for_var.unstack(level="filename")
                except Exception as e:
                    print(f"Error unstacking 'filename' for variable '{var_name}': {e}")
                    raise

                clean_sheet_name = (
                    str(var_name)
                    .replace("[", "")
                    .replace("]", "")
                    .replace("*", "")
                    .replace("/", "")
                    .replace("\\", "")
                    .replace("?", "")
                    .replace(":", "")
                )
                clean_sheet_name = clean_sheet_name[:31]

                df_sheet.to_excel(
                    writer,
                    sheet_name=clean_sheet_name,
                    index=True,
                    float_format="%.6f",
                    na_rep="NaN",
                )

        output_buffer.seek(0)
        excel_data = output_buffer.getvalue()
        output_buffer.close()

        yield excel_data

    @session.download(
        filename=lambda: f"ABF_Summary_Plots_{len(loaded_abf_data())}files_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    )
    def download_plots_pdf():
        """
        Generates and downloads a multi-page PDF with summary plots for two files
        per A4 landscape page.
        """
        results = analysis_results_list()
        req(results)

        num_files = len(results)
        _log_message(
            "INFO",
            "PDF Export",
            None,
            f"Starting PDF generation for {num_files} files (2 per page).",
        )

        pdf_content = None

        A4_LANDSCAPE_WIDTH = 11.69
        A4_LANDSCAPE_HEIGHT = 8.27

        try:
            with BytesIO() as pdf_buffer:
                with PdfPages(pdf_buffer) as pdf:
                    for i in range(0, num_files, 2):
                        fig, axes = (
                            None,
                            None,
                        )

                        try:
                            fig, axes = plt.subplots(
                                2,
                                3,
                                figsize=(A4_LANDSCAPE_WIDTH, A4_LANDSCAPE_HEIGHT),
                                squeeze=False,
                            )
                            result_data_1 = results[i]
                            filename_1 = result_data_1["original_filename"]
                            abf_obj_1 = result_data_1.get("abf_object")
                            load_err_1 = result_data_1.get("load_error")
                            analysis_df_1 = result_data_1.get("analysis_df")
                            _log_message(
                                "DEBUG",
                                "PDF Export",
                                None,
                                f"Processing File {i+1} for PDF (Top Row): {filename_1}",
                            )

                            try:
                                raw_title_1 = f"Raw: {filename_1}"
                                plot_raw_traces(
                                    abf_obj_1,
                                    raw_title_1,
                                    ax=axes[0, 0],
                                    load_error=load_err_1,
                                )
                            except Exception as e_plot:
                                _log_message(
                                    "ERROR",
                                    filename_1,
                                    None,
                                    f"PDF Raw Plot Error: {e_plot}",
                                )
                                axes[0, 0].text(
                                    0.5,
                                    0.5,
                                    f"Raw Plot Error:\n{e_plot}",
                                    ha="center",
                                    va="center",
                                    color="red",
                                    fontsize=9,
                                )
                                axes[0, 0].set_title(
                                    f"Raw Error: {filename_1}", fontsize=9
                                )

                            try:
                                plot_feature_vs_current(
                                    analysis_df_1,
                                    "spike_count",
                                    "current_step_pA",
                                    filename_1,
                                    abf_obj_1,
                                    ax=axes[0, 1],
                                )
                            except Exception as e_plot:
                                _log_message(
                                    "ERROR",
                                    filename_1,
                                    None,
                                    f"PDF SC Plot Error: {e_plot}",
                                )
                                axes[0, 1].text(
                                    0.5,
                                    0.5,
                                    f"SC Plot Error:\n{e_plot}",
                                    ha="center",
                                    va="center",
                                    color="red",
                                    fontsize=9,
                                )
                                axes[0, 1].set_title(
                                    "Spike Count vs Current", fontsize=9
                                )

                            try:
                                phase_v, phase_dvdt = None, None
                                target_sweep_num, target_current_pA = None, None
                                phase_title_suffix = "N/A"
                                current_column = "current_step_pA"
                                if (
                                    isinstance(analysis_df_1, pd.DataFrame)
                                    and not analysis_df_1.empty
                                    and current_column in analysis_df_1.columns
                                    and "spike_count" in analysis_df_1.columns
                                    and abf_obj_1
                                ):
                                    try:
                                        spiking_sweeps = analysis_df_1[
                                            (analysis_df_1["spike_count"] >= 1)
                                            & (analysis_df_1[current_column] > 0)
                                        ]
                                        if not spiking_sweeps.empty:
                                            rheobase_row = spiking_sweeps.loc[
                                                spiking_sweeps[current_column].idxmin()
                                            ]
                                            rheobase_current = rheobase_row[
                                                current_column
                                            ]
                                            target_current = 2 * rheobase_current
                                            valid_current_df = analysis_df_1.dropna(
                                                subset=[current_column]
                                            )
                                            if not valid_current_df.empty:
                                                closest_idx = (
                                                    (
                                                        valid_current_df[current_column]
                                                        - target_current
                                                    )
                                                    .abs()
                                                    .idxmin()
                                                )
                                                closest_sweep_row = (
                                                    valid_current_df.loc[closest_idx]
                                                )
                                                target_sweep_num = int(
                                                    closest_sweep_row["sweep"]
                                                )
                                                target_current_pA = closest_sweep_row[
                                                    current_column
                                                ]
                                                abf_obj_1.setSweep(target_sweep_num)
                                                phase_v = abf_obj_1.sweepY
                                                time_s = abf_obj_1.sweepX
                                                if len(phase_v) > 1 and len(time_s) > 1:
                                                    phase_dvdt = np.gradient(
                                                        phase_v, time_s * 1000.0
                                                    )
                                                    phase_title_suffix = ""
                                                else:
                                                    phase_title_suffix = "Sweep Invalid"
                                            else:
                                                phase_title_suffix = "No Valid I"
                                        else:
                                            phase_title_suffix = "Rheo Not Found"
                                    except Exception as e_inner:
                                        phase_title_suffix = f"Calc Error"
                                        _log_message(
                                            "ERROR",
                                            filename_1,
                                            target_sweep_num,
                                            f"PDF Phase Calc Error: {e_inner}",
                                        )
                                elif load_err_1:
                                    phase_title_suffix = "Load Error"
                                elif not abf_obj_1:
                                    phase_title_suffix = "No ABF"
                                elif (
                                    not isinstance(analysis_df_1, pd.DataFrame)
                                    or analysis_df_1.empty
                                ):
                                    phase_title_suffix = "No Analysis"
                                else:
                                    phase_title_suffix = "Cols Missing"

                                plot_phase_plane(
                                    phase_v,
                                    phase_dvdt,
                                    filename_1,
                                    target_sweep_num,
                                    target_current_pA,
                                    phase_title_suffix,
                                    ax=axes[0, 2],
                                )
                            except Exception as e_plot:
                                _log_message(
                                    "ERROR",
                                    filename_1,
                                    None,
                                    f"PDF Phase Plot Error: {e_plot}",
                                )
                                axes[0, 2].text(
                                    0.5,
                                    0.5,
                                    f"Phase Plot Error:\n{e_plot}",
                                    ha="center",
                                    va="center",
                                    color="red",
                                    fontsize=9,
                                )
                                axes[0, 2].set_title("Phase Plane", fontsize=9)

                            if i + 1 < num_files:
                                result_data_2 = results[i + 1]
                                filename_2 = result_data_2["original_filename"]
                                abf_obj_2 = result_data_2.get("abf_object")
                                load_err_2 = result_data_2.get("load_error")
                                analysis_df_2 = result_data_2.get("analysis_df")
                                _log_message(
                                    "DEBUG",
                                    "PDF Export",
                                    None,
                                    f"Processing File {i+2} for PDF (Bottom Row): {filename_2}",
                                )

                                try:
                                    raw_title_2 = f"Raw: {filename_2}"
                                    plot_raw_traces(
                                        abf_obj_2,
                                        raw_title_2,
                                        ax=axes[1, 0],
                                        load_error=load_err_2,
                                    )
                                except Exception as e_plot:
                                    _log_message(
                                        "ERROR",
                                        filename_2,
                                        None,
                                        f"PDF Raw Plot Error: {e_plot}",
                                    )
                                    axes[1, 0].text(
                                        0.5,
                                        0.5,
                                        f"Raw Plot Error:\n{e_plot}",
                                        ha="center",
                                        va="center",
                                        color="red",
                                        fontsize=9,
                                    )
                                    axes[1, 0].set_title(
                                        f"Raw Error: {filename_2}", fontsize=9
                                    )

                                try:
                                    plot_feature_vs_current(
                                        analysis_df_2,
                                        "spike_count",
                                        "current_step_pA",
                                        filename_2,
                                        abf_obj_2,
                                        ax=axes[1, 1],
                                    )
                                except Exception as e_plot:
                                    _log_message(
                                        "ERROR",
                                        filename_2,
                                        None,
                                        f"PDF SC Plot Error: {e_plot}",
                                    )
                                    axes[1, 1].text(
                                        0.5,
                                        0.5,
                                        f"SC Plot Error:\n{e_plot}",
                                        ha="center",
                                        va="center",
                                        color="red",
                                        fontsize=9,
                                    )
                                    axes[1, 1].set_title(
                                        "Spike Count vs Current", fontsize=9
                                    )

                                try:
                                    phase_v_2, phase_dvdt_2 = None, None
                                    target_sweep_num_2, target_current_pA_2 = None, None
                                    phase_title_suffix_2 = "N/A"
                                    if (
                                        isinstance(analysis_df_2, pd.DataFrame)
                                        and not analysis_df_2.empty
                                        and current_column in analysis_df_2.columns
                                        and "spike_count" in analysis_df_2.columns
                                        and abf_obj_2
                                    ):
                                        try:
                                            spiking_sweeps_2 = analysis_df_2[
                                                (analysis_df_2["spike_count"] >= 1)
                                                & (analysis_df_2[current_column] > 0)
                                            ]
                                            if not spiking_sweeps_2.empty:
                                                rheobase_row_2 = spiking_sweeps_2.loc[
                                                    spiking_sweeps_2[
                                                        current_column
                                                    ].idxmin()
                                                ]
                                                rheobase_current_2 = rheobase_row_2[
                                                    current_column
                                                ]
                                                target_current_2 = (
                                                    2 * rheobase_current_2
                                                )
                                                valid_current_df_2 = (
                                                    analysis_df_2.dropna(
                                                        subset=[current_column]
                                                    )
                                                )
                                                if not valid_current_df_2.empty:
                                                    closest_idx_2 = (
                                                        (
                                                            valid_current_df_2[
                                                                current_column
                                                            ]
                                                            - target_current_2
                                                        )
                                                        .abs()
                                                        .idxmin()
                                                    )
                                                    closest_sweep_row_2 = (
                                                        valid_current_df_2.loc[
                                                            closest_idx_2
                                                        ]
                                                    )
                                                    target_sweep_num_2 = int(
                                                        closest_sweep_row_2["sweep"]
                                                    )
                                                    target_current_pA_2 = (
                                                        closest_sweep_row_2[
                                                            current_column
                                                        ]
                                                    )
                                                    abf_obj_2.setSweep(
                                                        target_sweep_num_2
                                                    )
                                                    phase_v_2 = abf_obj_2.sweepY
                                                    time_s_2 = abf_obj_2.sweepX
                                                    if (
                                                        len(phase_v_2) > 1
                                                        and len(time_s_2) > 1
                                                    ):
                                                        phase_dvdt_2 = np.gradient(
                                                            phase_v_2, time_s_2 * 1000.0
                                                        )
                                                        phase_title_suffix_2 = ""
                                                    else:
                                                        phase_title_suffix_2 = (
                                                            "Sweep Invalid"
                                                        )
                                                else:
                                                    phase_title_suffix_2 = "No Valid I"
                                            else:
                                                phase_title_suffix_2 = "Rheo Not Found"
                                        except Exception as e_inner:
                                            phase_title_suffix_2 = f"Calc Error"
                                            _log_message(
                                                "ERROR",
                                                filename_2,
                                                target_sweep_num_2,
                                                f"PDF Phase Calc Error: {e_inner}",
                                            )
                                    elif load_err_2:
                                        phase_title_suffix_2 = "Load Error"
                                    elif not abf_obj_2:
                                        phase_title_suffix_2 = "No ABF"
                                    elif (
                                        not isinstance(analysis_df_2, pd.DataFrame)
                                        or analysis_df_2.empty
                                    ):
                                        phase_title_suffix_2 = "No Analysis"
                                    else:
                                        phase_title_suffix_2 = "Cols Missing"
                                    # --- End calc logic ---
                                    plot_phase_plane(
                                        phase_v_2,
                                        phase_dvdt_2,
                                        filename_2,
                                        target_sweep_num_2,
                                        target_current_pA_2,
                                        phase_title_suffix_2,
                                        ax=axes[1, 2],
                                    )
                                except Exception as e_plot:
                                    _log_message(
                                        "ERROR",
                                        filename_2,
                                        None,
                                        f"PDF Phase Plot Error: {e_plot}",
                                    )
                                    axes[1, 2].text(
                                        0.5,
                                        0.5,
                                        f"Phase Plot Error:\n{e_plot}",
                                        ha="center",
                                        va="center",
                                        color="red",
                                        fontsize=9,
                                    )
                                    axes[1, 2].set_title("Phase Plane", fontsize=9)
                            else:
                                for col_idx in range(3):
                                    axes[1, col_idx].axis("off")
                                _log_message(
                                    "DEBUG",
                                    "PDF Export",
                                    None,
                                    f"Odd number of files. Last page has only one row.",
                                )

                            fig.tight_layout(pad=1.5, h_pad=2.5)

                            pdf.savefig(fig)

                        except Exception as e_fig:
                            _log_message(
                                "ERROR",
                                "PDF Export",
                                None,
                                f"Failed to create composite PDF page starting with {filename_1}: {e_fig}",
                            )
                        finally:
                            if fig is not None:
                                plt.close(fig)

                    d = pdf.infodict()
                    d["Title"] = "ABF Analysis Summary Plots (2 Files per Page)"
                    d["Author"] = "ABF Spike Analyzer App"
                    d["CreationDate"] = pd.Timestamp.now(tz="UTC")
                    d["ModDate"] = pd.Timestamp.now(tz="UTC")

                pdf_content = pdf_buffer.getvalue()

        except Exception as outer_e:
            _log_message(
                "ERROR",
                "PDF Export",
                None,
                f"Critical error during PDF setup/generation: {outer_e}",
            )
            yield "Error: Failed to generate PDF."
            return

        _log_message("INFO", "PDF Export", None, "PDF generation complete.")

        if pdf_content:
            yield pdf_content
        else:
            _log_message(
                "WARN", "PDF Export", None, "PDF content was empty after generation."
            )
            yield "Error: Generated PDF was empty."


# ==============================================================================
# App Instantiation
# ==============================================================================
app = App(app_ui, server)
