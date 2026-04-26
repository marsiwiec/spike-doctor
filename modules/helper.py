import pyabf
import numpy as np
import pandas as pd
import io
import base64
from typing import Optional, Dict, Any, List
import matplotlib.pyplot as plt

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


def fig_to_src_and_close(fig: Optional[plt.Figure]) -> Optional[str]:
    """Converts a Matplotlib figure to a base64 encoded image source, then closes it."""
    if fig is None:
        return None
    try:
        with io.BytesIO() as buf:
            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            base64_str = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
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
        is_volt_y = "v" in y_units

        # Use the stimulus DAC channel's units rather than sweepUnitsC,
        # which may reflect the wrong DAC when stimulus is on a non-default channel.
        dac_channel = get_stimulus_dac_channel(abf)
        c_units = str(get_dac_units(abf, dac_channel)).lower()
        is_curr_c = any(u in c_units for u in ("pa", "na", "a"))

        return (is_volt_y and is_curr_c) or (is_volt_y and not c_units)
    except Exception:
        return False


def parse_efel_value(raw_efel_result: Optional[Dict[str, Any]], feature_key: str) -> list:
    """
    Safely extracts and converts a single float value or list of values from eFEL results.
    Handles None results, empty lists, NaN, and conversion errors.
    Always returns a list of floats.
    """
    if raw_efel_result is None:
        return [np.nan]
    val = raw_efel_result.get(feature_key)
    if val is None:
        return [np.nan]

    arr = np.atleast_1d(val)
    if len(arr) == 0:
        return [np.nan]

    out = []
    for v in arr:
        try:
            out.append(float(v) if pd.notna(v) else np.nan)
        except (TypeError, ValueError):
            out.append(np.nan)
    return out


def is_valid_analysis_df(df: Any) -> bool:
    """Check if an analysis DataFrame is valid and non-empty."""
    return isinstance(df, pd.DataFrame) and not df.empty


def get_feature_units(feature_name: str, abf: Any = None) -> str:
    """Get the units for a given eFEL feature name."""
    name_lower = feature_name.lower()
    if "resistance" in name_lower:
        return "MΩ"
    if "constant" in name_lower:
        return "ms"
    if "capacitance" in name_lower:
        return "pF"
    if "frequency" in name_lower or "isi" in name_lower:
        return "Hz"
    if "time_to_" in name_lower or "latency" in name_lower:
        return "ms"
    if "voltage" in name_lower or "potential" in name_lower:
        return getattr(abf, "sweepUnitsY", "mV") if abf else "mV"
    return ""


def clean_excel_sheet_name(name: str) -> str:
    """Clean a string to be a valid Excel sheet name (max 31 chars, alphanumeric)."""
    clean_name = str(name).replace("_", " ").title()
    clean_name = "".join(c for c in clean_name if c.isalnum() or c in (" ", "-")).rstrip()
    return clean_name[:31]


def get_stimulus_dac_channel(abf: pyabf.ABF, adc_channel: int = 0) -> int:
    """
    Determine which DAC channel contains the actual stimulus waveform.

    Defaults to adc_channel, but scans other DACs if the default has
    no enabled waveform or no meaningful (non-zero) epochs.
    """
    if not hasattr(abf, "_dacSection"):
        return adc_channel

    n_waveform_enable = getattr(abf._dacSection, "nWaveformEnable", [])
    if adc_channel < len(n_waveform_enable) and n_waveform_enable[adc_channel] == 1:
        try:
            epoch_table = pyabf.waveform.EpochTable(abf, adc_channel)
            if len(epoch_table.epochs) > 0:
                for ep in epoch_table.epochs:
                    if ep.level != 0 or ep.levelDelta != 0:
                        return adc_channel
        except Exception:
            pass

    for dac_ch in range(len(n_waveform_enable)):
        if n_waveform_enable[dac_ch] == 1:
            try:
                epoch_table = pyabf.waveform.EpochTable(abf, dac_ch)
                if len(epoch_table.epochs) > 0:
                    for ep in epoch_table.epochs:
                        if ep.level != 0 or ep.levelDelta != 0:
                            return dac_ch
            except Exception:
                pass

    return adc_channel


def get_dac_units(abf: pyabf.ABF, dac_channel: int) -> str:
    """Return the units string for a specific DAC channel."""
    if hasattr(abf, "_dacSection") and hasattr(abf, "_stringsSection"):
        try:
            units_index = abf._dacSection.lDACChannelUnitsIndex[dac_channel]
            return abf._stringsSection._indexedStrings[units_index]
        except (IndexError, AttributeError):
            pass
    return getattr(abf, "sweepUnitsC", "")


def get_epoch_waveform_for_sweep(abf: pyabf.ABF, sweep_num: int, adc_channel: int = 0):
    """
    Return the EpochSweepWaveform for a given sweep using the correct DAC channel.
    """
    dac_channel = get_stimulus_dac_channel(abf, adc_channel)
    try:
        epoch_table = pyabf.waveform.EpochTable(abf, dac_channel)
        return epoch_table.epochWaveformsBySweep[sweep_num]
    except Exception:
        return None


def get_sweep_c(abf: pyabf.ABF, sweep_num: int, channel: int = 0) -> Optional[np.ndarray]:
    """
    Returns the command current trace (sweepC) for a given sweep in pA.

    If the recorded sweepC is missing, empty, or all zeros/NaNs, this function
    reconstructs the command trace from the sweep epoch table, using the
    correct DAC channel when the stimulus is not on the default DAC.
    """
    try:
        abf.setSweep(sweep_num, channel=channel)
    except Exception:
        return None

    sweep_c = getattr(abf, "sweepC", None)
    has_valid = (
        isinstance(sweep_c, np.ndarray)
        and sweep_c.size > 0
        and not np.all(np.isnan(sweep_c))
        and not np.all(sweep_c == 0)
    )
    if has_valid:
        return sweep_c

    # Reconstruct from the correct DAC channel's epoch table
    try:
        sweep_points = getattr(abf, "sweepPointCount", 0)
        if sweep_points == 0:
            return None

        dac_channel = get_stimulus_dac_channel(abf, channel)
        epoch_waveform = get_epoch_waveform_for_sweep(abf, sweep_num, channel)
        if epoch_waveform is None:
            return None

        p1s = epoch_waveform.p1s
        p2s = epoch_waveform.p2s
        levels = epoch_waveform.levels

        if not (len(p1s) == len(p2s) == len(levels)):
            return None

        unit_raw = get_dac_units(abf, dac_channel).lower()
        if "na" in unit_raw:
            scale = 1000.0  # nA -> pA
        elif "pa" in unit_raw or not unit_raw:
            scale = 1.0
        else:
            scale = 1.0  # Assume pA

        reconstructed = np.zeros(sweep_points, dtype=float)
        for p1, p2, level in zip(p1s, p2s, levels):
            start = max(0, int(p1))
            # p2 == 0 in pyabf often means "to the end of the sweep"
            end = min(sweep_points, int(p2)) if int(p2) > 0 else sweep_points
            if start < end:
                reconstructed[start:end] = level * scale

        return reconstructed
    except Exception:
        return None


def get_file_type_info(abf: pyabf.ABF) -> dict:
    """
    Analyze the ABF file to determine its recording type.

    Returns a dict with keys:
        is_gap_free: bool -- True if gap-free recording mode
        is_stimulus_free: bool -- True if no DAC waveforms are enabled
        is_current_zero: bool -- True if enabled waveform has all-zero epochs
        has_protocol_epochs: bool -- True if there are active protocol epochs
    """
    info = {
        "is_gap_free": False,
        "is_stimulus_free": False,
        "is_current_zero": False,
        "has_protocol_epochs": False,
    }

    # Gap-free detection (ABF operation mode 3)
    op_mode = getattr(abf, "nOperationMode", None)
    if op_mode == 3:
        info["is_gap_free"] = True

    # Check enabled waveforms
    n_waveform_enable = []
    if hasattr(abf, "_dacSection"):
        n_waveform_enable = getattr(abf._dacSection, "nWaveformEnable", [])

    if not any(n_waveform_enable):
        info["is_stimulus_free"] = True
        return info

    # Check protocol epochs on the active/stimulus DAC
    dac_channel = get_stimulus_dac_channel(abf)
    try:
        epoch_table = pyabf.waveform.EpochTable(abf, dac_channel)
        info["has_protocol_epochs"] = len(epoch_table.epochs) > 0
        if info["has_protocol_epochs"]:
            all_zero = all(
                ep.level == 0 and ep.levelDelta == 0
                for ep in epoch_table.epochs
            )
            info["is_current_zero"] = all_zero
    except Exception:
        info["is_stimulus_free"] = True

    return info


def filter_stimulus_dependent_features(features: List[str]) -> List[str]:
    """Remove stimulus-dependent features from the given list."""
    from modules import constants
    return [f for f in features if f not in constants.STIMULUS_DEPENDENT_EFEL_FEATURES]


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
