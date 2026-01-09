import pyabf
import numpy as np
import pandas as pd
import io
import base64
from typing import Optional, Dict, Any
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


def fig_to_src(fig: Optional[plt.Figure]) -> Optional[str]:
    """Converts a Matplotlib figure to a base64 encoded image source."""
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
