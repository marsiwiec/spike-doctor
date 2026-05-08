import base64
import io
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyabf

from modules import logger as _logger_mod
from modules import stimulus


def _log_message(level: str, abf_id: str, sweep_num: int | None, message: str):
    _logger_mod.get_logger().log(level, abf_id, sweep_num, message)


def fig_to_src_and_close(fig: plt.Figure | None) -> str | None:
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


def get_abf_info_text(abf: pyabf.ABF | None, filename: str) -> str:
    if abf is None:
        return f"File: {filename}\nError: Could not load ABF object."

    try:
        info_lines = [
            f"File: {filename}",
            f"Protocol: {getattr(abf, 'protocol', 'N/A')}",
            f"ABF Version: {getattr(abf, 'abfVersionString', 'N/A')}",
        ]

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

        info_lines.extend([
            f"Sample Rate: {rate} Hz",
            f"Channels: {getattr(abf, 'channelCount', 'N/A')}",
            f"Sweeps: {getattr(abf, 'sweepCount', 'N/A')}",
            f"Voltage Units: {getattr(abf, 'sweepUnitsY', '?')}",
            f"Current Units: {getattr(abf, 'sweepUnitsC', '?')}",
        ])

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


def is_current_clamp(abf: pyabf.ABF | None) -> bool:
    if abf is None:
        return False
    try:
        y_units = str(getattr(abf, "sweepUnitsY", "")).lower()
        is_volt_y = "v" in y_units

        dac_channel = stimulus.get_stimulus_dac_channel(abf)
        c_units = str(stimulus.get_dac_units(abf, dac_channel)).lower()
        is_curr_c = any(u in c_units for u in ("pa", "na", "a"))

        return (is_volt_y and is_curr_c) or (is_volt_y and not c_units)
    except Exception:
        return False


def parse_efel_value(raw_efel_result: dict[str, Any] | None, feature_key: str) -> list:
    """Safely extract a float or list of floats from eFEL results."""
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
    return isinstance(df, pd.DataFrame) and not df.empty


def get_feature_units(feature_name: str, abf: Any = None) -> str:
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
    clean_name = str(name).replace("_", " ").title()
    clean_name = "".join(
        c for c in clean_name if c.isalnum() or c in (" ", "-")
    ).rstrip()
    return clean_name[:31]


def filter_stimulus_dependent_features(features: list[str]) -> list[str]:
    from modules import constants

    return [f for f in features if f not in constants.STIMULUS_DEPENDENT_EFEL_FEATURES]


def _validate_abf_for_analysis(abf: pyabf.ABF, abf_id_str: str) -> bool:
    checks = [
        (hasattr(abf, "sweepCount") and abf.sweepCount > 0, "No sweeps found."),
        (hasattr(abf, "dataRate") and abf.dataRate > 0, "Invalid data rate."),
        (
            hasattr(abf, "sweepPointCount") and abf.sweepPointCount > 0,
            "Invalid sweep point count.",
        ),
        (
            hasattr(abf, "dataPointsPerMs") and abf.dataPointsPerMs > 0,
            "Invalid data points per ms.",
        ),
    ]

    for ok, msg in checks:
        if not ok:
            _log_message("ERROR" if "Invalid" in msg else "WARN", abf_id_str, None, msg)
            return False

    if not is_current_clamp(abf):
        _log_message(
            "WARN",
            abf_id_str,
            None,
            "File may not be current clamp. Attempting analysis anyway.",
        )

    return True
