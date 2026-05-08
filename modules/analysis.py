import traceback
import warnings

import efel
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyabf

from modules import constants, helper, stimulus
from modules.models import AnalysisResult


def _build_efel_trace(
    abf: pyabf.ABF,
    sweep_num: int,
    channel: int,
    stimulus_epoch_index: int,
) -> tuple[dict, float]:
    abf.setSweep(sweep_num, channel=channel)

    window = stimulus.find_stimulus_window(
        abf, sweep_num, channel, stimulus_epoch_index
    )
    stim_start_ms = window.start_pt * abf.dataSecPerPoint * 1000.0
    stim_end_ms = window.end_pt * abf.dataSecPerPoint * 1000.0

    stimulus_current_pA = stimulus.compute_stimulus_current(
        abf, sweep_num, channel, window
    )
    stimulus_current_nA = stimulus_current_pA / 1000.0

    trace = {
        "T": abf.sweepX * 1000.0,
        "V": abf.sweepY,
        "stim_start": [stim_start_ms],
        "stim_end": [stim_end_ms],
        "stimulus_current": [float(stimulus_current_nA)],
    }
    return trace, stimulus_current_pA


def _sanitize_sweep_data(
    sweep_data: dict, spike_count: int, stimulus_current_pA: float
) -> None:
    for feature in list(sweep_data.keys()):
        name = feature.lower()
        if spike_count <= 1 and ("frequency" in name or "isi" in name):
            sweep_data[feature] = np.nan
        if spike_count == 0 and ("time_to_" in name or "latency" in name):
            sweep_data[feature] = np.nan

    if spike_count > 0 or stimulus_current_pA >= 0:
        sweep_data["capacitance_pF"] = np.nan

    if stimulus_current_pA >= 0:
        for feature in ("time_constant", "ohmic_input_resistance"):
            if feature in sweep_data:
                sweep_data[feature] = np.nan


def _assemble_dataframe(
    sweep_results_list: list[dict],
    current_col_name: str,
) -> pd.DataFrame | None:
    if not sweep_results_list:
        return None
    try:
        df = pd.DataFrame(sweep_results_list)
        priority_cols = [
            "filename", "sweep", current_col_name, "event_index", "capacitance_pF"
        ]
        efel_cols = sorted([c for c in df.columns if c not in priority_cols])
        ordered_cols = [c for c in priority_cols if c in df.columns] + efel_cols
        return (
            df[ordered_cols]
            .sort_values(by=["filename", "sweep"])
            .reset_index(drop=True)
        )
    except Exception as df_err:
        helper._log_message(
            "ERROR", "App", None,
            f"Failed to assemble final DataFrame: {df_err}"
        )
        return None


def _configure_efel(detection_threshold: float, derivative_threshold: float) -> None:
    """Reset and configure eFEL settings."""
    efel.reset()
    efel.api.set_setting("strict_stiminterval", True)
    efel.api.set_setting("Threshold", detection_threshold)
    efel.api.set_setting("DerivativeThreshold", derivative_threshold)


def _run_efel_extraction(
    trace: dict, features: list[str]
) -> tuple[dict | None, list[str]]:
    """Run eFEL feature extraction and return (results, warnings)."""
    efel_results_raw = None
    efel_warnings: list[str] = []

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", RuntimeWarning)
        warnings.simplefilter("ignore", DeprecationWarning)
        try:
            efel_results_list = efel.get_feature_values(
                [trace], features, raise_warnings=True
            )
            if efel_results_list:
                efel_results_raw = efel_results_list[0]
        except Exception:
            pass

        efel_warnings = [
            str(w.message)
            for w in caught_warnings
            if issubclass(w.category, RuntimeWarning)
        ]

    return efel_results_raw, efel_warnings


def _compute_capacitance(tau_ms: float, R_in_MOhm: float) -> float:
    """Compute membrane capacitance from tau and R_in."""
    if pd.notna(tau_ms) and pd.notna(R_in_MOhm) and tau_ms > 0 and R_in_MOhm > 0:
        Cm = (tau_ms / R_in_MOhm) * 1000.0
        if np.isfinite(Cm):
            return Cm
    return np.nan


def _build_sweep_rows(
    sweep_num: int,
    original_filename: str,
    stimulus_current_pA: float,
    Cm_manual_pF: float,
    efel_results_parsed: dict[str, list],
    effective_user_features: list[str],
) -> list[dict]:
    """Build data rows for a single sweep, expanding list-valued features."""
    max_len = 1
    for feat in effective_user_features:
        val = efel_results_parsed.get(feat, [np.nan])
        if isinstance(val, list) and len(val) > max_len:
            max_len = len(val)

    rows = []
    for i in range(max_len):
        sweep_data = {
            "filename": original_filename,
            "sweep": sweep_num,
            constants.CURRENT_COL_NAME: stimulus_current_pA,
            "event_index": i,
            "capacitance_pF": Cm_manual_pF if i == 0 else np.nan,
        }
        for feat in effective_user_features:
            val_list = efel_results_parsed.get(feat, [np.nan])
            sweep_data[feat] = val_list[i] if i < len(val_list) else np.nan
        rows.append(sweep_data)

    return rows


def run_analysis_on_abf(
    abf: pyabf.ABF | None,
    original_filename: str,
    user_selected_features: list[str],
    channel_selection: int,
    stimulus_epoch_index: int,
    detection_threshold: float,
    derivative_threshold: float,
    current_col_name: str = constants.CURRENT_COL_NAME,
) -> AnalysisResult:
    """Analyze a single ABF file using eFEL and manual calculations."""
    result = AnalysisResult()
    abf_id_str = original_filename

    if not isinstance(abf, pyabf.ABF):
        helper._log_message(
            "ERROR", abf_id_str, None,
            "Invalid or missing ABF object passed to analysis."
        )
        return result
    abf_id_str = getattr(abf, "abfID", original_filename)

    if not helper._validate_abf_for_analysis(abf, abf_id_str):
        return result

    file_info = stimulus.get_file_type_info(abf)
    effective_user_features = list(user_selected_features)
    effective_internal_features = list(constants.REQUIRED_INTERNAL_EFEL_FEATURES)

    if (
        file_info["is_stimulus_free"]
        or file_info["is_current_zero"]
        or file_info["is_gap_free"]
    ):
        before = set(effective_user_features) | set(effective_internal_features)
        effective_user_features[:] = helper.filter_stimulus_dependent_features(
            effective_user_features
        )
        effective_internal_features[:] = helper.filter_stimulus_dependent_features(
            effective_internal_features
        )
        after = set(effective_user_features) | set(effective_internal_features)
        omitted = sorted(before - after)
        if omitted:
            helper._log_message(
                "INFO", abf_id_str, None,
                "Stimulus-free/current-zero/gap-free file detected. "
                f"Omitting: {', '.join(omitted)}.",
            )

    all_efel_features_needed = list(
        set(effective_user_features) | set(effective_internal_features)
    )
    try:
        _configure_efel(detection_threshold, derivative_threshold)
    except Exception as e:
        helper._log_message(
            "ERROR", abf_id_str, None,
            f"Failed to configure eFEL settings: {e}. Aborting."
        )
        return result

    sweep_results_list: list[dict] = []

    try:
        for sweep_num in abf.sweepList:
            try:
                trace, stimulus_current_pA = _build_efel_trace(
                    abf, sweep_num, channel_selection, stimulus_epoch_index
                )
            except ValueError as e:
                helper._log_message(
                    "ERROR", abf_id_str, sweep_num,
                    f"Channel {channel_selection} not found: {e}"
                )
                continue

            efel_results_raw, efel_warnings = _run_efel_extraction(
                trace, all_efel_features_needed
            )

            if efel_results_raw is None:
                helper._log_message(
                    "ERROR", abf_id_str, sweep_num,
                    "eFEL feature extraction failed.\n" + traceback.format_exc()
                )
            if efel_warnings:
                summary = (
                    f"{efel_warnings[0]}"
                    f"{'...' if len(efel_warnings) > 1 else ''}"
                )
                helper._log_message(
                    "WARN", abf_id_str, sweep_num, f"eFEL warnings: {summary}"
                )

            efel_results_parsed = {
                feat: helper.parse_efel_value(efel_results_raw, feat)
                for feat in all_efel_features_needed
            }

            tau_ms = efel_results_parsed.get("time_constant", [np.nan])[0]
            R_in_MOhm = efel_results_parsed.get(
                "ohmic_input_resistance", [np.nan]
            )[0]
            spike_count_val = efel_results_parsed.get("spike_count", [np.nan])[0]
            spike_count = 0 if pd.isna(spike_count_val) else int(spike_count_val)

            Cm_manual_pF = _compute_capacitance(tau_ms, R_in_MOhm)

            rows = _build_sweep_rows(
                sweep_num,
                original_filename,
                stimulus_current_pA,
                Cm_manual_pF,
                efel_results_parsed,
                effective_user_features,
            )

            for row in rows:
                _sanitize_sweep_data(row, spike_count, stimulus_current_pA)
                sweep_results_list.append(row)

    except Exception as loop_err:
        helper._log_message(
            "ERROR", abf_id_str, None,
            f"Critical error during sweep processing: {loop_err}"
        )
        traceback.print_exc()
        if not sweep_results_list:
            return result
        helper._log_message(
            "WARN", abf_id_str, None,
            "Returning partial results due to error in loop."
        )

    if not sweep_results_list:
        helper._log_message(
            "WARN", abf_id_str, None,
            "Analysis finished, but no sweep results were generated."
        )
        result.analysis_df = pd.DataFrame()
        return result

    result.analysis_df = _assemble_dataframe(
        sweep_results_list, current_col_name
    )
    return result


def generate_debug_plot(
    abf: pyabf.ABF,
    sweep_num: int,
    channel: int,
    stimulus_epoch_index: int,
    detection_threshold: float,
    derivative_threshold: float,
) -> plt.Figure | None:
    """Generate a debug plot for a single sweep with spike markers.

    Returns the figure or None on failure. Caller must close the figure.
    """
    abf_id_str = getattr(abf, "abfID", "?")
    fig = None

    try:
        trace, _ = _build_efel_trace(
            abf, sweep_num, channel, stimulus_epoch_index
        )
    except Exception as e:
        helper._log_message(
            "ERROR", abf_id_str, sweep_num,
            f"Failed to build trace for debug plot: {e}"
        )
        return None

    _configure_efel(detection_threshold, derivative_threshold)

    debug_features = ["peak_time", "peak_voltage", "spike_count", "voltage_base"]
    efel_results_raw, _ = _run_efel_extraction(trace, debug_features)
    efel_results_parsed = {
        feat: helper.parse_efel_value(efel_results_raw, feat)
        for feat in debug_features
    }

    peak_times = efel_results_parsed.get("peak_time", [])
    peak_voltages = efel_results_parsed.get("peak_voltage", [])
    V_base = efel_results_parsed.get("voltage_base", [np.nan])[0]

    try:
        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
        fig.set_layout_engine("tight")

        # Voltage trace
        axs[0].plot(
            abf.sweepX, abf.sweepY, color="black", lw=0.7, label=f"Sweep {sweep_num}"
        )
        axs[0].axvspan(
            xmin=trace["stim_start"][0] / 1000.0,
            xmax=trace["stim_end"][0] / 1000.0,
            color="salmon", alpha=0.2, zorder=-10,
        )
        axs[0].axhline(
            detection_threshold, color="grey", linestyle="--", lw=1.0,
            label=f"Detection threshold: {detection_threshold:.0f} mV", zorder=0,
        )
        if pd.notna(V_base):
            axs[0].axhline(
                V_base, color="darkorchid", linestyle=":", lw=1.5,
                label=f"V_base (eFEL): {V_base:.1f} mV", zorder=0,
            )
        # Spike peak markers
        if len(peak_times) > 0 and len(peak_voltages) > 0:
            peak_times_s = np.array(peak_times) / 1000.0
            axs[0].plot(
                peak_times_s, peak_voltages, "ro", markersize=5,
                label=f"Spikes: {len(peak_times)}", zorder=5,
            )
        axs[0].set_ylabel(f"Voltage ({getattr(abf, 'sweepUnitsY', 'mV')})")
        axs[0].set_title(
            f"Debug Plot: {abf_id_str} - Sweep {sweep_num}", fontsize=10
        )
        axs[0].legend(fontsize=7, loc="best")
        axs[0].grid(True, linestyle=":", alpha=0.5)

        # Command current
        sweep_c = stimulus.get_sweep_c(abf, sweep_num, channel=channel)
        if sweep_c is not None:
            axs[1].plot(abf.sweepX, sweep_c, color="royalblue", lw=0.7)
        else:
            axs[1].text(
                0.5, 0.5, "Sweep Command (sweepC)\nNot Available",
                ha="center", va="center", color="red", transform=axs[1].transAxes,
            )
        axs[1].axvspan(
            xmin=trace["stim_start"][0] / 1000.0,
            xmax=trace["stim_end"][0] / 1000.0,
            color="salmon", alpha=0.2, zorder=-10,
        )
        axs[1].grid(True, linestyle=":", alpha=0.5)
        dac_channel = stimulus.get_stimulus_dac_channel(abf, channel)
        c_units = stimulus.get_dac_units(abf, dac_channel) or "pA"
        axs[1].set_ylabel(f"Current ({c_units})")
        axs[1].set_xlabel("Time (s)")

        return fig
    except Exception as plot_err:
        helper._log_message(
            "ERROR", abf_id_str, sweep_num,
            f"Debug plot generation failed: {plot_err}\n{traceback.format_exc()}"
        )
        if fig is not None:
            plt.close(fig)
        return None
