import traceback
import warnings
from typing import Dict, List, Optional, Union

import efel
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyabf

from modules import constants, helper


def _find_stimulus_window(
    abf: pyabf.ABF,
    sweep_num: int,
    channel: int,
    stimulus_epoch_index: int,
):
    epoch_waveform = helper.get_epoch_waveform_for_sweep(abf, sweep_num, channel)
    num_epochs = len(epoch_waveform.p1s) if epoch_waveform else 0
    abf_id = getattr(abf, "abfID", "?")

    if num_epochs == 0:
        sweep_points = getattr(abf, "sweepPointCount", 0)
        start = max(1, sweep_points // 64)
        helper._log_message("INFO", abf_id, sweep_num, "No stimulus epochs found. Treating as free-running.")
        return start, sweep_points, None, epoch_waveform

    if num_epochs == 2:
        helper._log_message("INFO", abf_id, sweep_num, "No protocol epochs found. Using post-holding period.")
        return epoch_waveform.p1s[1], epoch_waveform.p2s[1], 1, epoch_waveform

    if num_epochs > 2:
        if stimulus_epoch_index < num_epochs:
            return (
                epoch_waveform.p1s[stimulus_epoch_index],
                epoch_waveform.p2s[stimulus_epoch_index],
                stimulus_epoch_index,
                epoch_waveform,
            )
        helper._log_message(
            "WARN", abf_id, sweep_num,
            f"Epoch index {stimulus_epoch_index} out of range. Using epoch 1.",
        )
        return epoch_waveform.p1s[1], epoch_waveform.p2s[1], 1, epoch_waveform

    # num_epochs == 1 or unexpected
    sweep_points = getattr(abf, "sweepPointCount", 0)
    start = max(1, sweep_points // 64)
    helper._log_message("WARN", abf_id, sweep_num, f"Unexpected epoch count {num_epochs}. Using full sweep.")
    return start, sweep_points, None, epoch_waveform


def _compute_stimulus_current(
    abf: pyabf.ABF,
    sweep_num: int,
    channel: int,
    stim_start_pt: float,
    stim_end_pt: float,
    epoch_idx_used: Optional[int],
    epoch_waveform,
) -> float:
    sweep_c = helper.get_sweep_c(abf, sweep_num, channel=channel)
    stimulus_current_pA = np.nan

    if sweep_c is not None and len(sweep_c) > int(stim_start_pt):
        end_pt = min(int(stim_end_pt), len(sweep_c)) if int(stim_end_pt) > 0 else len(sweep_c)
        if end_pt > int(stim_start_pt):
            stimulus_current_pA = float(np.median(sweep_c[int(stim_start_pt):end_pt]))

    if np.isfinite(stimulus_current_pA) and not np.isclose(stimulus_current_pA, 0):
        return stimulus_current_pA

    if epoch_waveform is not None and epoch_idx_used is not None:
        raw_level = epoch_waveform.levels[epoch_idx_used]
    else:
        raw_level = 0.0

    dac_channel = helper.get_stimulus_dac_channel(abf, channel)
    unit_raw = helper.get_dac_units(abf, dac_channel).lower()

    if "na" in unit_raw:
        stimulus_current_pA = raw_level * 1000.0
    elif "pa" in unit_raw or not unit_raw:
        stimulus_current_pA = raw_level
    else:
        helper._log_message(
            "WARN", getattr(abf, "abfID", "?"), sweep_num,
            f"Command units '{unit_raw}'. Assuming pA.",
        )
        stimulus_current_pA = raw_level

    if not np.isfinite(stimulus_current_pA):
        helper._log_message(
            "WARN", getattr(abf, "abfID", "?"), sweep_num,
            "Stimulus current undetermined. Defaulting to 0 pA.",
        )
        stimulus_current_pA = 0.0

    return stimulus_current_pA


def _build_efel_trace(
    abf: pyabf.ABF,
    sweep_num: int,
    channel: int,
    stimulus_epoch_index: int,
) -> tuple[dict, float]:
    abf.setSweep(sweep_num, channel=channel)

    stim_start_pt, stim_end_pt, epoch_idx_used, epoch_waveform = _find_stimulus_window(
        abf, sweep_num, channel, stimulus_epoch_index
    )
    stim_start_ms = stim_start_pt * abf.dataSecPerPoint * 1000.0
    stim_end_ms = stim_end_pt * abf.dataSecPerPoint * 1000.0

    stimulus_current_pA = _compute_stimulus_current(
        abf, sweep_num, channel, stim_start_pt, stim_end_pt, epoch_idx_used, epoch_waveform
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


def _generate_debug_plot(
    abf: pyabf.ABF,
    sweep_num: int,
    trace: dict,
    V_base: float,
    abf_id_str: str,
    channel: int = 0,
) -> Optional[plt.Figure]:
    fig = None
    try:
        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
        fig.set_layout_engine("tight")

        axs[0].plot(abf.sweepX, abf.sweepY, color="black", lw=0.7, label=f"Sweep {sweep_num}")
        axs[0].axvspan(
            xmin=trace["stim_start"][0] / 1000.0,
            xmax=trace["stim_end"][0] / 1000.0,
            color="salmon", alpha=0.2, zorder=-10,
        )
        if pd.notna(V_base):
            axs[0].axhline(
                V_base, color="darkorchid", linestyle=":", lw=1.5,
                label=f"V_base (eFEL): {V_base:.1f} mV", zorder=0,
            )
        axs[0].set_ylabel(f"Voltage ({getattr(abf, 'sweepUnitsY', 'mV')})")
        axs[0].set_title(f"Debug Plot: {abf_id_str} - Sweep {sweep_num}", fontsize=10)
        axs[0].legend(fontsize=7, loc="best")
        axs[0].grid(True, linestyle=":", alpha=0.5)

        sweep_c = helper.get_sweep_c(abf, sweep_num, channel=channel)
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
        dac_channel = helper.get_stimulus_dac_channel(abf, channel)
        c_units = helper.get_dac_units(abf, dac_channel) or "pA"
        axs[1].set_ylabel(f"Current ({c_units})")
        axs[1].set_xlabel("Time (s)")

        return fig
    except Exception as plot_err:
        helper._log_message(
            "ERROR", abf_id_str, sweep_num,
            f"Debug plot generation failed: {plot_err}\n{traceback.format_exc()}",
        )
        if fig is not None:
            plt.close(fig)
        return None


def _sanitize_sweep_data(sweep_data: dict, spike_count: int, stimulus_current_pA: float) -> None:
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
    sweep_results_list: List[dict],
    current_col_name: str,
) -> Optional[pd.DataFrame]:
    if not sweep_results_list:
        return None
    try:
        df = pd.DataFrame(sweep_results_list)
        priority_cols = ["filename", "sweep", current_col_name, "event_index", "capacitance_pF"]
        efel_cols = sorted([c for c in df.columns if c not in priority_cols])
        ordered_cols = [c for c in priority_cols if c in df.columns] + efel_cols
        return df[ordered_cols].sort_values(by=["filename", "sweep"]).reset_index(drop=True)
    except Exception as df_err:
        helper._log_message("ERROR", "App", None, f"Failed to assemble final DataFrame: {df_err}")
        return None


def run_analysis_on_abf(
    abf: Optional[pyabf.ABF],
    original_filename: str,
    user_selected_features: List[str],
    channel_selection: int,
    stimulus_epoch_index: int,
    detection_threshold: float,
    derivative_threshold: float,
    debug_plot: bool = True,
    current_col_name: str = constants.CURRENT_COL_NAME,
) -> Dict[str, Union[Optional[pd.DataFrame], Optional[plt.Figure]]]:
    """Analyze a single ABF file using eFEL and manual calculations.

    Returns {'analysis_df': DataFrame or None, 'debug_plot_fig': Figure or None}.
    """
    analysis_output = {"analysis_df": None, "debug_plot_fig": None}
    abf_id_str = original_filename

    if not isinstance(abf, pyabf.ABF):
        helper._log_message("ERROR", abf_id_str, None, "Invalid or missing ABF object passed to analysis.")
        return analysis_output
    abf_id_str = getattr(abf, "abfID", original_filename)

    if not helper._validate_abf_for_analysis(abf, abf_id_str):
        return analysis_output

    file_info = helper.get_file_type_info(abf)
    effective_user_features = list(user_selected_features)
    effective_internal_features = list(constants.REQUIRED_INTERNAL_EFEL_FEATURES)

    if file_info["is_stimulus_free"] or file_info["is_current_zero"] or file_info["is_gap_free"]:
        before = set(effective_user_features) | set(effective_internal_features)
        effective_user_features[:] = helper.filter_stimulus_dependent_features(effective_user_features)
        effective_internal_features[:] = helper.filter_stimulus_dependent_features(effective_internal_features)
        after = set(effective_user_features) | set(effective_internal_features)
        omitted = sorted(before - after)
        if omitted:
            helper._log_message(
                "INFO", abf_id_str, None,
                f"Stimulus-free/current-zero/gap-free file detected. Omitting: {', '.join(omitted)}.",
            )

    all_efel_features_needed = list(set(effective_user_features) | set(effective_internal_features))
    try:
        efel.reset()
        efel.api.set_setting("strict_stiminterval", True)
        efel.api.set_setting("Threshold", detection_threshold)
        efel.api.set_setting("DerivativeThreshold", derivative_threshold)
    except Exception as e:
        helper._log_message("ERROR", abf_id_str, None, f"Failed to configure eFEL settings: {e}. Aborting.")
        return analysis_output

    sweep_results_list = []
    debug_plot_generated = False
    middle_sweep_idx = abf.sweepCount // 2
    middle_sweep = abf.sweepList[middle_sweep_idx] if abf.sweepList else 0

    try:
        for sweep_num in abf.sweepList:
            try:
                trace, stimulus_current_pA = _build_efel_trace(
                    abf, sweep_num, channel_selection, stimulus_epoch_index
                )
            except ValueError as e:
                helper._log_message("ERROR", abf_id_str, sweep_num, f"Channel {channel_selection} not found: {e}")
                continue

            efel_results_raw = None
            efel_error = None
            efel_warnings = []
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always", RuntimeWarning)
                warnings.simplefilter("ignore", DeprecationWarning)
                try:
                    efel_results_list = efel.get_feature_values(
                        [trace], all_efel_features_needed, raise_warnings=True
                    )
                    if efel_results_list:
                        efel_results_raw = efel_results_list[0]
                    else:
                        efel_error = ValueError("eFEL returned empty list unexpectedly.")
                except Exception as e:
                    efel_error = e

                efel_warnings = [
                    str(w.message)
                    for w in caught_warnings
                    if issubclass(w.category, RuntimeWarning)
                ]

            if efel_error:
                helper._log_message(
                    "ERROR", abf_id_str, sweep_num,
                    f"eFEL feature extraction failed: {efel_error}\n{traceback.format_exc()}",
                )
                efel_results_raw = None
            if efel_warnings:
                summary = f"{efel_warnings[0]}{'...' if len(efel_warnings) > 1 else ''}"
                helper._log_message("WARN", abf_id_str, sweep_num, f"eFEL warnings: {summary}")

            efel_results_parsed = {
                feat: helper.parse_efel_value(efel_results_raw, feat)
                for feat in all_efel_features_needed
            }

            Cm_manual_pF = np.nan
            tau_ms = efel_results_parsed.get("time_constant", [np.nan])[0]
            R_in_MOhm = efel_results_parsed.get("ohmic_input_resistance", [np.nan])[0]
            spike_count_val = efel_results_parsed.get("spike_count", [np.nan])[0]
            spike_count = 0 if pd.isna(spike_count_val) else int(spike_count_val)
            V_base = efel_results_parsed.get("voltage_base", [np.nan])[0]

            if pd.notna(tau_ms) and pd.notna(R_in_MOhm) and tau_ms > 0 and R_in_MOhm > 0:
                Cm_manual_pF = (tau_ms / R_in_MOhm) * 1000.0
                if not np.isfinite(Cm_manual_pF):
                    Cm_manual_pF = np.nan

            max_len = 1
            for feat in effective_user_features:
                val = efel_results_parsed.get(feat, [np.nan])
                if isinstance(val, list) and len(val) > max_len:
                    max_len = len(val)

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

                _sanitize_sweep_data(sweep_data, spike_count, stimulus_current_pA)
                sweep_results_list.append(sweep_data)

            if debug_plot and sweep_num == middle_sweep and not debug_plot_generated:
                fig_debug = _generate_debug_plot(
                    abf, sweep_num, trace, V_base, abf_id_str, channel=channel_selection
                )
                if fig_debug is not None:
                    analysis_output["debug_plot_fig"] = fig_debug
                    debug_plot_generated = True

    except Exception as loop_err:
        helper._log_message("ERROR", abf_id_str, None, f"Critical error during sweep processing: {loop_err}")
        traceback.print_exc()
        if not sweep_results_list:
            return analysis_output
        helper._log_message("WARN", abf_id_str, None, "Returning partial results due to error in loop.")

    if not sweep_results_list:
        helper._log_message("WARN", abf_id_str, None, "Analysis finished, but no sweep results were generated.")
        analysis_output["analysis_df"] = pd.DataFrame()
        return analysis_output

    analysis_output["analysis_df"] = _assemble_dataframe(sweep_results_list, current_col_name)
    return analysis_output
