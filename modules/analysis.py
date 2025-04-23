import traceback
import warnings
from typing import List, Dict, Optional, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyabf
import efel

from modules import constants, helper


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
    current_col_name: str = constants.CURRENT_COL_NAME,
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
        helper._log_message(
            "ERROR",
            abf_id_str,
            None,
            "Invalid or missing ABF object passed to analysis.",
        )
        return analysis_output
    abf_id_str = getattr(abf, "abfID", original_filename)

    # 1. Initial Validation
    if not helper._validate_abf_for_analysis(abf, abf_id_str):
        return analysis_output


    # 4. Determine Current Unit Conversion Factor (raw units -> pA)
    current_unit_factor = 1.0  # Assume pA by default
    current_unit_raw = getattr(abf, "sweepUnitsC", "").lower()
    output_plot_unit = "pA"  # For debug plot label
    if "na" in current_unit_raw:
        current_unit_factor = 1000.0  # nA to pA
        helper._log_message(
            "DEBUG", abf_id_str, None, "Detected nA command units. Factor = 1000."
        )
    elif not current_unit_raw or "pa" not in current_unit_raw:
        output_plot_unit = f"{current_unit_raw or '?'} (Assumed pA)"
        helper._log_message(
            "WARN",
            abf_id_str,
            None,
            f"Command units are '{current_unit_raw}'. Assuming pA.",
        )

    # Prepare eFEL
    all_efel_features_needed = list(
        set(user_selected_features) | set(constants.REQUIRED_INTERNAL_EFEL_FEATURES)
    )
    try:
        efel.reset()
        efel.set_setting("strict_stiminterval", True)
        efel.set_setting("Threshold", detection_threshold)
        efel.set_setting("DerivativeThreshold", derivative_threshold)
        helper._log_message(
            "DEBUG",
            abf_id_str,
            None,
            f"Configured eFEL. Requesting {len(all_efel_features_needed)} features.",
        )
    except Exception as e:
        helper._log_message(
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
                )  
                warnings.simplefilter(
                    "ignore", DeprecationWarning
                )  
                try:
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
                    efel_error = e  

                # Store relevant warnings (filter out known safe ones if needed)
                efel_warnings = [
                    str(w.message)
                    for w in caught_warnings
                    if issubclass(w.category, RuntimeWarning)
                ]

            if efel_error:
                tb_str = traceback.format_exc()
                helper._log_message(
                    "ERROR",
                    abf_id_str,
                    sweep_num,
                    f"eFEL feature extraction failed: {efel_error}\nTraceback:\n{tb_str}",
                )
                efel_results_raw = None
            if efel_warnings:
                # Log only the first warning to avoid spamming console
                summary_warning = (
                    f"{efel_warnings[0]}{'...' if len(efel_warnings) > 1 else ''}"
                )
                helper._log_message(
                    "WARN", abf_id_str, sweep_num, f"eFEL warnings: {summary_warning}"
                )

            # --- Parse eFEL Results Safely ---
            efel_results_parsed = {
                feat: helper.parse_efel_value(efel_results_raw, feat)
                for feat in all_efel_features_needed
            }

            # --- Manual Calculations (Cm) ---

            Cm_manual_pF = np.nan
            tau_efel_ms = efel_results_parsed.get("time_constant", np.nan)
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
                    Cm_manual_pF = (tau_efel_ms / R_in_MOhm) * 1000.0
                    if not np.isfinite(Cm_manual_pF):
                        Cm_manual_pF = np.nan
                except (ZeroDivisionError, FloatingPointError) as div_err:
                    helper._log_message(
                        "WARN",
                        abf_id_str,
                        sweep_num,
                        f"Division error during Cm calc: {div_err}",
                    )
                    Cm_manual_pF = np.nan
                except Exception as cm_err:
                    helper._log_message(
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
                constants.CURRENT_COL_NAME: trace_for_efel["stimulus_current"][0] * 1000,
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
            if stimulus_current_for_efel_nA >= 0:
                sweep_data["time_constant"] = np.nan
                sweep_data["capacitance_pF"] = np.nan
                sweep_data["ohmic_input_resistance"] = np.nan


            sweep_results_list.append(sweep_data)

            # --- Generate Debug Plot (only once per file, for a designated sweep) ---
            # Check flag and if current sweep is the target debug sweep
            if (
                debug_plot
                and sweep_num == middle_sweep_index_for_debug
                and not debug_plot_generated
            ):
                fig_debug = None  
                try:
                    fig_debug, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
                    fig_debug.set_layout_engine("tight")  

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
                        color="salmon",  
                        alpha=0.2,
                        zorder=-10,  # Place behind data
                    )
                    # Plot V_base from eFEL (if valid)
                    if pd.notna(V_base_efel):
                        axs[0].axhline(
                            V_base_efel,
                            color="darkorchid",  
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
                            color="royalblue",  
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

                    analysis_output["debug_plot_fig"] = fig_debug
                    debug_plot_generated = True  # Set flag
                    helper._log_message(
                        "DEBUG", abf_id_str, sweep_num, "Debug plot generated."
                    )

                except Exception as plot_err:
                    helper._log_message(
                        "ERROR",
                        abf_id_str,
                        sweep_num,
                        f"Debug plot generation failed: {plot_err}\n{traceback.format_exc()}",
                    )
                    analysis_output["debug_plot_fig"] = (
                        None  
                    )
                    if fig_debug is not None:
                        plt.close(fig_debug)  

        # --- End of Sweep Loop ---

    except Exception as loop_err:
        helper._log_message(
            "ERROR",
            abf_id_str,
            None,
            f"Critical error during sweep processing: {loop_err}",
        )
        traceback.print_exc()
        
        if sweep_results_list:
            helper._log_message(
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
        helper._log_message(
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
            current_col_name,
            "input_resistance_Mohm",
            "time_constant_ms",
            "capacitance_pF",
        ]

        efel_cols_present = sorted(
            [f for f in user_selected_features if f in analysis_df.columns]
        )

        final_ordered_cols = standard_cols + efel_cols_present
        final_ordered_cols = [
            col for col in final_ordered_cols if col in analysis_df.columns
        ]

        analysis_df = analysis_df.reindex(columns=final_ordered_cols)

        analysis_df = analysis_df.sort_values(by=["filename", "sweep"]).reset_index(
            drop=True
        )

        analysis_output["analysis_df"] = analysis_df
        helper._log_message(
            "DEBUG",
            abf_id_str,
            None,
            f"Analysis complete. DataFrame shape: {analysis_df.shape}",
        )

    except Exception as df_err:
        helper._log_message(
            "ERROR", abf_id_str, None, f"Failed to assemble final DataFrame: {df_err}"
        )
        analysis_output["analysis_df"] = None

    return analysis_output
