import traceback
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pyabf
from shiny import reactive, render, req, ui

from modules import analysis, constants, helper, plotting
from modules.logger import SessionLogger, set_logger
from server.downloads import register_download_handlers
from ui.layout import ADVANCED_EFEL_FEATURES, AVAILABLE_EFEL_FEATURES


def _file_plot_block(i: int, filename: str, content):
    return ui.div(
        ui.hr() if i > 0 else None,
        ui.h5(filename),
        ui.row(ui.column(12, content)),
    )


def server(input, output, session):
    session_logger = SessionLogger()
    set_logger(session_logger)

    loaded_abf_data = reactive.Value([])

    @reactive.Calc
    def selected_efel_features() -> list:
        features = [
            f
            for f in constants.BASIC_EFEL_FEATURES
            if f in AVAILABLE_EFEL_FEATURES
            and getattr(input, f"basic_feature_{f}")()
        ]
        try:
            features.extend(input.advanced_efel_features() or [])
        except Exception:
            pass
        return features

    @reactive.Effect
    @reactive.event(input.abf_files)
    def _load_abf_files():
        file_infos = input.abf_files()
        if not file_infos:
            loaded_abf_data.set([])
            helper._log_message("INFO", "App", None, "File selection cleared.")
            return

        data_list = []
        num_files = len(file_infos)
        helper._log_message("INFO", "App", None, f"Loading {num_files} ABF file(s)...")
        with ui.Progress(min=0, max=num_files) as p:
            p.set(message="Loading ABF files", detail="Starting...")
            for i, file_info in enumerate(file_infos):
                filename = file_info["name"]
                filepath = Path(file_info["datapath"]).resolve()
                p.set(i, detail=f"Loading {filename}...")
                abf_obj, error_msg = None, None
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        abf_obj = pyabf.ABF(str(filepath), loadData=True)
                except FileNotFoundError:
                    error_msg = "File not found at temporary path."
                    helper._log_message(
                        "ERROR",
                        filename,
                        None,
                        f"{error_msg} Path: {filepath}",
                    )
                except Exception as e:
                    error_msg = f"Failed to load: {e}"
                    helper._log_message(
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
        helper._log_message(
            "INFO",
            "App",
            None,
            f"Finished loading {len(data_list)} files.",
        )

        # Update debug sweep slider to the max sweeps across files
        max_sweeps = max(
            (
                getattr(d.get("abf_object"), "sweepCount", 0)
                for d in data_list
                if d.get("abf_object")
            ),
            default=0,
        )
        if max_sweeps > 0:
            ui.update_slider(
                "debug_sweep",
                max=max_sweeps - 1,
                value=max_sweeps // 2,
            )

    @reactive.Calc
    def analysis_results_list():
        channel = input.channel_selection()
        results = []
        for file_data in loaded_abf_data():
            analysis_result = analysis.run_analysis_on_abf(
                abf=file_data.get("abf_object"),
                original_filename=file_data.get("original_filename"),
                user_selected_features=selected_efel_features(),
                channel_selection=channel,
                stimulus_epoch_index=input.stimulus_epoch_index(),
                detection_threshold=input.detection_threshold(),
                derivative_threshold=input.derivative_threshold(),
                current_col_name=constants.CURRENT_COL_NAME,
            )
            results.append(
                {
                    "channel_selection": channel,
                    **file_data,
                    "analysis_df": analysis_result.analysis_df,
                }
            )
        return results

    @reactive.Calc
    def combined_analysis_df() -> pd.DataFrame:
        """Concatenate valid per-file DataFrames, returning empty on failure."""
        results_list = analysis_results_list()
        valid_dfs = [
            r.get("analysis_df")
            for r in results_list
            if helper.is_valid_analysis_df(r.get("analysis_df"))
        ]

        if not valid_dfs:
            helper._log_message(
                "WARN", "App", None, "No valid analysis DataFrames to combine."
            )
            return pd.DataFrame()
        try:
            return pd.concat(valid_dfs, ignore_index=True, sort=False)
        except Exception as e:
            helper._log_message(
                "ERROR", "App", None, f"Failed to concatenate DataFrames: {e}"
            )
            traceback.print_exc()
            return pd.DataFrame()

    @output
    @render.ui
    def advanced_features_filtered_ui():
        search = input.feature_search().lower().strip()
        filtered = [
            f for f in ADVANCED_EFEL_FEATURES if search in f.lower()
        ]

        if not filtered:
            return ui.p("No features match your search.", style="color: gray;")

        try:
            current = set(input.advanced_efel_features() or [])
        except Exception:
            current = set()

        selected = [f for f in filtered if f in current]
        return ui.input_checkbox_group(
            "advanced_efel_features",
            label=f"Showing {len(filtered)} of {len(ADVANCED_EFEL_FEATURES)} features:",
            choices=filtered,
            selected=selected,
        )

    @output
    @render.text
    def analysis_summary_text():
        results = analysis_results_list()
        num_total = len(results)
        if num_total == 0:
            return (
                "1. Upload one or more ABF files.\n"
                "2. Adjust parameters if needed.\n"
                "3. View results in tabs."
            )

        num_load_ok = sum(
            1 for r in results if r.get("abf_object") and not r.get("load_error")
        )
        num_load_err = sum(1 for r in results if r.get("load_error"))
        num_analyzed_ok = sum(
            1 for r in results if helper.is_valid_analysis_df(r.get("analysis_df"))
        )
        num_analysis_failed = num_total - num_analyzed_ok - num_load_err

        lines = [
            "--- File Status ---",
            f"Total Files Attempted: {num_total}",
            f"Successfully Loaded: {num_load_ok}",
            f"Load Errors: {num_load_err}",
            f"Successfully Analyzed: {num_analyzed_ok}",
            f"Analysis Skipped/Failed: {num_analysis_failed}",
            "---",
            "--- Current Settings ---",
            f"Stimulus Epoch Index Used: {input.stimulus_epoch_index()}",
            f"Spike V Threshold: {input.detection_threshold()} mV",
            f"Spike dV/dt Threshold: {input.derivative_threshold()} mV/ms",
            f"# eFEL Features Selected: {len(selected_efel_features())}",
            "---",
        ]

        first_ok = next((r for r in results if r.get("abf_object")), None)
        if first_ok:
            lines.append(f"First File Info ({first_ok['original_filename']}):")
            lines.append(
                helper.get_abf_info_text(
                    first_ok["abf_object"],
                    first_ok["original_filename"],
                )
            )
            lines.append("---")

        first_analyzed = next(
            (r for r in results if helper.is_valid_analysis_df(r.get("analysis_df"))),
            None,
        )
        if first_analyzed:
            cols = ", ".join(first_analyzed["analysis_df"].columns)
            max_line = 70
            wrapped = "\n".join(
                cols[i : i + max_line] for i in range(0, len(cols), max_line)
            )
            lines.append(
                f"--- Output Columns ({first_analyzed['original_filename']}) ---"
            )
            lines.append(wrapped)
        elif num_load_ok > 0:
            lines.append("--- Output Columns ---\n(Waiting for successful analysis...)")
        else:
            lines.append("--- Output Columns ---\n(Waiting for files to load...)")

        return "\n".join(lines)

    @output
    @render.ui
    def dynamic_summary_plots_ui():
        results = analysis_results_list()
        req(results)

        ui_elements = []
        for i, result_data in enumerate(results):
            filename = result_data.get("original_filename", f"File {i + 1}")
            plot_fig = None
            try:
                plot_fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
                plot_fig.set_layout_engine("tight")

                plotting._generate_summary_plots_for_file(
                    result_data, axes, current_col=constants.CURRENT_COL_NAME
                )

                plot_src = helper.fig_to_src_and_close(plot_fig)
                plot_fig = None

                if plot_src:
                    ui_elements.append(
                        _file_plot_block(
                            i,
                            filename,
                            ui.img(
                                src=plot_src,
                                style=(
                                    "width: 100%; height: auto; "
                                    "max-width: 1400px; border: 1px solid #ddd;"
                                ),
                            ),
                        )
                    )
                else:
                    helper._log_message(
                        "WARN",
                        filename,
                        None,
                        "Figure conversion to src failed for UI summary plot.",
                    )
                    ui_elements.append(
                        _file_plot_block(
                            i,
                            filename,
                            ui.p(
                                f"Could not generate summary plot image for {filename}."
                            ),
                        )
                    )

            except Exception as e_ui_plot:
                helper._log_message(
                    "ERROR",
                    filename,
                    None,
                    f"Failed to generate UI summary plot figure for "
                    f"{filename}: {e_ui_plot}",
                )
                traceback.print_exc()
                ui_elements.append(
                    _file_plot_block(
                        i,
                        filename,
                        ui.p(
                            f"Error generating plots: {e_ui_plot}",
                            style="color: red;",
                        ),
                    )
                )
            finally:
                if plot_fig is not None:
                    plt.close(plot_fig)

        return ui.TagList(*ui_elements) if ui_elements else ui.p("No plots generated.")

    @output
    @render.ui
    def dynamic_debug_plots_ui():
        results = analysis_results_list()
        req(results)

        sweep_num = input.debug_sweep()
        ui_elements = []
        for result_data in results:
            filename = result_data["original_filename"]
            abf_obj = result_data.get("abf_object")
            load_err = result_data.get("load_error")
            channel = result_data.get("channel_selection", 0)

            if load_err or not isinstance(abf_obj, pyabf.ABF):
                ui_elements.append(
                    ui.div(
                        ui.h5(f"Debug Details: {filename}"),
                        ui.p(f"File not loaded: {load_err or 'unknown error'}"),
                        ui.hr(),
                    )
                )
                continue

            max_sweep = getattr(abf_obj, "sweepCount", 0)
            if max_sweep == 0:
                ui_elements.append(
                    ui.div(
                        ui.h5(f"Debug Details: {filename}"),
                        ui.p("No sweeps in file."),
                        ui.hr(),
                    )
                )
                continue

            effective_sweep = min(sweep_num, max_sweep - 1)
            fig = None
            try:
                fig = analysis.generate_debug_plot(
                    abf=abf_obj,
                    sweep_num=effective_sweep,
                    channel=channel,
                    stimulus_epoch_index=input.stimulus_epoch_index(),
                    detection_threshold=input.detection_threshold(),
                    derivative_threshold=input.derivative_threshold(),
                )
                if fig is not None:
                    plot_src = helper.fig_to_src_and_close(fig)
                    fig = None
                    if plot_src:
                        ui_elements.append(
                            ui.div(
                                ui.h5(
                                    f"Debug Details: {filename} "
                                    f"(Sweep {effective_sweep})"
                                ),
                                ui.row(
                                    ui.column(
                                        12,
                                        ui.img(
                                            src=plot_src,
                                            style="width: 100%; height: auto;",
                                        ),
                                    )
                                ),
                                ui.hr(),
                            )
                        )
                    else:
                        ui_elements.append(
                            ui.div(
                                ui.h5(f"Debug Details: {filename}"),
                                ui.p("Could not convert debug plot to image."),
                                ui.hr(),
                            )
                        )
                else:
                    ui_elements.append(
                        ui.div(
                            ui.h5(f"Debug Details: {filename}"),
                            ui.p("Debug plot generation returned no figure."),
                            ui.hr(),
                        )
                    )
            except Exception as e:
                helper._log_message(
                    "ERROR", filename, None,
                    f"Debug plot generation failed: {e}"
                )
                ui_elements.append(
                    ui.div(
                        ui.h5(f"Debug Details: {filename}"),
                        ui.p(f"Error: {e}", style="color: red;"),
                        ui.hr(),
                    )
                )
            finally:
                if fig is not None:
                    plt.close(fig)

        if not ui_elements:
            return ui.p("No files loaded.")
        return ui.div(*ui_elements, class_="debug-plots-grid")

    @output
    @render.data_frame
    def analysis_data_table():
        df = combined_analysis_df()
        if df.empty:
            return pd.DataFrame()
        return render.DataGrid(
            df.round(3),
            selection_mode="none",
            width="100%",
            height="600px",
        )

    @output
    @render.ui
    def download_buttons():
        df = combined_analysis_df()
        if not helper.is_valid_analysis_df(df):
            return ui.help_text("Upload and analyse files to enable downloads.")
        return ui.div(
            ui.tooltip(
                ui.download_button("download_analysis_csv", "Download Results (CSV)"),
                "Download all analysed data as a single CSV file.",
                placement="right",
            ),
            ui.tooltip(
                ui.download_button(
                    "download_analysis_excel", "Download Results (Excel)"
                ),
                "Download with one sheet per feature, pivoted by filename.",
                placement="right",
            ),
            ui.tooltip(
                ui.download_button(
                    "download_plots_pdf", "Download Summary Plots (PDF)"
                ),
                "Download summary plots (2 files per A4 page).",
                placement="right",
            ),
            style="display: flex; flex-direction: column; gap: 8px;",
        )

    @output
    @render.text
    def analysis_logs_text():
        entries = session_logger.get_entries()
        if not entries:
            return "No logs yet. Upload files to see analysis messages."
        lines = []
        for entry in entries:
            prefix = f"[{entry.level}] {entry.abf_id}"
            if entry.sweep_num is not None:
                prefix += f" (Sw {entry.sweep_num})"
            lines.append(f"{prefix}: {entry.message}")
        return "\n".join(lines)

    @reactive.Effect
    @reactive.event(input.clear_logs)
    def _clear_logs():
        session_logger.clear()

    register_download_handlers(
        input, output, loaded_abf_data, combined_analysis_df, analysis_results_list
    )
