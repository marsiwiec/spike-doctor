import io
import traceback
import warnings
from pathlib import Path

import efel
import matplotlib.pyplot as plt
import pandas as pd
import pyabf
from matplotlib.backends.backend_pdf import PdfPages
from shiny import App, reactive, render, req, ui

from modules import analysis, constants, helper, plotting

try:
    AVAILABLE_EFEL_FEATURES = sorted(efel.get_feature_names())
except Exception as e:
    print(f"Warning: Could not dynamically get eFEL features: {e}. Using defaults.")
    AVAILABLE_EFEL_FEATURES = sorted(
        constants.DEFAULT_EFEL_FEATURES + constants.REQUIRED_INTERNAL_EFEL_FEATURES
    )

ADVANCED_EFEL_FEATURES = sorted([
    f for f in AVAILABLE_EFEL_FEATURES if f not in constants.BASIC_EFEL_FEATURES
])


def _create_basic_feature_checkboxes():
    elements = []
    for feature_id, (display_name, description) in (
        constants.BASIC_EFEL_FEATURES.items()
    ):
        if feature_id in AVAILABLE_EFEL_FEATURES:
            elements.append(
                ui.tooltip(
                    ui.input_checkbox(
                        f"basic_feature_{feature_id}",
                        display_name,
                        value=True,
                    ),
                    description,
                    placement="right",
                )
            )
    return elements


app_ui = ui.page_fluid(
    ui.tags.head(ui.tags.title("Spike Doctor")),
    ui.layout_sidebar(
        ui.sidebar(
            ui.h4("Spike Doctor"),
            ui.h5("Analyze current clamp ABF files"),
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
                "channel_selection",
                "Channel (0-based):",
                value=0,
                min=0,
                step=1,
            ),
            ui.input_numeric(
                "stimulus_epoch_index",
                "Stimulus Epoch Index (0-based):",
                value=2,
                min=0,
                step=1,
            ),
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
            ui.tags.b("Options:"),
            ui.input_checkbox(
                "debug_plots",
                "Generate Debug Plots",
                True,
            ),
            ui.h5("eFEL Features to Calculate:"),
            ui.navset_pill(
                ui.nav_panel(
                    "Basic",
                    ui.div(
                        *_create_basic_feature_checkboxes(),
                        style="margin-top: 10px;",
                    ),
                ),
                ui.nav_panel(
                    "Advanced",
                    ui.div(
                        ui.input_checkbox_group(
                            "advanced_efel_features",
                            label="Additional eFEL features:",
                            choices=ADVANCED_EFEL_FEATURES,
                            selected=[],
                        ),
                        style="margin-top: 10px; max-height: 300px; overflow-y: auto;",
                    ),
                ),
                id="feature_tabs",
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
                    "Generated if 'Generate Debug Plots' is enabled. "
                    "Shows details of analysis steps for one sweep per file."
                ),
                ui.output_ui("dynamic_debug_plots_ui"),
            ),
        ),
    ),
)


def _build_excel_bytes(df_before_pivot: pd.DataFrame) -> bytes:
    """Pivot each dependent variable into its own sheet."""
    index_cols = ["filename", "sweep", constants.CURRENT_COL_NAME]
    missing_cols = [c for c in index_cols if c not in df_before_pivot.columns]
    if missing_cols:
        raise ValueError(f"Index columns missing for Excel export: {missing_cols}")

    dependent_vars = [
        c for c in df_before_pivot.columns
        if c not in index_cols + ["event_index"]
    ]
    if not dependent_vars:
        raise ValueError("No dependent variable columns found for Excel export.")

    output_buffer = io.BytesIO()
    with pd.ExcelWriter(output_buffer, engine="openpyxl") as writer:
        for var_name in dependent_vars:
            df_subset = df_before_pivot[index_cols + [var_name]].copy()
            if df_subset.duplicated(subset=index_cols).any():
                num_dups = df_subset.duplicated(subset=index_cols).sum()
                helper._log_message(
                    "WARN",
                    "Download",
                    None,
                    f"Found {num_dups} duplicate index entries for var "
                    f"'{var_name}'. Keeping first occurrence.",
                )
                df_subset = df_subset.drop_duplicates(subset=index_cols, keep="first")

            pivot_idx = [c for c in index_cols if c != "filename"]
            df_pivot = df_subset.pivot_table(
                index=pivot_idx, columns="filename", values=var_name
            )
            df_pivot.to_excel(
                writer,
                sheet_name=helper.clean_excel_sheet_name(var_name),
                index=True,
                float_format="%.6g",
                na_rep="NaN",
            )

    return output_buffer.getvalue()


def _build_pdf_bytes(results: list) -> bytes:
    """Render summary plots 2 files per A4 landscape page."""
    num_files = len(results)
    helper._log_message(
        "INFO", "PDF Export", None,
        f"Generating PDF for {num_files} files (2 per page).",
    )

    A4_W, A4_H = 11.69, 8.27
    pdf_buffer = io.BytesIO()

    with PdfPages(pdf_buffer) as pdf:
        for i in range(0, num_files, 2):
            fig = None
            try:
                fig, axes = plt.subplots(2, 3, figsize=(A4_W, A4_H), squeeze=False)
                fig.set_layout_engine("tight", pad=1.5)

                plotting._generate_summary_plots_for_file(
                    results[i],
                    axes=list(axes[0, :]),
                    current_col=constants.CURRENT_COL_NAME,
                )
                if i + 1 < num_files:
                    plotting._generate_summary_plots_for_file(
                        results[i + 1],
                        axes=list(axes[1, :]),
                        current_col=constants.CURRENT_COL_NAME,
                    )
                else:
                    for ax_empty in axes[1, :]:
                        ax_empty.axis("off")

                pdf.savefig(fig)
            except Exception as e_page:
                helper._log_message(
                    "ERROR",
                    "PDF Export",
                    None,
                    f"Failed PDF page {i+1} "
                    f"({results[i]['original_filename']}): {e_page}",
                )
                traceback.print_exc()
            finally:
                if fig is not None:
                    plt.close(fig)

        page_count = pdf.get_pagecount()

    pdf_content = pdf_buffer.getvalue()
    pdf_buffer.close()

    helper._log_message(
        "INFO", "PDF Export", None,
        f"PDF complete ({page_count} pages).",
    )
    if not pdf_content:
        raise RuntimeError("Generated PDF was empty.")
    return pdf_content


def _file_plot_block(i: int, filename: str, content):
    return ui.div(
        ui.hr() if i > 0 else None,
        ui.h5(filename),
        ui.row(ui.column(12, content)),
    )


def server(input, output, session):
    loaded_abf_data = reactive.Value([])

    @reactive.Calc
    def selected_efel_features() -> list:
        features = [
            f for f in constants.BASIC_EFEL_FEATURES
            if f in AVAILABLE_EFEL_FEATURES and getattr(input, f"basic_feature_{f}")()
        ]
        features.extend(input.advanced_efel_features() or [])
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
                        "ERROR", filename, None,
                        f"{error_msg} Path: {filepath}",
                    )
                except Exception as e:
                    error_msg = f"Failed to load: {e}"
                    helper._log_message(
                        "ERROR", filename, None,
                        f"{error_msg}\n{traceback.format_exc()}",
                    )

                data_list.append({
                    "original_filename": filename,
                    "filepath": str(filepath),
                    "abf_object": abf_obj,
                    "load_error": error_msg,
                })
            p.set(num_files, detail="Loading complete.")

        loaded_abf_data.set(data_list)
        helper._log_message(
            "INFO", "App", None,
            f"Finished loading {len(data_list)} files.",
        )

    @reactive.Calc
    def analysis_results_list():
        channel = input.channel_selection()
        return [
            {
                "channel_selection": channel,
                **file_data,
                **analysis.run_analysis_on_abf(
                    abf=file_data.get("abf_object"),
                    original_filename=file_data.get("original_filename"),
                    user_selected_features=selected_efel_features(),
                    channel_selection=channel,
                    stimulus_epoch_index=input.stimulus_epoch_index(),
                    detection_threshold=input.detection_threshold(),
                    derivative_threshold=input.derivative_threshold(),
                    debug_plot=input.debug_plots(),
                    current_col_name=constants.CURRENT_COL_NAME,
                ),
            }
            for file_data in loaded_abf_data()
        ]

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
                "WARN", "App", None,
                "No valid analysis DataFrames to combine.",
            )
            return pd.DataFrame()
        try:
            return pd.concat(valid_dfs, ignore_index=True, sort=False)
        except Exception as e:
            helper._log_message(
                "ERROR", "App", None,
                f"Failed to concatenate DataFrames: {e}",
            )
            traceback.print_exc()
            return pd.DataFrame()

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
            1 for r in results
            if r.get("abf_object") and not r.get("load_error")
        )
        num_load_err = sum(1 for r in results if r.get("load_error"))
        num_analyzed_ok = sum(
            1 for r in results
            if helper.is_valid_analysis_df(r.get("analysis_df"))
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
            f"Debug Plots Enabled: {'Yes' if input.debug_plots() else 'No'}",
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
            (
                r for r in results
                if helper.is_valid_analysis_df(r.get("analysis_df"))
            ),
            None,
        )
        if first_analyzed:
            cols = ", ".join(first_analyzed["analysis_df"].columns)
            max_line = 70
            wrapped = "\n".join(
                cols[i : i + max_line]
                for i in range(0, len(cols), max_line)
            )
            lines.append(
                f"--- Output Columns "
                f"({first_analyzed['original_filename']}) ---"
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
            filename = result_data.get("original_filename", f"File {i+1}")
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
                    ui_elements.append(_file_plot_block(
                        i, filename,
                        ui.img(
                            src=plot_src,
                            style=(
                                "width: 100%; height: auto; "
                                "max-width: 1400px; border: 1px solid #ddd;"
                            ),
                        ),
                    ))
                else:
                    helper._log_message(
                        "WARN", filename, None,
                        "Figure conversion to src failed for UI summary plot.",
                    )
                    ui_elements.append(_file_plot_block(
                        i, filename,
                        ui.p(f"Could not generate summary plot image for {filename}."),
                    ))

            except Exception as e_ui_plot:
                helper._log_message(
                    "ERROR", filename, None,
                    f"Failed to generate UI summary plot figure for "
                    f"{filename}: {e_ui_plot}",
                )
                traceback.print_exc()
                ui_elements.append(_file_plot_block(
                    i, filename,
                    ui.p(f"Error generating plots: {e_ui_plot}", style="color: red;"),
                ))
            finally:
                if plot_fig is not None:
                    plt.close(plot_fig)

        return ui.TagList(*ui_elements) if ui_elements else ui.p("No plots generated.")

    @output
    @render.ui
    def dynamic_debug_plots_ui():
        if not input.debug_plots():
            return ui.tags.p("Debug plots are disabled in the configuration.")

        results = analysis_results_list()
        req(results)

        ui_elements = []
        plots_found = False
        for result_data in results:
            filename = result_data["original_filename"]
            debug_fig = result_data.get("debug_plot_fig")

            if isinstance(debug_fig, plt.Figure):
                plots_found = True
                debug_plot_src = helper.fig_to_src_and_close(debug_fig)
                if debug_plot_src:
                    ui_elements.append(ui.div(
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
                    ))
                else:
                    helper._log_message(
                        "WARN", filename, None,
                        "Figure conversion to src failed for UI debug plot.",
                    )
                    ui_elements.append(ui.div(
                        ui.h5(f"Debug Details: {filename}"),
                        ui.p(f"Could not generate debug plot image for {filename}."),
                        ui.hr(),
                    ))
        if not plots_found:
            return ui.help_text(
                "Debug plots are enabled, but none were generated. "
                "This might happen if analysis failed early for all files, "
                "or if the middle sweep processing encountered an error."
            )
        return ui.TagList(*ui_elements)

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

    @render.download(
        filename=lambda: (
            f"ABF_analysis_{len(loaded_abf_data())}files_"
            f"{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
    )
    def download_analysis_csv():
        df_to_download = combined_analysis_df()
        req(df_to_download is not None and not df_to_download.empty)
        helper._log_message(
            "INFO", "Download", None,
            f"Generating CSV download for {df_to_download.shape[0]} rows.",
        )
        with io.StringIO() as buf:
            df_to_download.to_csv(buf, index=False, float_format="%.6g")
            yield buf.getvalue()

    @render.download(
        filename=lambda: (
            f"ABF_analysis_{len(loaded_abf_data())}files_"
            f"{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        )
    )
    def download_analysis_excel():
        df = combined_analysis_df()
        req(df is not None and not df.empty)
        helper._log_message(
            "INFO", "Download", None,
            f"Generating Excel download for {df.shape[0]} rows.",
        )
        try:
            yield _build_excel_bytes(df)
        except Exception as e:
            helper._log_message(
                "ERROR", "Download", None,
                f"Critical error during Excel generation: {e}",
            )
            traceback.print_exc()
            yield f"Error generating Excel file: {e}".encode()

    @render.download(
        filename=lambda: (
            f"ABF_Summary_Plots_{len(loaded_abf_data())}files_"
            f"{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        )
    )
    def download_plots_pdf():
        results = analysis_results_list()
        req(results)
        try:
            yield _build_pdf_bytes(results)
        except Exception as e:
            helper._log_message(
                "ERROR", "PDF Export", None,
                f"Critical error during PDF generation: {e}",
            )
            traceback.print_exc()
            yield f"Error: Failed to generate PDF. Check logs. ({e})".encode()


app = App(app_ui, server)
