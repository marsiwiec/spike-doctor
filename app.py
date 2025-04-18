import io
import traceback
import warnings
from pathlib import Path
from io import BytesIO
import shiny
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import pyabf
import efel
from shiny import App, render, ui, reactive, req

from modules import analysis, constants, helper, plotting


try:
    AVAILABLE_EFEL_FEATURES = sorted(efel.get_feature_names())
except Exception as e:
    print(f"Warning: Could not dynamically get eFEL features: {e}. Using defaults.")
    AVAILABLE_EFEL_FEATURES = sorted(
        constants.DEFAULT_EFEL_FEATURES + constants.REQUIRED_INTERNAL_EFEL_FEATURES
    )

VALID_DEFAULT_FEATURES = [
    f for f in constants.DEFAULT_EFEL_FEATURES if f in AVAILABLE_EFEL_FEATURES
]

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



def server(input: shiny.Inputs, output: shiny.Outputs, session: shiny.Session):

    loaded_abf_data = reactive.Value([])

    @reactive.Effect
    @reactive.event(input.abf_files)
    def _load_abf_files():
        """Loads ABF files selected by the user."""
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
                    # Ensure warnings during loading are not treated as errors here
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        abf_obj = pyabf.ABF(str(filepath), loadData=True)
                except FileNotFoundError:
                    error_msg = "File not found at temporary path."
                    helper._log_message(
                        "ERROR", filename, None, f"{error_msg} Path: {filepath}"
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
        helper._log_message("INFO", "App", None, f"Finished loading {len(data_list)} files.")

    analysis_results_list = reactive.Calc(
        lambda: [
            {
                **file_data,
                **analysis.run_analysis_on_abf(
                    abf=file_data.get("abf_object"),
                    original_filename=file_data.get("original_filename"),
                    user_selected_features=input.selected_efel_features(),
                    stimulus_epoch_index=input.stimulus_epoch_index(),
                    detection_threshold=input.detection_threshold(),
                    derivative_threshold=input.derivative_threshold(),
                    debug_plot=input.debug_plots(),
                    current_col_name=constants.CURRENT_COL_NAME,
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
        results_list = analysis_results_list()  
        helper._log_message(
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
            helper._log_message(
                "WARN", "App", None, "No valid analysis DataFrames to combine."
            )
            return pd.DataFrame()
        try:
            combined_df = pd.concat(valid_dfs, ignore_index=True, sort=False)
            helper._log_message(
                "DEBUG", "App", None, f"Combined DataFrame shape: {combined_df.shape}"
            )
            return combined_df
        except Exception as e:
            helper._log_message("ERROR", "App", None, f"Failed to concatenate DataFrames: {e}")
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
            f"Stimulus Epoch Index Used: {input.stimulus_epoch_index()}\n"  
            f"Spike V Threshold: {input.detection_threshold()} mV\n"
            f"Spike dV/dt Threshold: {input.derivative_threshold()} mV/ms\n"
            f"Debug Plots Enabled: {'Yes' if input.debug_plots() else 'No'}\n"
            f"# eFEL Features Selected: {len(input.selected_efel_features())}\n"
            f"---\n"
        )

        first_ok_file_data = next((r for r in results if r.get("abf_object")), None)
        if first_ok_file_data:
            summary += f"First File Info ({first_ok_file_data['original_filename']}):\n"
            summary += helper.get_abf_info_text(
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

        helper._log_message(
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

                plotting._generate_summary_plots_for_file(
                    result_data, axes, current_col=constants.CURRENT_COL_NAME
                )

                plot_src = helper.fig_to_src(plot_fig)

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
                    helper._log_message(
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
                helper._log_message(
                    "ERROR",
                    filename,
                    None,
                    f"Failed to generate UI summary plot figure for {filename}: {e_ui_plot}",
                )
                traceback.print_exc()
                if plot_fig:  
                    plt.close(plot_fig)
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
                debug_plot_src = helper.fig_to_src(debug_fig)
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
                    helper._log_message(
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
        if not plots_found and results:
            return ui.tags.div(
                ui.help_text(
                    "Debug plots are enabled, but none were generated. This might happen if analysis failed early for all files, or if the middle sweep processing encountered an error."
                )
            )
        elif not results:  
            return ui.tags.div(
                ui.help_text("Load ABF files to generate debug plots (if enabled).")
            )
        else:  
            return ui.TagList(*ui_elements)

    @output
    @render.data_frame
    def analysis_data_table():
        """Renders the combined analysis DataFrame."""
        df = combined_analysis_df()
        if df.empty:
            helper._log_message("DEBUG", "UI", None, "Rendering empty DataFrame.")
            return pd.DataFrame()
        return render.DataGrid(
            df.round(3), selection_mode="none", width="100%", height="600px"
        )

        helper._log_message("DEBUG", "UI", None, f"Rendering DataFrame with shape {df.shape}")
        return render.DataGrid(
            df.round(4),  
            row_selection_mode="none",
            width="100%",
            height="600px",
            filters=True,  
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
        )  
        helper._log_message(
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
        Within each sheet, rows are indexed by sweep and constants.CURRENT_COL_NAME,
        and columns correspond to unique filenames. Yields the binary
        content of the Excel file.
        """
        df_before_pivot = combined_analysis_df()
        req(
            df_before_pivot is not None and not df_before_pivot.empty,
            cancel_output=ValueError("No analysis data available to download."),
        )
        helper._log_message(
            "INFO",
            "Download",
            None,
            f"Generating Excel download for {df_before_pivot.shape[0]} rows.",
        )

        # --- Prepare the Base DataFrame with MultiIndex ---
        index_cols = ["filename", "sweep", constants.CURRENT_COL_NAME]
        if not all(col in df_before_pivot.columns for col in index_cols):
            missing_cols = [
                col for col in index_cols if col not in df_before_pivot.columns
            ]
            err_msg = f"Error: One or more index columns ({missing_cols}) not found in DataFrame for Excel export."
            helper._log_message("ERROR", "Download", None, err_msg)
            raise ValueError(err_msg)
        dependent_vars = [
            col for col in df_before_pivot.columns if col not in index_cols
        ]
        if not dependent_vars:
            err_msg = "Error: No dependent variable columns found for Excel export."
            helper._log_message("ERROR", "Download", None, err_msg)
            raise ValueError(err_msg)

        output_buffer = io.BytesIO()

        try:

            # Use ExcelWriter to manage writing multiple sheets to the buffer
            with pd.ExcelWriter(output_buffer, engine="openpyxl") as writer:
                for var_name in dependent_vars:
                    helper._log_message(
                        "DEBUG", "Download", None, f"Processing sheet for: {var_name}"
                    )
                    df_subset = df_before_pivot[index_cols + [var_name]].copy()

                    if df_subset.duplicated(subset=index_cols).any():
                        num_duplicates = df_subset.duplicated(subset=index_cols).sum()
                        helper._log_message(
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
                            index=["sweep", constants.CURRENT_COL_NAME],
                            columns="filename",
                            values=var_name,
                        )

                    except Exception as e_pivot:
                        helper._log_message(
                            "ERROR",
                            "Download",
                            None,
                            f"Error pivoting data for variable '{var_name}': {e_pivot}",
                        )
                        raise RuntimeError(
                            f"Failed to pivot data for {var_name}"
                        ) from e_pivot

                    # Clean sheet name (max 31 chars, no invalid chars)
                    clean_sheet_name = (
                        str(var_name).replace("_", " ").title()
                    ) 
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
            output_buffer.seek(0)
            excel_data = output_buffer.getvalue()
            helper._log_message("INFO", "Download", None, "Excel file generated successfully.")
            yield excel_data

        except Exception as e_excel:
            helper._log_message(
                "ERROR",
                "Download",
                None,
                f"Critical error during Excel generation: {e_excel}",
            )
            traceback.print_exc()
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
        helper._log_message(
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
                        fig_pdf_page = None  
                        try:
                            # Create a figure with a 2x3 grid for two files
                            fig_pdf_page, axes = plt.subplots(
                                2,  
                                3,  
                                figsize=(A4_LANDSCAPE_WIDTH_IN, A4_LANDSCAPE_HEIGHT_IN),
                                squeeze=False,  
                            )
                            fig_pdf_page.set_layout_engine(
                                "tight", pad=1.5
                            )  

                            # --- Process First File (Top Row) ---
                            result_data_1 = results[i]
                            filename_1 = result_data_1["original_filename"]
                            helper._log_message(
                                "DEBUG",
                                "PDF Export",
                                None,
                                f"Processing File {i+1} for PDF (Top Row): {filename_1}",
                            )
                            
                            plotting._generate_summary_plots_for_file(
                                result_data_1,
                                axes=axes[0, :],  # Pass the first row of axes
                                current_col=constants.CURRENT_COL_NAME,
                            )

                            # --- Process Second File (Bottom Row) if it exists ---
                            if i + 1 < num_files:
                                result_data_2 = results[i + 1]
                                filename_2 = result_data_2["original_filename"]
                                helper._log_message(
                                    "DEBUG",
                                    "PDF Export",
                                    None,
                                    f"Processing File {i+2} for PDF (Bottom Row): {filename_2}",
                                )
                                plotting._generate_summary_plots_for_file(
                                    result_data_2,
                                    axes=axes[1, :],  # Pass the second row of axes
                                    current_col=constants.CURRENT_COL_NAME,
                                )
                            else:
                                helper._log_message(
                                    "DEBUG",
                                    "PDF Export",
                                    None,
                                    f"Odd number of files. Page {pdf.get_pagecount()+1} has only one file.",
                                )
                                for ax_empty in axes[1, :]:
                                    ax_empty.axis("off")

                            pdf.savefig(fig_pdf_page)

                        except Exception as e_page:
                            helper._log_message(
                                "ERROR",
                                "PDF Export",
                                None,
                                f"Failed to create PDF page starting with file {i+1} ({results[i]['original_filename']}): {e_page}",
                            )
                            traceback.print_exc()

                        finally:
                            if fig_pdf_page is not None:
                                plt.close(fig_pdf_page)

                    pdf_page_count = pdf.get_pagecount()

                # After PdfPages context closes, get the buffer's content
                pdf_content = pdf_buffer.getvalue()

        except Exception as outer_e:
            helper._log_message(
                "ERROR",
                "PDF Export",
                None,
                f"Critical error during PDF setup/generation: {outer_e}",
            )
            traceback.print_exc()
            yield f"Error: Failed to generate PDF. Check logs. ({outer_e})".encode(
                "utf-8"
            )
            return

        helper._log_message(
            "INFO",
            "PDF Export",
            None,
            f"PDF generation complete ({pdf_page_count} pages).",
        )

        if pdf_content:
            yield pdf_content
        else:
            helper._log_message(
                "WARN",
                "PDF Export",
                None,
                "PDF content was empty after generation attempt.",
            )
            yield "Error: Generated PDF was empty.".encode("utf-8")


# ==============================================================================
# Shiny App Start
# ==============================================================================
app = App(app_ui, server)
