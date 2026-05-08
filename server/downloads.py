import io
import traceback

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from shiny import render

from modules import constants, helper, plotting


def _build_excel_bytes(df_before_pivot: pd.DataFrame) -> bytes:
    """Pivot each dependent variable into its own sheet."""
    index_cols = ["filename", "sweep", constants.CURRENT_COL_NAME]
    missing_cols = [c for c in index_cols if c not in df_before_pivot.columns]
    if missing_cols:
        raise ValueError(
            f"Index columns missing for Excel export: {missing_cols}"
        )

    dependent_vars = [
        c
        for c in df_before_pivot.columns
        if c not in index_cols + ["event_index"]
    ]
    if not dependent_vars:
        raise ValueError(
            "No dependent variable columns found for Excel export."
        )

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
                df_subset = df_subset.drop_duplicates(
                    subset=index_cols, keep="first"
                )

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
        f"Generating PDF for {num_files} files (2 per page)."
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
                    f"Failed PDF page {i + 1} "
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
        "INFO", "PDF Export", None, f"PDF complete ({page_count} pages)."
    )
    if not pdf_content:
        raise RuntimeError("Generated PDF was empty.")
    return pdf_content


def register_download_handlers(
    input, output, loaded_abf_data, combined_analysis_df, analysis_results_list
):
    @render.download(
        filename=lambda: (
            f"ABF_analysis_{len(loaded_abf_data())}files_"
            f"{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
    )
    def download_analysis_csv():
        df_to_download = combined_analysis_df()
        if df_to_download is None or df_to_download.empty:
            return
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
        filename=lambda: (
            f"ABF_analysis_{len(loaded_abf_data())}files_"
            f"{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        )
    )
    def download_analysis_excel():
        df = combined_analysis_df()
        if df is None or df.empty:
            return
        helper._log_message(
            "INFO", "Download", None,
            f"Generating Excel download for {df.shape[0]} rows."
        )
        try:
            yield _build_excel_bytes(df)
        except Exception as e:
            helper._log_message(
                "ERROR", "Download", None,
                f"Critical error during Excel generation: {e}"
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
        if not results:
            return
        try:
            yield _build_pdf_bytes(results)
        except Exception as e:
            helper._log_message(
                "ERROR", "PDF Export", None,
                f"Critical error during PDF generation: {e}"
            )
            traceback.print_exc()
            yield f"Error: Failed to generate PDF. Check logs. ({e})".encode()
