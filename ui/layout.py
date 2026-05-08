from shiny import ui

from modules import constants

try:
    import efel

    AVAILABLE_EFEL_FEATURES = sorted(efel.get_feature_names())
except Exception as e:
    print(f"Warning: Could not dynamically get eFEL features: {e}. Using defaults.")
    AVAILABLE_EFEL_FEATURES = sorted(
        constants.DEFAULT_EFEL_FEATURES + constants.REQUIRED_INTERNAL_EFEL_FEATURES
    )

ADVANCED_EFEL_FEATURES = sorted(
    [f for f in AVAILABLE_EFEL_FEATURES if f not in constants.BASIC_EFEL_FEATURES]
)


def _create_basic_feature_checkboxes():
    elements = []
    for feature_id, (
        display_name,
        description,
    ) in constants.BASIC_EFEL_FEATURES.items():
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


def create_app_ui():
    return ui.page_fluid(
        ui.tags.head(
            ui.tags.title("Spike Doctor"),
            ui.tags.style(
                """
                .debug-plots-grid {
                    display: grid;
                    gap: 1rem;
                }
                @media (min-width: 1200px) {
                    .debug-plots-grid {
                        grid-template-columns: repeat(2, 1fr);
                    }
                }
                """
            ),
        ),
        ui.layout_sidebar(
            ui.sidebar(
                ui.h4("Spike Doctor"),
                ui.h5("Analyze current clamp ABF files"),
                ui.input_file(
                    "abf_files", "Select ABF File(s):",
                    accept=[".abf"], multiple=True
                ),
                ui.help_text(
                    "Supports ABF v2 files. ABF v1 float data is unsupported "
                    "and must be re-saved in Clampfit."
                ),
                ui.hr(),
                ui.output_ui("download_buttons"),
                ui.hr(),
                ui.h5("Analysis Parameters"),
                ui.tags.b("Stimulus Definition:"),
                ui.tooltip(
                    ui.input_numeric(
                        "channel_selection",
                        "Channel (0-based):",
                        value=0,
                        min=0,
                        step=1,
                    ),
                    "ADC channel to analyse. Usually 0 for single-channel recordings.",
                    placement="right",
                ),
                ui.tooltip(
                    ui.input_numeric(
                        "stimulus_epoch_index",
                        "Stimulus Epoch Index (0-based):",
                        value=2,
                        min=0,
                        step=1,
                    ),
                    "Index of the stimulus epoch in the protocol. "
                    "Epoch 0 is typically holding; epoch 2 is often the step.",
                    placement="right",
                ),
                ui.tags.b("Spike Detection:"),
                ui.tooltip(
                    ui.input_numeric(
                        "detection_threshold",
                        "Detection Threshold (mV):",
                        value=-20,
                        step=1,
                    ),
                    "Voltage threshold for spike detection. "
                    "More negative values detect smaller spikes.",
                    placement="right",
                ),
                ui.tooltip(
                    ui.input_numeric(
                        "derivative_threshold",
                        "Derivative Threshold (mV/ms):",
                        value=10,
                        step=1,
                    ),
                    "Minimum dV/dt required to count as a spike. "
                    "Higher values reduce noise false-positives.",
                    placement="right",
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
                            ui.input_text(
                                "feature_search",
                                "Filter features",
                                placeholder="Type to search...",
                            ),
                            ui.output_ui("advanced_features_filtered_ui"),
                            style=(
                                "margin-top: 10px; max-height: 400px; overflow-y: auto;"
                            ),
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
                    ui.h4("Debug Plots"),
                    ui.help_text(
                        "Voltage trace, detected spikes, threshold, and "
                        "command current for a chosen sweep."
                    ),
                    ui.input_slider(
                        "debug_sweep",
                        "Debug Sweep",
                        min=0,
                        max=0,
                        value=0,
                        step=1,
                    ),
                    ui.output_ui("dynamic_debug_plots_ui"),
                ),
                ui.nav_panel(
                    "Analysis Logs",
                    ui.h4("Analysis Logs"),
                    ui.input_action_button("clear_logs", "Clear Logs"),
                    ui.output_text_verbatim("analysis_logs_text"),
                ),
            ),
        ),
    )
