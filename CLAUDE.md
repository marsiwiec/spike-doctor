# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Spike-doctor is a Python Shiny web application for analyzing whole-cell current clamp electrophysiology recordings. It focuses on spike detection and feature extraction from Axon Instruments ABF files using pyABF for file reading and eFEL for electrophysiological feature extraction.

## Development Commands

```bash
# Start the application
uv run shiny run app.py

# Access the app at http://127.0.0.1:8000

# Type checking
ty check

# Linting
ruff check .

# Code formatting
ruff format .
```

## Environment Setup

The project uses a [flake-parts](https://flake.parts/)-based Nix flake for reproducible development environments. The flake provides `uv`, Python 3.12, `ruff`, and `ty`. With `nix-direnv` installed, the environment auto-activates via `.envrc` after running `direnv allow`. Python dependencies are managed by `uv` and pinned in `uv.lock`.

## Architecture

### Main Components

- **app.py**: Shiny application entry point containing UI definition (`app_ui`) and server logic. Handles file uploads, parameter inputs, reactive analysis pipeline, and three download handlers (CSV, Excel, PDF).

- **modules/analysis.py**: Core analysis logic via `run_analysis_on_abf()`. Processes ABF files sweep-by-sweep, configures eFEL settings, extracts features, and calculates derived metrics (capacitance from time constant and input resistance).

- **modules/plotting.py**: All visualization functions. `_generate_summary_plots_for_file()` creates the standard 3-panel view (raw traces, spike count vs current, phase plane). `_prepare_phase_plot_data()` finds sweeps near 2x rheobase for phase plots.

- **modules/helper.py**: Utilities including `_log_message()` for standardized logging, `fig_to_src()` for matplotlib-to-base64 conversion, `parse_efel_value()` for safe eFEL result extraction, and ABF validation functions.

- **modules/constants.py**: Configuration values including `DEFAULT_EFEL_FEATURES`, `REQUIRED_INTERNAL_EFEL_FEATURES`, and `CURRENT_COL_NAME`.

### Data Flow

1. User uploads ABF files via Shiny file input
2. `_load_abf_files()` reactive effect loads files with pyabf
3. `analysis_results_list` reactive calc runs `run_analysis_on_abf()` for each file
4. Results flow to UI renderers (plots, data table) and download handlers
5. Analysis DataFrame contains per-sweep results with filename, sweep number, current step, and eFEL features

### Key Dependencies

- **pyabf**: Reads Axon Binary Format files
- **efel**: eFEL (Electrophys Feature Extraction Library) for spike detection and feature calculation
- **shiny**: Python Shiny for reactive web UI
- **matplotlib**: Plotting and PDF generation
- **pandas**: Data manipulation and Excel/CSV export
- **openpyxl**: Excel file writing with per-feature sheets

### eFEL Integration

eFEL requires traces as dictionaries with `T` (time in ms), `V` (voltage), `stim_start`, `stim_end`, and `stimulus_current`. Settings controlled via UI include spike detection threshold and derivative threshold. Features like `spike_count`, `voltage_base`, `time_constant`, and `ohmic_input_resistance` are required internally even if not user-selected.

### Feature Selection UI

The feature selection uses a tabbed interface (Basic/Advanced):
- **Basic tab**: Individual checkboxes with tooltips, defined in `constants.BASIC_EFEL_FEATURES` as `{feature_id: (display_name, description)}`
- **Advanced tab**: Checkbox group with all other eFEL features
- Server combines selections via `selected_efel_features()` reactive calculation
