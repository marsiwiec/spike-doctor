# Spike Doctor

A [Shiny](https://shiny.posit.co/py/) web application written in Python for analyzing whole-cell current clamp recordings. Spike Doctor focuses on action potential detection and extraction of subthreshold and suprathreshold features from Axon Instruments ABF files using [pyABF](https://github.com/swharden/pyABF) and the [eFEL](https://github.com/BlueBrain/eFEL) feature extraction library.

<img src="https://github.com/marsiwiec/spike-doctor/blob/main/assets/spike-doctor.png?raw=true" width="80%">

## Features

- **Batch Processing**: Analyze multiple ABF files simultaneously
- **Automatic Spike Detection**: Configurable voltage and derivative thresholds
- **Comprehensive Feature Extraction**: 200+ electrophysiological features via eFEL
- **Interactive Visualizations**: Raw traces, spike count vs current (F-I curves), and phase-plane plots
- **Multiple Export Formats**: CSV, Excel (with per-feature sheets), and PDF summary plots
- **Debug Mode**: Detailed plots showing stimulus windows and analysis parameters

## Installation

Choose the method that best fits your platform and workflow.

### Docker (Recommended — Cross-Platform)

The easiest way to run Spike Doctor on any system with Docker installed (Linux, macOS, or Windows):

```bash
git clone https://github.com/marsiwiec/spike-doctor.git
cd spike-doctor
docker compose up
```

Then open http://localhost:8000 in your web browser.

Alternatively, build and run manually:

```bash
docker build -t spike-doctor .
docker run -p 8000:8000 spike-doctor
```

### PyInstaller — Windows Executable

A standalone Windows executable is available for users who prefer not to install Docker or Python.

1. Download `SpikeDoctor.exe` from the [Releases](https://github.com/marsiwiec/spike-doctor/releases) page.
2. Double-click `SpikeDoctor.exe`.
3. Your default browser will open automatically at http://127.0.0.1:8000.
4. Keep the console window open while using the application.

> **Note:** The executable must be built on Windows. To build from source:
> ```bash
> uv pip install pyinstaller
> pyinstaller SpikeDoctor.spec
> ```
> The output will be in `dist/SpikeDoctor/`.

### Manual Installation (uv)

If you prefer a native Python environment, the project uses [uv](https://docs.astral.sh/uv/) for dependency management:

```bash
git clone https://github.com/marsiwiec/spike-doctor.git
cd spike-doctor
uv sync
```

Then start the application:

```bash
uv run shiny run app.py
```

Open http://127.0.0.1:8000 in your web browser.

### Using Nix + uv (Alternative)

For [Nix](https://nixos.org/) users, a [flake-parts](https://flake.parts/) configuration is included. It provides `uv`, Python 3.12, `ruff`, and `ty` in a reproducible shell:

```bash
git clone https://github.com/marsiwiec/spike-doctor.git
cd spike-doctor
# If you have nix-direnv installed, the environment activates automatically after you run `direnv allow`
nix develop  # Or enter the dev shell manually
uv sync
uv run shiny run app.py
```

### Quick Start

1. **Upload Files**: Click "Select ABF File(s)" and choose one or more `.abf` files
2. **Configure Parameters**: Adjust stimulus epoch index and spike detection thresholds if needed
3. **Select Features**: Use the Basic tab for common features or Advanced tab for the full eFEL feature set
4. **View Results**: Browse Summary Plots, Detailed Results Table, and Debug Plots tabs
5. **Export**: Download results as CSV, Excel, or PDF

## Analysis Parameters

### Stimulus Definition

- **Stimulus Epoch Index**: The epoch containing the current injection step (0-based). Most protocols use index 2, but this depends on your ABF protocol structure.

### Spike Detection

- **Detection Threshold (mV)**: Voltage threshold for spike detection. Default: -20 mV
- **Derivative Threshold (mV/ms)**: Minimum dV/dt to qualify as a spike. Default: 10 mV/ms

## Basic Features

The Basic tab provides commonly used features with friendly names:

| Feature | Description |
|---------|-------------|
| **Spike Count** | Number of action potentials detected during the stimulus period |
| **Resting Voltage** | Mean membrane voltage before stimulus onset (mV) |
| **Steady-State Voltage** | Mean voltage at the end of the stimulus period (mV) |
| **Voltage Deflection** | Difference between steady-state and resting voltage (mV) |
| **Input Resistance** | Membrane input resistance calculated from voltage deflection (MΩ) |
| **Mean Firing Frequency** | Average firing rate during the stimulus period (Hz) |
| **Latency to First Spike** | Time from stimulus onset to first action potential (ms) |
| **Membrane Time Constant** | Tau from exponential fit of voltage response to hyperpolarizing current (ms) |

### Advanced Features

The Advanced tab provides access to the complete [eFEL](https://github.com/BlueBrain/eFEL) feature library (200+ features). See the [eFEL documentation](https://efel.readthedocs.io/en/latest/eFeatures.html) for detailed descriptions.

## Output Formats

### CSV Export
A flat table with all sweeps and selected features. Each row represents one sweep from one file.

### Excel Export
Multi-sheet workbook where each sheet contains one feature:
- Rows: Sweep number and current step
- Columns: Individual files
- Useful for comparing the same feature across multiple recordings

### PDF Export
Summary plots (2 files per page, A4 landscape):
- Raw voltage traces
- Spike count vs injected current (F-I curve)
- Phase-plane plot (dV/dt vs V) at ~2x rheobase

## Calculated Metrics

In addition to eFEL features, Spike Doctor calculates:

- **Capacitance (pF)**: Derived from time constant and input resistance: `Cm = τ / Rin × 1000`
  - Only calculated for hyperpolarizing sweeps without spikes

## File Support

Currently supports:
- **Format**: Axon Binary Format (ABF) via [pyABF](https://github.com/swharden/pyABF)
- **Recording Mode**: Current clamp (voltage recorded, current commanded)

## Dependencies

- [Shiny for Python](https://shiny.posit.co/py/) - Web framework
- [pyABF](https://github.com/swharden/pyABF) - ABF file reading
- [eFEL](https://github.com/BlueBrain/eFEL) - Electrophysiological feature extraction
- [pandas](https://pandas.pydata.org/) - Data manipulation
- [matplotlib](https://matplotlib.org/) - Plotting
- [numpy](https://numpy.org/) - Numerical computing
- [openpyxl](https://openpyxl.readthedocs.io/) - Excel file writing

## License

See LICENSE file for details.

## Acknowledgments

- [eFEL](https://github.com/BlueBrain/eFEL) by the Blue Brain Project for the comprehensive feature extraction library
-[pyABF](https://github.com/swharden/pyABF) by [swharden](https://github.com/swharden) for the excellent ABF file reader
