# Spike Doctor

A Shiny web application for analyzing whole-cell current-clamp recordings. Spike Doctor detects action potentials and extracts electrophysiological features from Axon Instruments ABF files using [pyABF](https://github.com/swharden/pyABF) and [eFEL](https://github.com/BlueBrain/eFEL).

<img src="https://github.com/marsiwiec/spike-doctor/blob/main/assets/spike-doctor.png?raw=true" width="80%">

## Features

- **Batch processing** of multiple ABF files
- **Spike detection** with configurable voltage and derivative thresholds
- **Feature extraction** via  [eFEL](https://github.com/BlueBrain/eFEL) (200+ features)
- **Interactive plots**: raw traces, F-I curves, and phase-plane plots
- **Exports**: CSV, multi-sheet Excel, and PDF summary plots
- **Debug mode** with per-sweep stimulus window visualizations

## Installation

### Docker (recommended)

```bash
git clone https://github.com/marsiwiec/spike-doctor.git
cd spike-doctor
docker compose up
```

Then open http://localhost:8000.

### Standalone executables

Pre-built binaries for all platforms are available on the [Releases](https://github.com/marsiwiec/spike-doctor/releases) page:

- **Windows**: `SpikeDoctor-windows.zip` -- extract and run `SpikeDoctor.exe`
- **macOS**: `SpikeDoctor-macos.zip` -- extract and run the `SpikeDoctor` binary
- **Linux**: `SpikeDoctor-linux.zip` -- extract and run the `SpikeDoctor` binary

The console window must stay open while the app is running.

> To build from source:
> ```bash
> uv pip install pyinstaller
> pyinstaller SpikeDoctor.spec
> ```

### Manual (uv)

```bash
git clone https://github.com/marsiwiec/spike-doctor.git
cd spike-doctor
uv sync
uv run shiny run app.py
```

Then open http://127.0.0.1:8000.

### Nix + uv

For [Nix](https://nixos.org/) users, a [flake-parts](https://flake.parts/) dev shell is included:

```bash
git clone https://github.com/marsiwiec/spike-doctor.git
cd spike-doctor
nix develop   # or `direnv allow` if using nix-direnv
uv sync
uv run shiny run app.py
```

## Quick Start

1. **Upload** one or more `.abf` files.
2. **Adjust parameters** if your protocol uses a non-standard stimulus epoch or spike thresholds.
3. **Select features** from the Basic or Advanced tabs.
4. **View results** in the Summary Plots, Results Table, and Debug Plots tabs.
5. **Export** as CSV, Excel, or PDF.

## Parameters

| Parameter | Description | Default |
|---|---|---|
| **Stimulus Epoch Index** | Epoch containing the current step (0-based). Most protocols use epoch 2. | 2 |
| **Detection Threshold** | Voltage threshold for spike detection (mV). | -20 mV |
| **Derivative Threshold** | Minimum dV/dt to qualify as a spike (mV/ms). | 10 mV/ms |

## Basic Features

The Basic tab exposes commonly used eFEL features with readable names (some of them are simply needed for further calculations):

| Feature | Description |
|---|---|
| **Spike Count** | Number of action potentials generated during the stimulus period |
| **Resting Voltage** | Mean membrane potential before stimulus onset (mV) |
| **Steady-State Voltage** | Mean membrane potential at the end of the stimulus (mV) |
| **Voltage Deflection** | Steady-state minus resting voltage (mV) |
| **Input Resistance** | From voltage deflection (MOhm) |
| **Mean Firing Frequency** | Average spiking rate during the stimulus (Hz) |
| **Latency to First Spike** | Time from stimulus onset to first AP (ms) |
| **Membrane Time Constant** | Tau fitted to voltage response to a hyperpolarizing current step (ms) |

The Advanced tab exposes the full  [eFEL](https://github.com/BlueBrain/eFEL) feature library. See the [eFEL documentation](https://efel.readthedocs.io/en/latest/eFeatures.html) for details.

## Output Formats

- **CSV** -- Flat table: one row per sweep per file.
- **Excel** -- One sheet per feature. Rows are sweeps/current steps; columns are files.
- **PDF** -- Summary plots (2 files per A4 landscape page): raw traces, F-I curve, and phase-plane plot at ~2x rheobase. Perfect for printing out/displaying for overview.

## Calculated Metrics

- **Capacitance (pF)**: `Cm = tau / Rin x 1000`. Only calculated for hyperpolarizing sweeps without spikes.

## File Support

- **Format**: Axon Binary Format (ABF)
- **Recording mode**: Current clamp only so far

## License

See LICENSE.

## Acknowledgments

- [eFEL](https://github.com/BlueBrain/eFEL) by the Blue Brain Project
- [pyABF](https://github.com/swharden/pyABF) by [swharden](https://github.com/swharden)
