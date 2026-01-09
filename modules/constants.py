from typing import List, Dict, Tuple

# ==============================================================================
# Configuration Constants
# ==============================================================================

# Basic features with display names and descriptions for tooltips
# Format: feature_name -> (display_name, description)
BASIC_EFEL_FEATURES: Dict[str, Tuple[str, str]] = {
    "spike_count": (
        "Spike Count",
        "Number of action potentials detected during the stimulus period"
    ),
    "voltage_base": (
        "Resting Voltage",
        "Mean voltage before stimulus onset (mV)"
    ),
    "steady_state_voltage_stimend": (
        "Steady-State Voltage",
        "Mean voltage at the end of stimulus period (mV)"
    ),
    "voltage_deflection": (
        "Voltage Deflection",
        "Difference between steady-state and resting voltage (mV)"
    ),
    "ohmic_input_resistance": (
        "Input Resistance",
        "Membrane input resistance calculated from voltage deflection (MΩ)"
    ),
    "mean_frequency": (
        "Mean Firing Frequency",
        "Average firing rate during the stimulus period (Hz)"
    ),
    "time_to_first_spike": (
        "Latency to First Spike",
        "Time from stimulus onset to first action potential (ms)"
    ),
    "time_constant": (
        "Membrane Time Constant",
        "Tau from exponential fit of voltage response to hyperpolarizing current (ms)"
    ),
}

DEFAULT_EFEL_FEATURES: List[str] = list(BASIC_EFEL_FEATURES.keys())

# Features needed internally even if not selected
REQUIRED_INTERNAL_EFEL_FEATURES: List[str] = [
    "spike_count",
    "voltage_base",
    "time_constant",
    "ohmic_input_resistance",
]

# --- Plotting & Debugging ---
MAX_RAW_PLOT_SWEEPS: int = 100
CURRENT_COL_NAME = "current_step_pA"
