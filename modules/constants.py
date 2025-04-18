from typing import List

# ==============================================================================
# Configuration Constants
# ==============================================================================

# --- Analysis Settings ---
# IMPORTANT: Set this index based on your protocol's stimulus epoch.
# This index refers to the epoch within abf.sweepEpochs that defines the main
# stimulus start and end times used for eFEL and other calculations.
# Check your ABF header in Clampfit or pyABF to determine the correct index.

DEFAULT_EFEL_FEATURES: List[str] = [
    "spike_count",
    "voltage_base",
    "steady_state_voltage_stimend",
    "voltage_deflection",
    "ohmic_input_resistance",
    "mean_frequency",
    "time_to_first_spike",
    "decay_time_constant_after_stim",
]
# Features needed internally even if not selected
REQUIRED_INTERNAL_EFEL_FEATURES: List[str] = [
    "spike_count",
    "voltage_base",
    "decay_time_constant_after_stim",
    "ohmic_input_resistance",
]

# --- Plotting & Debugging ---
MAX_RAW_PLOT_SWEEPS: int = 100  
CURRENT_COL_NAME = "current_step_pA"
