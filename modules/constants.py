from typing import List

# ==============================================================================
# Configuration Constants
# ==============================================================================

DEFAULT_EFEL_FEATURES: List[str] = [
    "spike_count",
    "voltage_base",
    "steady_state_voltage_stimend",
    "voltage_deflection",
    "ohmic_input_resistance",
    "mean_frequency",
    "time_to_first_spike",
    "time_constant",
]
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
