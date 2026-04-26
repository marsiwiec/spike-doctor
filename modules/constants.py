from typing import List, Dict, Tuple

BASIC_EFEL_FEATURES: Dict[str, Tuple[str, str]] = {
    "spike_count": (
        "Spike Count",
        "Number of action potentials detected during the stimulus period",
    ),
    "voltage_base": (
        "Resting Voltage",
        "Mean voltage before stimulus onset (mV)",
    ),
    "steady_state_voltage_stimend": (
        "Steady-State Voltage",
        "Mean voltage at the end of stimulus period (mV)",
    ),
    "voltage_deflection": (
        "Voltage Deflection",
        "Difference between steady-state and resting voltage (mV)",
    ),
    "ohmic_input_resistance": (
        "Input Resistance",
        "Membrane input resistance calculated from voltage deflection (MΩ)",
    ),
    "mean_frequency": (
        "Mean Firing Frequency",
        "Average firing rate during the stimulus period (Hz)",
    ),
    "time_to_first_spike": (
        "Latency to First Spike",
        "Time from stimulus onset to first action potential (ms)",
    ),
    "time_constant": (
        "Membrane Time Constant",
        "Tau from exponential fit of voltage response to hyperpolarizing current (ms)",
    ),
}

DEFAULT_EFEL_FEATURES: List[str] = list(BASIC_EFEL_FEATURES.keys())

# Needed internally even if the user does not select them.
REQUIRED_INTERNAL_EFEL_FEATURES: List[str] = [
    "spike_count",
    "voltage_base",
    "time_constant",
    "ohmic_input_resistance",
]

# These features are meaningless when there is no stimulus step.
STIMULUS_DEPENDENT_EFEL_FEATURES: List[str] = [
    "time_constant",
    "time_constant_slow",
    "ohmic_input_resistance",
    "voltage_deflection",
    "voltage_deflection_begin",
    "voltage_deflection_vb_ssse",
    "steady_state_voltage_stimend",
    "voltage_after_stim",
    "single_decay_time_constant_after_stim",
    "multiple_decay_time_constant_after_stim",
]

MAX_RAW_PLOT_SWEEPS: int = 100
CURRENT_COL_NAME = "current_step_pA"
