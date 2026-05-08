import numpy as np
import pyabf

from modules import logger as _logger_mod
from modules.models import StimulusWindow


def get_stimulus_dac_channel(abf: pyabf.ABF, adc_channel: int = 0) -> int:
    """Find the DAC channel that actually carries the stimulus waveform.

    Defaults to *adc_channel*, but scans other DACs when the default has
    no enabled waveform or only zero-amplitude epochs.
    """
    if not hasattr(abf, "_dacSection"):
        return adc_channel

    n_waveform_enable = getattr(abf._dacSection, "nWaveformEnable", [])

    def _has_meaningful_epochs(dac_ch: int) -> bool:
        try:
            epoch_table = pyabf.waveform.EpochTable(abf, dac_ch)
            if len(epoch_table.epochs) > 0:
                return any(
                    ep.level != 0 or ep.levelDelta != 0 for ep in epoch_table.epochs
                )
        except Exception:
            pass
        return False

    if adc_channel < len(n_waveform_enable) and n_waveform_enable[adc_channel] == 1:
        if _has_meaningful_epochs(adc_channel):
            return adc_channel

    for dac_ch in range(len(n_waveform_enable)):
        if n_waveform_enable[dac_ch] == 1 and _has_meaningful_epochs(dac_ch):
            return dac_ch

    return adc_channel


def get_dac_units(abf: pyabf.ABF, dac_channel: int) -> str:
    if hasattr(abf, "_dacSection") and hasattr(abf, "_stringsSection"):
        try:
            units_index = abf._dacSection.lDACChannelUnitsIndex[dac_channel]
            return abf._stringsSection._indexedStrings[units_index]
        except (IndexError, AttributeError):
            pass
    return getattr(abf, "sweepUnitsC", "")


def get_epoch_waveform_for_sweep(abf: pyabf.ABF, sweep_num: int, adc_channel: int = 0):
    dac_channel = get_stimulus_dac_channel(abf, adc_channel)
    try:
        epoch_table = pyabf.waveform.EpochTable(abf, dac_channel)
        return epoch_table.epochWaveformsBySweep[sweep_num]
    except Exception:
        return None


def get_sweep_c(
    abf: pyabf.ABF, sweep_num: int, channel: int = 0
) -> np.ndarray | None:
    """Return the command current trace for a sweep in pA.

    If the recorded *sweepC* is missing or all zeros/NaNs, reconstruct it
    from the epoch table of the correct DAC channel.
    """
    try:
        abf.setSweep(sweep_num, channel=channel)
    except Exception:
        return None

    sweep_c = getattr(abf, "sweepC", None)
    has_valid = (
        isinstance(sweep_c, np.ndarray)
        and sweep_c.size > 0
        and not np.all(np.isnan(sweep_c))
        and not np.all(sweep_c == 0)
    )
    if has_valid:
        return sweep_c

    try:
        sweep_points = getattr(abf, "sweepPointCount", 0)
        if sweep_points == 0:
            return None

        dac_channel = get_stimulus_dac_channel(abf, channel)
        epoch_waveform = get_epoch_waveform_for_sweep(abf, sweep_num, channel)
        if epoch_waveform is None:
            return None

        p1s = epoch_waveform.p1s
        p2s = epoch_waveform.p2s
        levels = epoch_waveform.levels
        if not (len(p1s) == len(p2s) == len(levels)):
            return None

        unit_raw = get_dac_units(abf, dac_channel).lower()
        if "na" in unit_raw:
            scale = 1000.0
        elif "pa" in unit_raw or not unit_raw:
            scale = 1.0
        else:
            scale = 1.0

        reconstructed = np.zeros(sweep_points, dtype=float)
        for p1, p2, level in zip(p1s, p2s, levels):
            start = max(0, int(p1))
            end = min(sweep_points, int(p2)) if int(p2) > 0 else sweep_points
            if start < end:
                reconstructed[start:end] = level * scale

        return reconstructed
    except Exception:
        return None


def get_file_type_info(abf: pyabf.ABF) -> dict:
    info = {
        "is_gap_free": False,
        "is_stimulus_free": False,
        "is_current_zero": False,
        "has_protocol_epochs": False,
    }

    op_mode = getattr(abf, "nOperationMode", None)
    if op_mode == 3:
        info["is_gap_free"] = True

    n_waveform_enable = []
    if hasattr(abf, "_dacSection"):
        n_waveform_enable = getattr(abf._dacSection, "nWaveformEnable", [])

    if not any(n_waveform_enable):
        info["is_stimulus_free"] = True
        return info

    dac_channel = get_stimulus_dac_channel(abf)
    try:
        epoch_table = pyabf.waveform.EpochTable(abf, dac_channel)
        info["has_protocol_epochs"] = len(epoch_table.epochs) > 0
        if info["has_protocol_epochs"]:
            all_zero = all(
                ep.level == 0 and ep.levelDelta == 0 for ep in epoch_table.epochs
            )
            info["is_current_zero"] = all_zero
    except Exception:
        info["is_stimulus_free"] = True

    return info


def find_stimulus_window(
    abf: pyabf.ABF,
    sweep_num: int,
    channel: int,
    stimulus_epoch_index: int,
) -> StimulusWindow:
    """Determine the stimulus window for a given sweep."""
    epoch_waveform = get_epoch_waveform_for_sweep(abf, sweep_num, channel)
    num_epochs = len(epoch_waveform.p1s) if epoch_waveform else 0
    abf_id = getattr(abf, "abfID", "?")

    if num_epochs == 0:
        sweep_points = getattr(abf, "sweepPointCount", 0)
        start = max(1, sweep_points // 64)
        _logger_mod.get_logger().log(
            "INFO", abf_id, sweep_num,
            "No stimulus epochs found. Treating as free-running.",
        )
        return StimulusWindow(start, sweep_points, None, epoch_waveform)

    if num_epochs == 2:
        _logger_mod.get_logger().log(
            "INFO", abf_id, sweep_num,
            "No protocol epochs found. Using post-holding period.",
        )
        return StimulusWindow(
            epoch_waveform.p1s[1], epoch_waveform.p2s[1], 1, epoch_waveform
        )

    if num_epochs > 2:
        if stimulus_epoch_index < num_epochs:
            return StimulusWindow(
                epoch_waveform.p1s[stimulus_epoch_index],
                epoch_waveform.p2s[stimulus_epoch_index],
                stimulus_epoch_index,
                epoch_waveform,
            )
        _logger_mod.get_logger().log(
            "WARN", abf_id, sweep_num,
            f"Epoch index {stimulus_epoch_index} out of range. Using epoch 1.",
        )
        return StimulusWindow(
            epoch_waveform.p1s[1], epoch_waveform.p2s[1], 1, epoch_waveform
        )

    # num_epochs == 1 or unexpected
    sweep_points = getattr(abf, "sweepPointCount", 0)
    start = max(1, sweep_points // 64)
    _logger_mod.get_logger().log(
        "WARN", abf_id, sweep_num,
        f"Unexpected epoch count {num_epochs}. Using full sweep.",
    )
    return StimulusWindow(start, sweep_points, None, epoch_waveform)


def compute_stimulus_current(
    abf: pyabf.ABF,
    sweep_num: int,
    channel: int,
    window: StimulusWindow,
) -> float:
    """Compute stimulus current in pA for the given window."""
    sweep_c = get_sweep_c(abf, sweep_num, channel=channel)
    stimulus_current_pA = np.nan

    if sweep_c is not None and len(sweep_c) > int(window.start_pt):
        end_pt = (
            min(int(window.end_pt), len(sweep_c))
            if int(window.end_pt) > 0
            else len(sweep_c)
        )
        if end_pt > int(window.start_pt):
            stimulus_current_pA = float(
                np.median(sweep_c[int(window.start_pt) : end_pt])
            )

    if np.isfinite(stimulus_current_pA) and not np.isclose(stimulus_current_pA, 0):
        return stimulus_current_pA

    if window.epoch_waveform is not None and window.epoch_idx_used is not None:
        raw_level = window.epoch_waveform.levels[window.epoch_idx_used]
    else:
        raw_level = 0.0

    dac_channel = get_stimulus_dac_channel(abf, channel)
    unit_raw = get_dac_units(abf, dac_channel).lower()

    if "na" in unit_raw:
        stimulus_current_pA = raw_level * 1000.0
    elif "pa" in unit_raw or not unit_raw:
        stimulus_current_pA = raw_level
    else:
        _logger_mod.get_logger().log(
            "WARN", getattr(abf, "abfID", "?"), sweep_num,
            f"Command units '{unit_raw}'. Assuming pA.",
        )
        stimulus_current_pA = raw_level

    if not np.isfinite(stimulus_current_pA):
        _logger_mod.get_logger().log(
            "WARN", getattr(abf, "abfID", "?"), sweep_num,
            "Stimulus current undetermined. Defaulting to 0 pA.",
        )
        stimulus_current_pA = 0.0

    return stimulus_current_pA
