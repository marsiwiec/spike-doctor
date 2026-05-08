"""Tests for modules.stimulus."""

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import numpy as np

from modules import stimulus
from modules.models import StimulusWindow


@dataclass
class _MockEpochWaveform:
    p1s: list
    p2s: list
    levels: list


class TestFindStimulusWindow:
    def _make_abf(self, sweep_point_count=10000):
        abf = MagicMock()
        abf.abfID = "test"
        abf.sweepPointCount = sweep_point_count
        abf.dataSecPerPoint = 1e-5
        return abf

    @patch("modules.stimulus.get_epoch_waveform_for_sweep", return_value=None)
    def test_no_epochs_uses_full_sweep(self, _mock):
        abf = self._make_abf(sweep_point_count=1000)
        result = stimulus.find_stimulus_window(abf, 0, 0, 2)
        assert result == StimulusWindow(15, 1000, None, None)

    @patch(
        "modules.stimulus.get_epoch_waveform_for_sweep",
        return_value=_MockEpochWaveform([0, 100], [0, 900], [0, 50]),
    )
    def test_two_epochs_uses_post_holding(self, _mock):
        abf = self._make_abf()
        result = stimulus.find_stimulus_window(abf, 0, 0, 2)
        assert result == StimulusWindow(100, 900, 1, _mock.return_value)

    @patch(
        "modules.stimulus.get_epoch_waveform_for_sweep",
        return_value=_MockEpochWaveform(
            [0, 100, 500, 900, 1000], [0, 500, 900, 1000, 1000], [0, 10, 50, 0, 0]
        ),
    )
    def test_many_epochs_uses_requested_index(self, _mock):
        abf = self._make_abf()
        result = stimulus.find_stimulus_window(abf, 0, 0, 2)
        assert result == StimulusWindow(500, 900, 2, _mock.return_value)

    @patch(
        "modules.stimulus.get_epoch_waveform_for_sweep",
        return_value=_MockEpochWaveform(
            [0, 100, 500, 900, 1000], [0, 500, 900, 1000, 1000], [0, 10, 50, 0, 0]
        ),
    )
    def test_index_out_of_range_falls_back(self, _mock):
        abf = self._make_abf()
        result = stimulus.find_stimulus_window(abf, 0, 0, 10)
        assert result.epoch_idx_used == 1


class TestComputeStimulusCurrent:
    def _make_abf(self):
        abf = MagicMock()
        abf.abfID = "test"
        return abf

    @patch(
        "modules.stimulus.get_sweep_c",
        return_value=np.array([0.0] * 50 + [100.0] * 50),
    )
    def test_from_sweep_c_median(self, _mock):
        abf = self._make_abf()
        window = StimulusWindow(50, 100, 1, None)
        result = stimulus.compute_stimulus_current(abf, 0, 0, window)
        assert result == 100.0

    @patch("modules.stimulus.get_sweep_c", return_value=None)
    @patch("modules.stimulus.get_stimulus_dac_channel", return_value=0)
    @patch("modules.stimulus.get_dac_units", return_value="pA")
    def test_fallback_to_epoch_level(self, _mock_units, _mock_dac, _mock_sweep_c):
        abf = self._make_abf()
        epoch = _MockEpochWaveform([0, 10], [10, 100], [0, -200.0])
        window = StimulusWindow(10, 100, 1, epoch)
        result = stimulus.compute_stimulus_current(abf, 0, 0, window)
        assert result == -200.0

    @patch("modules.stimulus.get_sweep_c", return_value=None)
    @patch("modules.stimulus.get_stimulus_dac_channel", return_value=0)
    @patch("modules.stimulus.get_dac_units", return_value="nA")
    def test_fallback_na_conversion(self, _mock_units, _mock_dac, _mock_sweep_c):
        abf = self._make_abf()
        epoch = _MockEpochWaveform([0, 10], [10, 100], [0, -0.5])
        window = StimulusWindow(10, 100, 1, epoch)
        result = stimulus.compute_stimulus_current(abf, 0, 0, window)
        assert result == -500.0

    @patch("modules.stimulus.get_sweep_c", return_value=None)
    @patch("modules.stimulus.get_stimulus_dac_channel", return_value=0)
    @patch("modules.stimulus.get_dac_units", return_value="")
    def test_no_epoch_defaults_to_zero(self, _mock_units, _mock_dac, _mock_sweep_c):
        abf = self._make_abf()
        window = StimulusWindow(10, 100, None, None)
        result = stimulus.compute_stimulus_current(abf, 0, 0, window)
        assert result == 0.0


class TestGetFileTypeInfo:
    def test_gap_free(self):
        abf = MagicMock()
        abf.nOperationMode = 3
        abf._dacSection = MagicMock()
        abf._dacSection.nWaveformEnable = [1]
        info = stimulus.get_file_type_info(abf)
        assert info["is_gap_free"] is True

    def test_stimulus_free(self):
        abf = MagicMock()
        abf.nOperationMode = 0
        abf._dacSection = MagicMock()
        abf._dacSection.nWaveformEnable = [0, 0]
        info = stimulus.get_file_type_info(abf)
        assert info["is_stimulus_free"] is True
        assert info["has_protocol_epochs"] is False
