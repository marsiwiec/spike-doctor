"""Tests for modules.helper."""

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from modules import helper


class TestParseEfelValue:
    def test_none_input_returns_nan(self):
        assert helper.parse_efel_value(None, "spike_count") == [np.nan]

    def test_missing_key_returns_nan(self):
        assert helper.parse_efel_value({"other": 1}, "spike_count") == [np.nan]

    def test_scalar_float(self):
        assert helper.parse_efel_value({"spike_count": 5.0}, "spike_count") == [5.0]

    def test_list_of_floats(self):
        assert helper.parse_efel_value({"isi": [10.0, 20.0]}, "isi") == [10.0, 20.0]

    def test_empty_list_returns_nan(self):
        assert helper.parse_efel_value({"isi": []}, "isi") == [np.nan]

    def test_nan_value_returns_nan(self):
        result = helper.parse_efel_value({"v": np.nan}, "v")
        assert len(result) == 1
        assert np.isnan(result[0])

    def test_non_numeric_returns_nan(self):
        assert helper.parse_efel_value({"v": "bad"}, "v") == [np.nan]


class TestCleanExcelSheetName:
    def test_simple(self):
        assert helper.clean_excel_sheet_name("spike_count") == "Spike Count"

    def test_truncates_to_31(self):
        long_name = "a" * 50
        assert len(helper.clean_excel_sheet_name(long_name)) == 31

    def test_strips_invalid_chars(self):
        assert helper.clean_excel_sheet_name("feat:/name") == "FeatName"


class TestGetFeatureUnits:
    def test_resistance(self):
        assert helper.get_feature_units("ohmic_input_resistance") == "MΩ"

    def test_time_constant(self):
        assert helper.get_feature_units("time_constant") == "ms"

    def test_capacitance(self):
        assert helper.get_feature_units("capacitance_pF") == "pF"

    def test_frequency(self):
        assert helper.get_feature_units("mean_frequency") == "Hz"

    def test_latency(self):
        assert helper.get_feature_units("time_to_first_spike") == "ms"

    def test_voltage(self):
        assert helper.get_feature_units("voltage_base") == "mV"

    def test_unknown(self):
        assert helper.get_feature_units("unknown_feature") == ""


class TestIsValidAnalysisDf:
    def test_none_is_invalid(self):
        assert helper.is_valid_analysis_df(None) is False

    def test_empty_df_is_invalid(self):
        assert helper.is_valid_analysis_df(pd.DataFrame()) is False

    def test_non_df_is_invalid(self):
        assert helper.is_valid_analysis_df("not a df") is False

    def test_valid_df(self):
        df = pd.DataFrame({"a": [1, 2]})
        assert helper.is_valid_analysis_df(df) is True


class TestFigToSrcAndClose:
    def test_none_returns_none(self):
        assert helper.fig_to_src_and_close(None) is None

    def test_valid_figure(self):
        fig = Figure()
        fig.add_subplot(111)
        src = helper.fig_to_src_and_close(fig)
        assert src is not None
        assert src.startswith("data:image/png;base64,")
