from dataclasses import dataclass
from typing import Any

import pandas as pd
import pyabf
from matplotlib.figure import Figure


@dataclass(frozen=True)
class LogEntry:
    """A single log entry from analysis or loading operations."""

    level: str
    abf_id: str
    sweep_num: int | None
    message: str


@dataclass
class LoadedFile:
    """Represents an ABF file loaded into the session."""

    original_filename: str
    filepath: str
    abf_object: pyabf.ABF | None = None
    load_error: str | None = None

    @property
    def is_loaded(self) -> bool:
        return self.abf_object is not None and self.load_error is None


@dataclass
class AnalysisResult:
    """Result of analysing a single ABF file."""

    analysis_df: pd.DataFrame | None = None
    debug_plot_fig: Figure | None = None

    @property
    def has_data(self) -> bool:
        return self.analysis_df is not None and not self.analysis_df.empty


@dataclass
class StimulusWindow:
    """Defines the stimulus window for a given sweep."""

    start_pt: float
    end_pt: float
    epoch_idx_used: int | None
    epoch_waveform: Any  # pyabf EpochWaveform or None

    @property
    def duration_pts(self) -> float:
        return self.end_pt - self.start_pt
