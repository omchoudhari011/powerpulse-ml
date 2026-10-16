"""Chronological splitting and train-only normalisation."""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import SCALED_EXOG_COLUMNS
from .errors import ArtifactError, DatasetError


def chronological_split(n_rows: int, train_frac: float, val_frac: float) -> Tuple[int, int]:
    """Return ``(train_end, val_end)`` row boundaries: train=[0,a) val=[a,b) test=[b,n). No shuffling."""
    a = int(n_rows * train_frac)
    b = int(n_rows * (train_frac + val_frac))
    if not (0 < a < b < n_rows):
        raise DatasetError(f"Cannot split {n_rows} rows chronologically with the configured fractions.")
    return a, b


class Preprocessor:
    """Standardises target and continuous exogenous features using TRAIN rows only.

    ``columns`` = [target] + exogenous columns, in that order. ``equipment_status`` (on/off) and
    the cyclical calendar features are passed through unscaled.
    """

    def __init__(self, target: str, exog_columns: Sequence[str]):
        self.target = target
        self.exog_columns: List[str] = list(exog_columns)
        self.columns: List[str] = [target] + self.exog_columns
        self.scale_columns: List[str] = [target] + [c for c in self.exog_columns if c in SCALED_EXOG_COLUMNS]
        self.scaler: StandardScaler | None = None

    def fit(self, train_frame: pd.DataFrame) -> "Preprocessor":
        self._check_columns(train_frame)
        x = train_frame[self.scale_columns].to_numpy(dtype=float)
        scaler = StandardScaler().fit(x)  # NaNs are ignored by StandardScaler
        if np.isnan(scaler.mean_).any():
            bad = [c for c, m in zip(self.scale_columns, scaler.mean_) if np.isnan(m)]
            raise DatasetError(f"Column(s) {bad} have no valid values in the training period.")
        self.scaler = scaler
        return self

    def _check_columns(self, frame: pd.DataFrame) -> None:
        missing = [c for c in self.columns if c not in frame.columns]
        if missing:
            raise DatasetError(f"Columns required by the preprocessor are missing: {missing}")

    def _require_fit(self) -> StandardScaler:
        if self.scaler is None:
            raise ArtifactError("Preprocessor has not been fitted.")
        return self.scaler

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        scaler = self._require_fit()
        self._check_columns(frame)
        out = frame[self.columns].astype(float).copy()
        out[self.scale_columns] = scaler.transform(out[self.scale_columns].to_numpy(dtype=float))
        return out

    def inverse_target(self, scaled: np.ndarray) -> np.ndarray:
        scaler = self._require_fit()
        return np.asarray(scaled, dtype=float) * scaler.scale_[0] + scaler.mean_[0]

    # --- persistence (plain dict so the file does not depend on this class's import path)
    def save(self, path) -> None:
        joblib.dump({"target": self.target, "exog_columns": self.exog_columns,
                     "scaler": self._require_fit()}, path)

    @classmethod
    def load(cls, path) -> "Preprocessor":
        state = joblib.load(path)
        pre = cls(state["target"], state["exog_columns"])
        pre.scaler = state["scaler"]
        return pre
