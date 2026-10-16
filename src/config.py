"""Central configuration for the PowerPulse ML module.

All values below are *starting* values, not tuned optima. Every one of them can be
overridden with a JSON file (``--config``) or with command-line flags.

Sequence length / horizon / lags are expressed in HOURS and converted to a number of
time steps using the dataset's detected sampling interval (see
``feature_engineering.resolve_steps``).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .errors import ConfigError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"      # documents searched by the RAG assistant (src/rag)
RAG_INDEX_DIR = ARTIFACTS_DIR / "rag"

TIMESTAMP_COLUMN = "timestamp"
TARGET_COLUMNS = ("demand_kw", "energy_kwh")  # priority order for auto target selection
TARGET_UNITS = {"demand_kw": "kW", "energy_kwh": "kWh"}
EXOG_COLUMNS = ("temperature", "humidity", "occupancy", "equipment_status")
SCALED_EXOG_COLUMNS = ("temperature", "humidity", "occupancy")  # equipment_status is on/off
KNOWN_COLUMNS = (TIMESTAMP_COLUMN,) + TARGET_COLUMNS + EXOG_COLUMNS
MODEL_NAMES = ("lstm", "mlp")
METRIC_NAMES = ("mae", "rmse", "wape")


@dataclass
class DataConfig:
    timestamp_column: str = TIMESTAMP_COLUMN
    timestamp_format: Optional[str] = None      # e.g. "%d/%m/%Y %H:%M"; None = infer
    target: Optional[str] = None                # None = demand_kw if present else energy_kwh
    train_frac: float = 0.70
    val_frac: float = 0.15                      # test = 1 - train - val
    duplicate_policy: str = "last"              # first | last | mean
    max_gap_hours: float = 3.0                  # longest gap that may be forward-filled
    temperature_range: Tuple[float, float] = (-60.0, 150.0)  # plausible C or F readings
    min_train_samples: int = 100
    min_val_samples: int = 24
    min_test_samples: int = 24


@dataclass
class FeatureConfig:
    lag_hours: Tuple[float, ...] = (1, 2, 3, 24, 48, 168)
    use_calendar: bool = True


@dataclass
class TrainConfig:
    input_hours: float = 24                     # LSTM look-back window
    horizon_hours: float = 24                   # forecast horizon (both models)
    batch_size: int = 32
    max_epochs: int = 100
    early_stopping_patience: int = 10
    learning_rate: float = 1e-3
    seed: int = 42
    shuffle_train: bool = True                  # shuffles batches *inside* the train split only
    clip_non_negative: bool = True              # kW / kWh cannot be negative
    selection_metric: str = "rmse"              # validation metric used to pick the model
    deterministic_ops: bool = False             # True = TF op determinism (slower)


@dataclass
class LSTMConfig:
    num_layers: int = 1                         # 1 or 2
    units: int = 64
    dropout: float = 0.2


@dataclass
class MLPConfig:
    hidden_units: Tuple[int, ...] = (128, 64)
    dropout: float = 0.0


_SECTIONS = {
    "data": DataConfig,
    "features": FeatureConfig,
    "train": TrainConfig,
    "lstm": LSTMConfig,
    "mlp": MLPConfig,
}


def _build_section(cls, values: Dict[str, Any]):
    valid = {f.name for f in fields(cls)}
    unknown = set(values) - valid
    if unknown:
        raise ConfigError(f"Unknown {cls.__name__} option(s): {sorted(unknown)}. Valid: {sorted(valid)}")
    inst = cls(**values)
    defaults = cls()
    for f in fields(cls):  # JSON has no tuples
        if isinstance(getattr(defaults, f.name), tuple):
            setattr(inst, f.name, tuple(getattr(inst, f.name)))
    return inst


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    lstm: LSTMConfig = field(default_factory=LSTMConfig)
    mlp: MLPConfig = field(default_factory=MLPConfig)

    # ------------------------------------------------------------------ io
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        unknown = set(d) - set(_SECTIONS)
        if unknown:
            raise ConfigError(f"Unknown config section(s): {sorted(unknown)}")
        cfg = cls(**{name: _build_section(sec, d[name]) for name, sec in _SECTIONS.items() if name in d})
        cfg.validate()
        return cfg

    @classmethod
    def from_json(cls, path) -> "Config":
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Config file not found: {p}")
        try:
            return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))
        except json.JSONDecodeError as exc:
            raise ConfigError(f"Config file {p} is not valid JSON: {exc}") from exc

    def save_json(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    # ---------------------------------------------------------- validation
    def validate(self) -> "Config":
        d, f, t, l, m = self.data, self.features, self.train, self.lstm, self.mlp
        if not (0 < d.train_frac < 1 and 0 < d.val_frac < 1 and d.train_frac + d.val_frac < 1):
            raise ConfigError("train_frac and val_frac must be in (0,1) and sum to < 1 (the rest is the test set).")
        if d.duplicate_policy not in ("first", "last", "mean"):
            raise ConfigError("duplicate_policy must be one of: first, last, mean.")
        if d.target is not None and d.target not in TARGET_COLUMNS:
            raise ConfigError(f"target must be one of {TARGET_COLUMNS} (or null for automatic selection).")
        if d.max_gap_hours < 0:
            raise ConfigError("max_gap_hours must be >= 0.")
        if min(d.min_train_samples, d.min_val_samples, d.min_test_samples) < 1:
            raise ConfigError("min_*_samples must be >= 1.")
        if t.input_hours <= 0 or t.horizon_hours <= 0:
            raise ConfigError("input_hours and horizon_hours must be > 0.")
        if t.batch_size < 1 or t.max_epochs < 1 or t.early_stopping_patience < 1:
            raise ConfigError("batch_size, max_epochs and early_stopping_patience must be >= 1.")
        if t.learning_rate <= 0:
            raise ConfigError("learning_rate must be > 0.")
        if t.selection_metric not in METRIC_NAMES:
            raise ConfigError(f"selection_metric must be one of {METRIC_NAMES}.")
        if l.num_layers not in (1, 2):
            raise ConfigError("lstm.num_layers must be 1 or 2.")
        if l.units < 1 or not 0 <= l.dropout < 1:
            raise ConfigError("lstm.units must be >= 1 and lstm.dropout in [0,1).")
        if not m.hidden_units or min(m.hidden_units) < 1 or not 0 <= m.dropout < 1:
            raise ConfigError("mlp.hidden_units must be non-empty positive ints and mlp.dropout in [0,1).")
        if not f.lag_hours or min(f.lag_hours) <= 0:
            raise ConfigError("features.lag_hours must be a non-empty list of positive numbers.")
        return self


# ---------------------------------------------------------------------- CLI helpers
def add_config_arguments(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("configuration (all optional; defaults live in src/config.py)")
    g.add_argument("--config", help="JSON file with config overrides (same layout as Config.to_dict()).")
    g.add_argument("--target", choices=TARGET_COLUMNS, help="Force the forecasting target.")
    g.add_argument("--input-hours", type=float, help="LSTM look-back window in hours (default 24).")
    g.add_argument("--horizon-hours", type=float, help="Forecast horizon in hours (default 24).")
    g.add_argument("--epochs", type=int, help="Maximum epochs (default 100).")
    g.add_argument("--batch-size", type=int, help="Batch size (default 32).")
    g.add_argument("--learning-rate", type=float, help="Adam learning rate (default 0.001).")
    g.add_argument("--patience", type=int, help="Early-stopping patience (default 10).")
    g.add_argument("--seed", type=int, help="Random seed (default 42).")
    g.add_argument("--train-frac", type=float, help="Chronological train fraction (default 0.70).")
    g.add_argument("--val-frac", type=float, help="Chronological validation fraction (default 0.15).")
    g.add_argument("--max-gap-hours", type=float, help="Longest gap to forward-fill (default 3).")
    g.add_argument("--lstm-layers", type=int, choices=(1, 2), help="Number of LSTM layers (default 1).")
    g.add_argument("--lstm-units", type=int, help="LSTM hidden units (default 64).")
    g.add_argument("--lstm-dropout", type=float, help="LSTM dropout (default 0.2).")
    g.add_argument("--mlp-hidden", help="MLP hidden layers, comma separated (default 128,64).")
    g.add_argument("--mlp-dropout", type=float, help="MLP dropout (default 0.0).")


def config_from_args(args: argparse.Namespace) -> Config:
    cfg = Config.from_json(args.config) if getattr(args, "config", None) else Config()
    mapping = [
        ("target", cfg.data, "target"), ("train_frac", cfg.data, "train_frac"),
        ("val_frac", cfg.data, "val_frac"), ("max_gap_hours", cfg.data, "max_gap_hours"),
        ("input_hours", cfg.train, "input_hours"), ("horizon_hours", cfg.train, "horizon_hours"),
        ("epochs", cfg.train, "max_epochs"), ("batch_size", cfg.train, "batch_size"),
        ("learning_rate", cfg.train, "learning_rate"), ("patience", cfg.train, "early_stopping_patience"),
        ("seed", cfg.train, "seed"), ("lstm_layers", cfg.lstm, "num_layers"),
        ("lstm_units", cfg.lstm, "units"), ("lstm_dropout", cfg.lstm, "dropout"),
        ("mlp_dropout", cfg.mlp, "dropout"),
    ]
    for arg_name, section, attr in mapping:
        value = getattr(args, arg_name, None)
        if value is not None:
            setattr(section, attr, value)
    if getattr(args, "mlp_hidden", None):
        try:
            cfg.mlp.hidden_units = tuple(int(x) for x in args.mlp_hidden.split(","))
        except ValueError as exc:
            raise ConfigError("--mlp-hidden must look like 128,64") from exc
    return cfg.validate()
