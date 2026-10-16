"""Save / load trained model, preprocessing artifacts, config, history and metadata."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .config import ARTIFACTS_DIR
from .errors import ArtifactError
from .preprocessing import Preprocessor

FILES = {"model": "model.keras", "preprocessor": "preprocessor.joblib", "config": "config.json",
         "history": "history.json", "metadata": "metadata.json"}


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if np.isnan(v) or np.isinf(v) else v
    if isinstance(obj, float):
        return None if np.isnan(obj) or np.isinf(obj) else obj
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    return obj


def artifact_dir(model_name: str, root=None) -> Path:
    return Path(root or ARTIFACTS_DIR) / model_name


def save_keras_model(model, path: Path) -> None:       # separate function => easy to stub in tests
    model.save(str(path))


def load_keras_model(path: Path):
    from .models import require_keras
    _, keras = require_keras()
    return keras.models.load_model(str(path))


@dataclass
class LoadedModel:
    name: str
    model: Any
    preprocessor: Preprocessor
    config: Dict[str, Any]
    metadata: Dict[str, Any]
    history: Dict[str, Any]


def save_artifacts(model_name: str, model, preprocessor: Preprocessor, config_dict: Dict[str, Any],
                   training: Dict[str, Any], metadata: Dict[str, Any], root=None) -> Path:
    d = artifact_dir(model_name, root)
    d.mkdir(parents=True, exist_ok=True)
    save_keras_model(model, d / FILES["model"])
    preprocessor.save(d / FILES["preprocessor"])
    (d / FILES["config"]).write_text(json.dumps(to_jsonable(config_dict), indent=2), encoding="utf-8")
    (d / FILES["history"]).write_text(json.dumps(to_jsonable(training), indent=2), encoding="utf-8")
    (d / FILES["metadata"]).write_text(json.dumps(to_jsonable(metadata), indent=2), encoding="utf-8")
    return d


def load_artifacts(model_name: str, root=None) -> LoadedModel:
    d = artifact_dir(model_name, root)
    missing = [f for f in FILES.values() if not (d / f).exists()]
    if missing:
        raise ArtifactError(
            f"Trained artifacts for '{model_name}' not found or incomplete in {d} (missing: {missing}).\n"
            f"Train first, e.g.:  python -m src.train --model {model_name} --csv data\\your_file.csv")
    try:
        return LoadedModel(
            name=model_name,
            model=load_keras_model(d / FILES["model"]),
            preprocessor=Preprocessor.load(d / FILES["preprocessor"]),
            config=json.loads((d / FILES["config"]).read_text(encoding="utf-8")),
            metadata=json.loads((d / FILES["metadata"]).read_text(encoding="utf-8")),
            history=json.loads((d / FILES["history"]).read_text(encoding="utf-8")),
        )
    except (json.JSONDecodeError, OSError) as exc:
        raise ArtifactError(f"Artifacts in {d} are corrupt: {exc}") from exc
