"""JSON <-> pandas helpers, used for the mock files and anywhere engine output must be stored as JSON.

Encoding rules (stable contract for P3):
* ``pd.Series`` with a date index  -> ``{"2021-01-01": 100.0, ...}`` (ISO date keys)
* ``pd.DataFrame``                 -> ``{"_type": "dataframe", "index": [...], "columns": [...], "data": [[...]]}``
  (index values are ISO strings when they are dates)
* numpy scalars / arrays and NaN are converted to plain JSON types (NaN -> null)

``load_mock(name)`` rebuilds real pandas objects, so the UI can call it exactly as it would call the engine.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MOCK_DIR = ROOT / "data" / "mocks"

_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}")

# keys whose dict value is a date -> float series in every engine return shape
_SERIES_KEYS = {"growth"}


def _key(k: Any) -> str:
    if isinstance(k, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(k).strftime("%Y-%m-%d")
    return str(k)


def _scalar(v: Any) -> Any:
    if isinstance(v, (np.floating, float)):
        v = float(v)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(v).isoformat()
    return v


def to_jsonable(obj: Any) -> Any:
    """Recursively convert engine output (dicts, Series, DataFrames, numpy) to JSON-safe python."""
    if isinstance(obj, pd.DataFrame):
        return {
            "_type": "dataframe",
            "index": [_key(i) for i in obj.index],
            "columns": [str(c) for c in obj.columns],
            "data": [[_scalar(v) for v in row] for row in obj.to_numpy().tolist()],
        }
    if isinstance(obj, pd.Series):
        return {_key(k): _scalar(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return [to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, dict):
        return {_key(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return _scalar(obj)


def from_jsonable(obj: Any, _key_name: str | None = None) -> Any:
    """Inverse of :func:`to_jsonable` (DataFrames and ``growth`` series are rebuilt)."""
    if isinstance(obj, dict):
        if obj.get("_type") == "dataframe":
            index = obj["index"]
            if index and all(isinstance(i, str) and _ISO_DATE.match(i) for i in index):
                index = pd.DatetimeIndex(pd.to_datetime(index))
            return pd.DataFrame(obj["data"], index=index, columns=obj["columns"])
        if _key_name in _SERIES_KEYS:
            s = pd.Series(obj, dtype="float64")
            s.index = pd.to_datetime(s.index)
            return s
        return {k: from_jsonable(v, k) for k, v in obj.items()}
    if isinstance(obj, list):
        return [from_jsonable(v) for v in obj]
    return obj


def dump_json(obj: Any, path: str | Path) -> None:
    Path(path).write_text(json.dumps(to_jsonable(obj), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def load_mock(name: str) -> Any:
    """Load ``data/mocks/mock_<name>.json`` (e.g. ``load_mock("metrics")``) as real engine-shaped objects."""
    path = MOCK_DIR / f"mock_{name}.json"
    return from_jsonable(json.loads(path.read_text(encoding="utf-8")))
