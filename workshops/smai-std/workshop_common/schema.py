"""Dataset feature schema — shared across labs (part of workshop_common).

This is the ONE place labs read the dataset's column names, types, and roles
(identifier / timestamp / target / feature / auxiliary). lab2 and lab3 import
this instead of hardcoding a feature list; lab5's ``src`` code has its own
copy of the parser (``src/config/schema.py``) because that module must ship
standalone into SageMaker processing and Lambda containers with only
stdlib + PyYAML on the path.

Single YAML source: this module reads lab5-monitoring's canonical
``dataset_schema.yaml`` when it is present in the repo, and falls back to a
colocated copy otherwise (e.g. if the labs are shipped without lab5). Either
way there is one authoritative schema definition, never a re-typed feature
list in a notebook.

Depends only on the standard library + PyYAML.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent

# Prefer lab5's canonical schema file (the documented single source); fall back
# to a colocated copy so the labs still work if shipped without lab5.
_CANDIDATE_PATHS = [
    _REPO_ROOT / "lab5-monitoring" / "src" / "config" / "dataset_schema.yaml",
    _HERE / "dataset_schema.yaml",
]

ATHENA_TYPE_MAP: Dict[str, str] = {
    "double": "DOUBLE",
    "string": "STRING",
    "boolean": "BOOLEAN",
    "int": "INT",
    "timestamp": "TIMESTAMP",
}


@dataclass(frozen=True)
class Feature:
    name: str
    type: str

    @property
    def athena_type(self) -> str:
        try:
            return ATHENA_TYPE_MAP[self.type]
        except KeyError:
            raise ValueError(
                f"Unknown type '{self.type}' for column '{self.name}'. "
                f"Supported types: {sorted(ATHENA_TYPE_MAP)}"
            )


def _schema_path() -> Path:
    for p in _CANDIDATE_PATHS:
        if p.exists():
            return p
    raise FileNotFoundError(
        "dataset_schema.yaml not found in any of: "
        + ", ".join(str(p) for p in _CANDIDATE_PATHS)
    )


@functools.lru_cache(maxsize=1)
def _load() -> Dict[str, Any]:
    with open(_schema_path(), "r") as f:
        doc = yaml.safe_load(f) or {}
    dataset = doc.get("dataset")
    if not dataset:
        raise ValueError(
            f"dataset_schema.yaml at {_schema_path()} is missing the top-level "
            "'dataset:' key."
        )
    return dataset


def identifier_column() -> str:
    return _load()["identifier_column"]


def timestamp_column() -> str:
    return _load()["timestamp_column"]


def target_column() -> str:
    return _load()["target_column"]


def target_type() -> str:
    return _load().get("target_type", "boolean")


def features() -> List[Feature]:
    return [Feature(f["name"], f["type"]) for f in _load()["features"]]


def feature_names() -> List[str]:
    return [f.name for f in features()]


def auxiliary_columns() -> List[Feature]:
    return [Feature(c["name"], c["type"]) for c in _load().get("auxiliary_columns", [])]


def csv_column_order() -> List[str]:
    return (
        [identifier_column()]
        + feature_names()
        + [c.name for c in auxiliary_columns()]
        + [target_column()]
    )


if __name__ == "__main__":
    print(f"Schema file:       {_schema_path()}")
    print(f"Identifier column: {identifier_column()}")
    print(f"Target column:     {target_column()} ({target_type()})")
    print(f"Feature count:     {len(features())}")
    print(f"Feature names:     {feature_names()}")
