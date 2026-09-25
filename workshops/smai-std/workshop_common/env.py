"""Shared ``.env`` loader for every workshop lab (part of workshop_common).

lab0-setup/setup.ipynb discovers the AWS environment ONCE and writes a single
repo-root ``.env``. Every lab imports ``load_workshop_env`` from here (via the
``workshop_common`` package, or the thin root ``workshop_env`` re-export) so the
loader logic lives in exactly one place.

It:
  * finds the repo-root ``.env`` by walking up from the caller's CWD,
  * loads it into ``os.environ`` (does not override already-set vars),
  * resolves the canonical workshop variables with fail-safe defaults.

The env-var names and defaults intentionally match lab5-monitoring's
``src/config/config.py`` so all labs share one contract.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

try:
    from dotenv import load_dotenv
except Exception:  # python-dotenv not installed — use a minimal parser.
    def load_dotenv(path, override: bool = False):  # type: ignore
        try:
            for line in Path(path).read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if override or k not in os.environ:
                    os.environ[k] = v
            return True
        except Exception:
            return False

# Canonical defaults — used only as a fail-safe when .env / environment is
# missing the value. lab0-setup writes the real (discovered) values into .env.
_DEFAULTS: Dict[str, str] = {
    "PROJECT_NAME": "bank-marketing-prediction",
    "ATHENA_DATABASE": "bank_marketing",
    "S3T_NAMESPACE": "bank_marketing",
    "ATHENA_TRAINING_TABLE": "training_data",
    "ATHENA_EVALUATION_TABLE": "evaluation_data",
    # Workshop convention: S3 prefix where lab2 writes prepared train/test CSVs
    # and lab3 reads them back. Not a deployed resource — a fixed path segment.
    "DATA_PREFIX": "bank-marketing-lab",
}


def find_env(start: Path | None = None) -> Path | None:
    """Walk up from ``start`` (default CWD) to find the repo-root ``.env``."""
    start = (start or Path.cwd()).resolve()
    for cand in [start, *start.parents]:
        env = cand / ".env"
        if env.exists():
            return env
    return None


def load_workshop_env(require: bool = True) -> Dict[str, str]:
    """Load the shared ``.env`` and return the resolved workshop variables.

    Args:
        require: when True (default) raise if no ``.env`` is found — the labs
            depend on lab0-setup having written it. Set False for a soft load
            that falls back to defaults.

    Returns:
        Dict of the canonical variables (PROJECT_NAME, ATHENA_DATABASE,
        S3T_NAMESPACE, TRAINING_TABLE, EVALUATION_TABLE, DATA_PREFIX) resolved
        from the environment with canonical fallbacks.
    """
    env_path = find_env()
    if env_path is None:
        if require:
            raise RuntimeError(
                "No .env found. Run lab0-setup/setup.ipynb first to discover "
                "the AWS environment and write the shared .env."
            )
    else:
        # override=False: an explicitly-exported var (e.g. a custom deploy)
        # wins over the file, matching config.py precedence.
        load_dotenv(env_path, override=False)

    resolved = {
        "PROJECT_NAME": os.environ.get("PROJECT_NAME", _DEFAULTS["PROJECT_NAME"]),
        "ATHENA_DATABASE": os.environ.get("ATHENA_DATABASE", _DEFAULTS["ATHENA_DATABASE"]),
        "S3T_NAMESPACE": os.environ.get("S3T_NAMESPACE", _DEFAULTS["S3T_NAMESPACE"]),
        # Notebook-facing names map to the ATHENA_* env keys config.py uses.
        "TRAINING_TABLE": os.environ.get("ATHENA_TRAINING_TABLE", _DEFAULTS["ATHENA_TRAINING_TABLE"]),
        "EVALUATION_TABLE": os.environ.get("ATHENA_EVALUATION_TABLE", _DEFAULTS["ATHENA_EVALUATION_TABLE"]),
        "DATA_PREFIX": os.environ.get("DATA_PREFIX", _DEFAULTS["DATA_PREFIX"]),
    }
    return resolved
