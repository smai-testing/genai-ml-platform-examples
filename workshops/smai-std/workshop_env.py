"""Thin compatibility shim — the real code lives in ``workshop_common``.

Labs add the repo root to ``sys.path`` and do
``from workshop_env import load_workshop_env``. That continues to work: this
module simply re-exports the loader from the ``workshop_common`` package, which
is the single home for shared workshop code (env loader + dataset schema).

Prefer ``from workshop_common import load_workshop_env`` in new code.
"""
from workshop_common.env import find_env, load_workshop_env  # noqa: F401

__all__ = ["load_workshop_env", "find_env"]
