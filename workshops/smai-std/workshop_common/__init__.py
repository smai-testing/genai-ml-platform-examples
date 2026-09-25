"""workshop_common — the single shared package for all workshop labs.

Holds the values and schema that MORE THAN ONE lab needs, so nothing is
hardcoded or duplicated across notebooks:

* ``env``    — the shared ``.env`` loader (project name, prefixes, schema names)
* ``schema`` — the dataset feature schema (parsed from ``dataset_schema.yaml``)

Every lab adds the repo root to ``sys.path`` and imports from here. lab5's
``src`` package keeps its own lab5-specific code (drift monitoring, governance,
config.py) and reads the shared schema from this package.
"""
from workshop_common.env import load_workshop_env  # noqa: F401
