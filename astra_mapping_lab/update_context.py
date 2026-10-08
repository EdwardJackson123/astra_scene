"""Server-owned per-job paths for map update subprocesses."""
import json, os
from pathlib import Path


def settings():
    return json.loads(Path(os.environ['ASTRA_UPDATE_CONTEXT']).read_text())
