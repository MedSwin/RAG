from __future__ import annotations

import sys
from pathlib import Path

# The shared facet templates live at the repository root so the eval service
# (started from eval/) and pytest (started from the repo root) resolve the
# same file.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from facets import *  # noqa: F401,F403
