"""Resolve the companion ``unifying_attention`` package without user-specific paths."""
from __future__ import annotations

import os
from pathlib import Path
import sys


def configure_unifying_attention() -> Path:
    configured = os.environ.get("MECH_INTERP_ROOT")
    roots = []
    if configured:
        roots.append(Path(configured).expanduser())
    here = Path(__file__).resolve()
    roots.extend((Path.cwd(), *here.parents))
    for root in roots:
        package_root = root / "unifying_algorithm"
        if (package_root / "unifying_attention").is_dir():
            for path in (root, package_root):
                if str(path) not in sys.path:
                    sys.path.insert(0, str(path))
            return root
    raise ImportError(
        "Could not locate unifying_algorithm/unifying_attention. Run from the "
        "Mech-Interp checkout or set MECH_INTERP_ROOT to that checkout."
    )

