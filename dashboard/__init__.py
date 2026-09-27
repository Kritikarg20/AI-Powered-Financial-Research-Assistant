"""
Dashboard package.

Ensures ``src/`` is on ``sys.path`` so every page can ``import fin_ai...``
regardless of Streamlit's working directory.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PARENT_DIR = BASE_DIR.parent
SRC_DIR = PARENT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

__all__ = ["BASE_DIR", "PARENT_DIR", "SRC_DIR"]
