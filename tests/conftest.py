"""Pytest configuration and path setup for Surveillance tests."""
import sys
from pathlib import Path

# Add project root and scripts to path so we can import scripts
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
if str(_project_root / "scripts") not in sys.path:
    sys.path.insert(0, str(_project_root / "scripts"))
