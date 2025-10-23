#!/usr/bin/env python3
"""Wrapper script that exposes the asahi_ane_llm conversion helper."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running the script directly from the repository without installing the
# helper package in the current environment.
REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_SRC = REPO_ROOT / "libs" / "asahi_ane_llm" / "src"
if str(PKG_SRC) not in sys.path:
    sys.path.insert(0, str(PKG_SRC))

try:
    from asahi_ane_llm.tools.convert_to_ane import main
except ModuleNotFoundError as exc:  # pragma: no cover - user environment guard
    if exc.name == "onnx":
        raise SystemExit(
            "Missing dependency: install the 'onnx' package or run 'pip install -e "
            "libs/asahi_ane_llm' before using the conversion tool."
        )
    raise


if __name__ == "__main__":
    raise SystemExit(main())
