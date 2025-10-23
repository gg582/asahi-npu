#!/usr/bin/env python3
"""Wrapper around the ANE microcode assembler helper."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_SRC = REPO_ROOT / "libs" / "asahi_ane_llm" / "src"
if str(PKG_SRC) not in sys.path:
    sys.path.insert(0, str(PKG_SRC))

from asahi_ane_llm.tools.build_microcode import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
