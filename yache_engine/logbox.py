"""Plugin-owned diagnostics: every load error and every route crash is
logged to <plugin>/logs/yache-plugin.log, independent of the host log tree.
The host can fail *while importing* this plugin, so this module keeps to
stdlib basics and derives its own path from __file__ (no paths.py import —
a broken paths.py must not take the logger down with it).
"""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_NAME = "yache-local-voice"


def log_file() -> str:
    return str(_LOG_DIR / "yache-plugin.log")


def get_logger() -> logging.Logger:
    lg = logging.getLogger("yache." + _NAME)
    if getattr(lg, "_yache_handler", False):
        return lg
    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        h = RotatingFileHandler(_LOG_DIR / "yache-plugin.log",
                                maxBytes=400_000, backupCount=2, encoding="utf-8")
        h.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-7s [%(process)d] %(message)s"))
        lg.addHandler(h)
        lg.setLevel(logging.INFO)
        lg.propagate = False
        lg._yache_handler = True
    except OSError:
        pass
    return lg
