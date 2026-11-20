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


class _ShortNameFormatter(logging.Formatter):
    """One book for the WHOLE plugin: the dashboard (logger
    ``yache.yache-local-voice``), the engine modules (``yache_engine.*``)
    and the vendored engines (``faster_qwen3_tts.*`` / ``qwentts_cpp.*``)
    all land here, and the [name] tag tells the reader which module spoke.
    Raw namespaces are long; shorten the two loud regulars so the trail
    stays scannable."""

    def format(self, record):
        n = record.name
        if n == "yache." + _NAME:
            n = "panel"
        elif n.startswith("yache_engine"):
            n = n.replace("yache_engine", "engine", 1)
        record.name = n
        return super().format(record)


# Engine + vendored-engine namespaces: their WARNING-and-up chatter is the
# part owners report bugs about (synthesis/transcript internals), yet by
# default it propagates to the host root logger and never reaches the
# plugin log the "open log" button reveals. Mirroring them here closes the
# loop: ONE file, one button, whole plugin. Level stays WARNING — engine
# INFO chatter would flush the 400 KB rotating window that download
# trails need to survive a bug report.
_ENGINE_NS = ("yache_engine", "faster_qwen3_tts", "qwentts_cpp")


def attach_engine_loggers(level: int = logging.WARNING) -> None:
    lg = get_logger()
    if not lg.handlers:
        return                      # log dir was unwritable; nothing to share
    for name in _ENGINE_NS:
        e = logging.getLogger(name)
        if getattr(e, "_yache_mirror", False):
            continue                # idempotent: every entry point may call
        for h in lg.handlers:
            e.addHandler(h)
        e.setLevel(level)
        e.propagate = False         # stop double-writing into host agent.log
        e._yache_mirror = True


def get_logger() -> logging.Logger:
    lg = logging.getLogger("yache." + _NAME)
    if getattr(lg, "_yache_handler", False):
        return lg
    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        h = RotatingFileHandler(_LOG_DIR / "yache-plugin.log",
                                maxBytes=400_000, backupCount=2, encoding="utf-8")
        h.setFormatter(_ShortNameFormatter(
            "%(asctime)s %(levelname)-7s [%(process)d] [%(name)s] %(message)s"))
        lg.addHandler(h)
        lg.setLevel(logging.INFO)
        lg.propagate = False
        lg._yache_handler = True
    except OSError:
        pass
    return lg
