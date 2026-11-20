"""yache_engine package.

Importing ANY module here (TTS provider, STT engine, paths…) mirrors the
package's + the vendored engines' loggers into the single plugin-owned
log file (<plugin>/logs/yache-plugin.log) — the "open log" button must
reveal the whole plugin, synthesis internals included. logbox keeps to
stdlib basics and derives its path from __file__, so this stays cheap and
safe even when everything else is broken.
"""
from . import logbox as _logbox

_logbox.attach_engine_loggers()
