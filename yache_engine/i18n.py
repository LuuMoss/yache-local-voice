"""Language routing for user-visible backend strings (Yache Voice plugin).

The desktop UI is bilingual, but HTTP ``detail`` / ``error`` strings raised by
the backend were previously hard-coded Chinese, so English-locale users saw
Chinese errors.  ``ui(zh, en)`` picks the variant that matches the host's own
display language.

Source of truth: ``display.language`` in config.yaml -- the very key the
desktop UI writes -- read with a tiny stdlib line-scan (no YAML dependency,
works in any spawn context).  ``YV_LANG=zh|en`` in the environment overrides,
for tests.  Both the language probe and the config lines are cached 60 s.
"""
from __future__ import annotations

import os
import time

try:                                            # package mode
    from . import paths
except (ImportError, SystemError):              # direct-script / bare-module mode
    try:
        import paths
    except ImportError:
        paths = None                            # _cfg_path() env fallback covers us

_LANG_CACHE = {"lang": None, "t": 0.0}
_CFG_CACHE = {"lines": None, "t": 0.0}


def _cfg_path() -> str:
    try:
        return str(paths.hermes_home() / "config.yaml")
    except Exception:                                   # pragma: no cover
        return os.path.join(
            os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes"),
            "config.yaml")


def _cfg_lines():
    """config.yaml lines, 60 s cache (the file is tiny; a line-scan beats
    pulling a YAML lib into the plugin)."""
    now = time.time()
    if _CFG_CACHE["lines"] is not None and now - _CFG_CACHE["t"] < 60:
        return _CFG_CACHE["lines"]
    try:
        with open(_cfg_path(), encoding="utf-8", errors="replace") as fh:
            lines = fh.read().splitlines()
    except OSError:
        lines = []
    _CFG_CACHE["lines"] = lines
    _CFG_CACHE["t"] = now
    return lines


def config_get(*keys):
    """Scalar string at a nested path of config.yaml, e.g.
    config_get("display", "language").  Stdlib indent-stack scan; list items
    and multi-line blocks are ignored.  None when absent."""
    want = [str(k) for k in keys]
    stack = []                                          # (indent, key)
    for ln in _cfg_lines():
        if not ln.strip() or ln.lstrip().startswith("#") or ":" not in ln:
            continue
        indent = len(ln) - len(ln.lstrip(" "))
        raw_key, _, raw_val = ln.strip().partition(":")
        key = raw_key.strip()
        if not key or key.startswith("-"):              # skip list items
            continue
        val = raw_val.split(" #")[0].strip().strip('"').strip("'")
        while stack and stack[-1][0] >= indent:
            stack.pop()
        stack.append((indent, key))
        if [k for _, k in stack] == want and val:
            return val
    return None


def system_language():
    """App display language: 'zh' | 'en' (default 'zh').

    Follows display.language in config.yaml regardless of which model or
    provider is loaded, so backend errors land in the same language the UI
    uses.  YV_LANG=zh|en overrides (for tests).  Cached 60 s.
    """
    now = time.time()
    if _LANG_CACHE["lang"] is not None and now - _LANG_CACHE["t"] < 60:
        return _LANG_CACHE["lang"]
    env = os.environ.get("YV_LANG", "").strip().lower()
    if env:
        lang = "en" if env.startswith("en") else "zh"
    else:
        val = (config_get("display", "language") or "zh").lower()
        lang = "en" if val.startswith("en") else "zh"
    _LANG_CACHE["lang"] = lang
    _LANG_CACHE["t"] = now
    return lang


def ui(zh, en):
    """Pick the user-facing string for the current display language."""
    return en if system_language() == "en" else zh
