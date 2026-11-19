"""In-process registration state shared by BOTH halves of the plugin.

The agent half (``__init__.register``) records what got registered here;
the web half (``dashboard/api.py``) reads it for /health and /doctor.
Because both halves import this module under the SAME absolute key
``engine.state`` (video-xray single-instance lesson), one process-level
dict is enough. Stdlib only — the gateway web half may not have heavy
deps, so anything imported from api.py must stay stdlib-only.
"""
from __future__ import annotations

REGISTRATION = {
    "loaded": False,
    "tts": None,
    "stt": None,
    "wake": None,
    "version": "",  # overwritten by register() from plugin.yaml
}


def mark_registered(tts=None, stt=None, wake=None, version=None) -> None:
    if tts:
        REGISTRATION["tts"] = tts
    if stt:
        REGISTRATION["stt"] = stt
    if wake:
        REGISTRATION["wake"] = wake
    if version:
        REGISTRATION["version"] = version
    REGISTRATION["loaded"] = True


def snapshot() -> dict:
    return dict(REGISTRATION)
