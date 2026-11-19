"""yache-local-voice — Yache Voice: one full local voice pipeline plugin.

Bundles the local voice engines into a single plugin:

- TTS   : Qwen3-TTS 1.7B (ggml ``qwen.dll``), zero-disk streaming PCM,
          fully offline, VRAM released after every playback. The TTS
          provider name stays ``qwen3-tts`` for host-config compatibility.
- STT   : local faster-whisper (CPU int8), provider ``local-stt``.
- Wake  : NOT owned by this plugin. The host's official built-in
          wake word engine (sherpa-onnx KWS, ``wake_word.provider:
          sherpa``) does the listening; the dashboard panel here only
          exposes its official settings. An earlier release shipped a
          plugin-side wake engine that wrote into core registries at
          runtime — that core override was retired in 0.2.0 (catalog
          rule 9); the earlier plugin-ized copy now rests outside the package.

All filesystem paths resolve through ``engine/paths.py``
(env -> data file -> platform default). A missing engine returns control
to the host with a clear message; nothing silently downloads or
substitutes another model.
"""
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:  # absolute-first: one shared module instance (same lesson as video-xray)
    from yache_engine import stt as _stt
    from yache_engine import tts as _tts
    from yache_engine import state as _state
except ImportError:  # package-context fallback (hermes_plugins.<slug>)
    from .yache_engine import stt as _stt
    from .yache_engine import tts as _tts
    from .yache_engine import state as _state


def _register_impl(ctx) -> None:
    """Register every provider, tool and command hosted by this plugin.

    Provider names are intentionally unchanged from the standalone
    ``qwen3-tts`` / ``local-stt`` plugins, so existing ``config.yaml``
    routing keeps working after the swap. Wake word detection is left
    entirely to the official built-in engine — no core registry writes.
    """
    _tts.register(ctx)   # provider "qwen3-tts" + doctor tool + commands
    _stt.register(ctx)   # provider "local-stt" + status tool + /stt

    # Visible to the web half (dashboard/api.py) via the shared
    # engine.state module — feeds /health and the doctor "plugin" lamp.
    _state.mark_registered(
        tts="qwen3-tts", stt="local-stt", wake="sherpa (official built-in)",
        version="0.1.0",
    )


def register(ctx) -> None:
    """Public entry: log any failure (the host loader swallows tracebacks)
    before re-raising, so the plugin dir keeps its own load diary."""
    try:
        from yache_engine import logbox as _lx
    except Exception:  # even the logger import may fail
        _lx = None
    try:
        _register_impl(ctx)
        if _lx:
            _lx.get_logger().info("agent register OK")
    except Exception:
        if _lx:
            _lx.get_logger().exception("agent register FAILED")
        raise
