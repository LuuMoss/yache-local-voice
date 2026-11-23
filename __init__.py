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


def _heal_tts_pointer() -> None:
    """Fresh-install self-heal (owner, 2026-11-20).

    The host's read-aloud path routes through ``tts.provider``; on a fresh
    install NOTHING sets that key, so the TTS this plugin registers stays
    "ready but unused" and read-aloud silently falls back to the built-in
    edge engine — the panel then shows 已就绪·未生效 and only the tail
    punctuation gets spoken. Unlike the wake-word bridge there is no
    "save" moment to carry this, so it heals once at plugin registration
    (gateway start): write the pointer ONLY when the key is ABSENT. An
    explicit choice the user made (edge, groq, any other provider) is
    never overwritten. Idempotent: after the first write every later
    start passes straight through.
    """
    try:
        from hermes_cli.config import load_config, save_config
        cfg = load_config() or {}
        tts = cfg.get("tts")
        if isinstance(tts, dict) and "provider" in tts:
            return
        if not isinstance(tts, dict):
            tts = {}
        tts["provider"] = "qwen3-tts"
        cfg["tts"] = tts
        save_config(cfg)
        try:
            from yache_engine import logbox as _lx
            _lx.get_logger().info(
                "tts.provider was unset -> pointed at qwen3-tts (read-aloud route)")
        except Exception:
            pass
    except Exception:
        # never fail registration over a config hiccup; the panel's doctor
        # lamp still tells the user the route is not pointing here
        pass


def _register_impl(ctx) -> None:
    """Register every provider, tool and command hosted by this plugin.

    Provider names are intentionally unchanged from the standalone
    ``qwen3-tts`` / ``local-stt`` plugins, so existing ``config.yaml``
    routing keeps working after the swap. Wake word detection is left
    entirely to the official built-in engine — no core registry writes.
    """
    _tts.register(ctx)   # provider "qwen3-tts" + doctor tool + commands
    _stt.register(ctx)   # provider "local-stt" + status tool + /stt
    _heal_tts_pointer()  # fresh-install route pointer (key absent -> ours)

    # Visible to the web half (dashboard/api.py) via the shared
    # engine.state module — feeds /health and the doctor "plugin" lamp.
    _state.mark_registered(
        tts="qwen3-tts", stt="local-stt", wake="sherpa (official built-in)",
        version="0.1.3",
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
