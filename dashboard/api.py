"""Yache Voice — dashboard API for the desktop plugin panel (M1).

Mount contract (web_server_dashboard.py): the host imports this file as a
standalone module and include_router()s a module-level ``router`` under
/api/plugins/yache-local-voice/ with plugin auth.

Single-instance discipline (video-xray ctx-disconnect lesson): we insert the
plugin root on sys.path and import the engine package by ABSOLUTE name first,
so the web half and the agent half share one ``engine.state`` module object
in the same gateway process.

Everything imported at module level is stdlib + fastapi only — the gateway
python has no numpy/torch; heavy synthesis happens in a short-lived worker
subprocess using the engine venv python.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import wave
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

_PLUGIN_ROOT = Path(__file__).resolve().parents[1]
if str(_PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_ROOT))

from yache_engine import logbox as _lx  # noqa: E402  (plugin-owned log)
_log = _lx.get_logger()
from yache_engine import paths as _p  # noqa: E402  (stdlib only)
from yache_engine import state as _st  # noqa: E402  (shared with agent half)
from yache_engine import worker as _w  # noqa: E402  (stdlib only; cuda_free_mib)
from yache_engine import yache_voice as _vy  # noqa: E402  (GGUF scanner)
from yache_engine import i18n as _i18n  # noqa: E402  (bilingual user-facing strings)

# Download wizard (dashboard/download.py): the model + GPU-accelerator state
# machine. Every route below is a thin wrapper over it; if the module ever
# fails to import the panel degrades to the manual directory-load path.
try:
    _DASH_DIR = str(Path(__file__).resolve().parent)
    if _DASH_DIR not in sys.path:
        sys.path.insert(0, _DASH_DIR)
    import download as _dl  # noqa: E402
except Exception:  # pragma: no cover
    _dl = None  # type: ignore

router = APIRouter()

_CREATE_NO_WINDOW = 0x08000000  # GUI host: nvidia-smi/stdout dies without it
_SAMPLE_RATE = 24000


# ── helpers ──────────────────────────────────────────────────────────────

def _has_nvidia() -> bool:
    """Cheap probe; cuda_free_mib() itself returns None without nvidia-smi."""
    return _w.cuda_free_mib() is not None


def _voices_summary() -> Dict[str, Any]:
    reg = _p.voices_registry_path()
    default = None
    count = 0
    try:
        data = json.loads(reg.read_text(encoding="utf-8"))
        voices = data.get("voices") or data or {}
        if isinstance(voices, dict):
            count = len(voices)
            default = data.get("default") if isinstance(data, dict) else None
    except Exception:
        pass
    return {"path": str(reg), "exists": reg.is_file(), "count": count,
            "default": default}


# ── routes ───────────────────────────────────────────────────────────────

@router.get("/health")
def health() -> Dict[str, Any]:
    """Liveness + which providers this plugin registered (agent half)."""
    return {"ok": True, "plugin": "yache-local-voice",
            "registration": _st.snapshot()}


@router.post("/debug")
def ui_debug(payload: Optional[Dict[str, Any]] = None) -> Dict[str, bool]:
    """Frontend log pipe (mirrors video-xray's /debug, adopted 2026-11-18):
    the panel POSTs breadcrumbs and swallowed JS errors so a future
    "failed to render" leaves more than a white error page behind. Capped
    and never raises: logging is diagnostics, not function."""
    try:
        msg = (payload or {}).get("msg")
        _log.info("[panel] %s", "" if msg is None else str(msg)[:4000])
    except Exception:  # noqa: BLE001
        pass
    return {"ok": True}


@router.get("/status")
def status() -> Dict[str, Any]:
    free = _w.cuda_free_mib()
    rec = _p.recommend_quant(free, _has_nvidia())
    return {
        "free_vram_mib": free,
        "has_nvidia": free is not None,
        "installed_tiers": _p.installed_tiers(),
        "recommended": rec,
        "selected": _p.selected_quant(),
        "model_base_dir": str(_p.tts_base_dir()),
        "voices": _voices_summary(),
        "registration": _st.snapshot(),
        # Experimental playback wake-gate (default off = stock behavior)
        "wake_gate": _p.wake_gate_enabled(),
        "sample_rate": _SAMPLE_RATE,
    }


@router.get("/doctor")
def doctor() -> Dict[str, Any]:
    """Six lamps: plugin / engine / models / vram / voice lib / worker.

    Lamp *labels* are localized on the frontend by id; ``detail`` here is
    machine facts (paths, sizes) which stay language-neutral.
    """
    lamps = []
    snap = _st.snapshot()

    lamps.append({"id": "plugin", "ok": bool(snap["loaded"]),
                  "detail": json.dumps(snap, ensure_ascii=False)})

    py = _p.engine_python()
    dll = _p.engine_site_packages() / "qwentts_cpp" / "lib" / "qwen.dll"
    lamps.append({"id": "engine", "ok": py.is_file(),
                  "detail": f"python={py} qwen.dll={dll if dll.is_file() else 'MISSING'}"})

    tiers = _p.installed_tiers()
    pair = ", ".join(tiers) if tiers else "no gguf pair found"
    lamps.append({"id": "models", "ok": bool(tiers),
                  "detail": f"{_p.tts_base_dir()} :: {pair}"})

    free = _w.cuda_free_mib()
    if free is None:
        lamps.append({"id": "vram", "ok": True,
                      "detail": "no nvidia-smi — CPU mode assumed"})
    else:
        rec = _p.recommend_quant(free, True)
        lamps.append({"id": "vram", "ok": rec["tier"] != "cpu" or free >= 1200,
                      "detail": f"free {free} MiB — {rec['reason']}"})

    voices = _voices_summary()
    lamps.append({"id": "voice", "ok": bool(voices["exists"] and voices["count"]),
                  "detail": f"{voices['count']} voices, default={voices['default']}"})

    ws = _p.worker_script()
    lamps.append({"id": "worker", "ok": ws.is_file(), "detail": str(ws)})

    return {"ok": all(l["ok"] for l in lamps), "lamps": lamps}


# ---------------------------------------------------------------------------
# Activation gate — which modules the host actually routes right now
# ---------------------------------------------------------------------------
# The plugin can be fully installed and healthy while the host is pointed
# at another voice provider (a cloud TTS/STT picked in the settings page,
# for instance).  In that state our engines are ready but dormant: zero
# RAM/VRAM residency, and the official provider entry point stays fully
# in charge — we never block it.  The panel must be honest about this:
#   serving   the host routes through this plugin and it is healthy
#   ready     healthy, but the host currently uses another provider
#   notready  prerequisites are missing (see ``detail``)
# These routes read the same official keys the host itself reads
# (``tts.provider`` / ``stt.provider`` / ``wake_word.provider``) and the
# POST writes the local provider back through the official config writer
# — the same normalize/merge path the settings page uses.

_LOCAL_TTS_PROVIDER = "qwen3-tts"
_LOCAL_STT_PROVIDER = "local-stt"


def _wake_provider() -> str:
    """Host wake-word provider (stock default = built-in sherpa)."""
    try:
        from hermes_cli.config import read_raw_config

        raw = read_raw_config() or {}
        wake = raw.get("wake_word")
        prov = wake.get("provider") if isinstance(wake, dict) else None
    except Exception:  # noqa: BLE001
        prov = None
    return str(prov or "sherpa").strip().lower()


def _stt_readiness():
    """(ready, detail) for the speech-to-text light, decided by the one
    interpreter the plugin would actually transcribe with."""
    try:
        from yache_engine.stt import LocalSTTProvider

        if LocalSTTProvider().is_available():
            return True, ""
    except Exception as exc:
        return False, str(exc)[:200]
    return False, ("local speech-to-text engine is not loading; open "
                   "speech-to-text settings and run the repair")


def _activation_state() -> Dict[str, Any]:
    """Per-module live view; no subprocesses, no model loads."""
    try:
        from hermes_cli.config import read_raw_config

        raw = read_raw_config() or {}
    except Exception:  # noqa: BLE001
        raw = {}
    tts_sec = raw.get("tts")
    tts_prov = (str(tts_sec.get("provider") or "").strip()
                if isinstance(tts_sec, dict) else "")
    stt_sec = raw.get("stt")
    stt_prov = (str(stt_sec.get("provider") or "").strip()
                if isinstance(stt_sec, dict) else "")
    wake_prov = _wake_provider()

    snap = _st.snapshot()
    lamps = doctor()["lamps"]          # same payload the panel already renders
    lamp = {l["id"]: l["ok"] for l in lamps}
    voices = _voices_summary()

    tts_ready = bool(lamp.get("engine") and lamp.get("models"))
    wake_ready = (wake_prov == "sherpa")
    # Readiness comes from the plugin's own probe, not from the host's
    # engine report: that report looks for faster-whisper in the gateway's
    # own Python environment, and the engine deliberately lives outside it
    # now (an official upgrade rebuilds that environment and wipes the
    # dependency, which used to leave this light yellow while real
    # transcription kept working).  The probe is TTL-cached, so polling
    # this route stays cheap: 120 s on success, 20 s after a failure.
    stt_ready, stt_detail = _stt_readiness()
    clone_ready = bool(voices["exists"] and voices["count"])

    def pack(ready: bool, active: bool, detail: str) -> Dict[str, Any]:
        state = "notready" if not ready else ("serving" if active else "ready")
        return {"ready": ready, "active": active, "state": state,
                "detail": detail}

    mods = {
        "tts": pack(tts_ready, tts_prov == _LOCAL_TTS_PROVIDER,
                    _i18n.ui("雅澈本地合成引擎", "Yache local synthesis engine")),
        "wake": pack(wake_ready, wake_ready,
                     _i18n.ui("官方内置唤醒引擎 (sherpa)",
                              "official built-in wake engine (sherpa)")),
        "stt": pack(stt_ready, stt_prov == _LOCAL_STT_PROVIDER,
                    stt_detail or _i18n.ui("本地 faster-whisper 转录引擎",
                                           "local faster-whisper transcription engine")),
        "clone": pack(clone_ready, tts_prov == _LOCAL_TTS_PROVIDER,
                      _i18n.ui("音色克隆库（跟着合成 provider 走）",
                               "voice-clone library (follows the TTS provider)")),
    }
    if not snap["loaded"]:
        overall = "notready"
    elif all(mods[k]["state"] == "serving" for k in ("tts", "wake", "stt")):
        overall = "serving"
    else:
        overall = "ready"
    return {"plugin_loaded": bool(snap["loaded"]),
            "state": overall,
            "providers": {"tts": tts_prov or "(stock default)",
                          "stt": stt_prov or "(stock default)",
                          "wake": wake_prov},
            "local": {"tts": _LOCAL_TTS_PROVIDER,
                      "stt": _LOCAL_STT_PROVIDER},
            "modules": mods}


@router.get("/activation")
def activation_get() -> Dict[str, Any]:
    """Which modules the host routes through right now (cheap; pollable)."""
    return _activation_state()


class _ActivationIn(BaseModel):
    module: str = "tts"           # tts | stt (wake has no switch; clone follows tts)
    probe: bool = True            # verify the local engine really works first


@router.post("/activation")
def activation_switch(payload: _ActivationIn) -> Dict[str, Any]:
    """Switch one module back to this plugin's local provider.

    A provider key is a single scalar: switching back is one write through
    the official config writer (same atomic path as the stop-phrase card),
    effective for the very next synthesis / transcription — no restart.
    The probe defaults ON: the local engine must really answer before the
    provider flips, so a cold machine never wakes up silent after the
    click.  On probe failure nothing was written and the fresh activation
    state comes back so the panel can re-sync its lights.
    """
    module = str(payload.module or "").strip().lower()
    if module not in ("tts", "stt"):
        raise HTTPException(
            status_code=400,
            detail=_i18n.ui("可切换的模块只有 合成 和 转录",
                            "Only the synthesis and transcription modules can be switched"))
    probe = bool(payload.probe)
    probe_ms = None
    if probe:
        if module == "tts":
            try:
                out = test_speak(_SpeakIn(text=_i18n.ui(
                    "雅澈本地合成引擎已接管，随时为您出声。",
                    "Yache local engine is in charge and ready to speak."),
                    timeout_s=180))
                probe_ms = out.get("ms")
            except HTTPException as exc:
                return {"ok": False,
                        "error": _i18n.ui("本地合成引擎验证失败，未切换: %s",
                                          "Local synthesis engine failed verification; nothing switched: %s")
                                 % exc.detail,
                        "activation": _activation_state()}
            except Exception as exc:  # noqa: BLE001
                return {"ok": False,
                        "error": _i18n.ui("本地合成引擎验证失败，未切换: %s: %s",
                                          "Local synthesis engine failed verification; nothing switched: %s: %s")
                                 % (type(exc).__name__, exc),
                        "activation": _activation_state()}
        else:
            try:
                from hermes_cli.config import read_raw_config

                raw = read_raw_config() or {}
                stt_cfg = raw.get("stt")
                local_cfg = (stt_cfg.get("local")
                             if isinstance(stt_cfg, dict) and isinstance(stt_cfg.get("local"), dict)
                             else {})
                model = str(local_cfg.get("model") or "").strip()
                if not model:
                    data = _p.load_data()
                    block = data.get("stt") if isinstance(data.get("stt"), dict) else {}
                    model = str(block.get("model") or "").strip()
                if not model:
                    raise RuntimeError(_i18n.ui(
                        "还没有保存过转录模型，先在转录设置里选一个",
                        "No transcription model saved yet — pick one in the transcription settings first"))
                data = _p.load_data()
                block = data.get("stt") if isinstance(data.get("stt"), dict) else {}
                device = str(block.get("device") or "auto")
                from yache_engine.stt import LocalSTTProvider
                if not LocalSTTProvider().is_available():
                    raise RuntimeError(_i18n.ui(
                        "本地转录引擎探测失败（worker 解释器无法加载 faster-whisper，"
                        "或本机离线且模型未缓存）",
                        "local transcription probe failed (the worker interpreter cannot "
                        "load faster-whisper, or this machine is offline with no cached model)"))
            except HTTPException as exc:
                return {"ok": False, "error": exc.detail,
                        "activation": _activation_state()}
            except Exception as exc:  # noqa: BLE001
                return {"ok": False,
                        "error": _i18n.ui("本地转录引擎验证失败，未切换: %s: %s",
                                          "Local transcription engine failed verification; nothing switched: %s: %s")
                                 % (type(exc).__name__, exc),
                        "activation": _activation_state()}
    local = _LOCAL_TTS_PROVIDER if module == "tts" else _LOCAL_STT_PROVIDER
    try:
        from hermes_cli.config import load_config, save_config

        cfg = load_config()
        if not isinstance(cfg, dict):
            cfg = {}
        section = cfg.get(module)
        if not isinstance(section, dict):
            section = {}
            cfg[module] = section
        section["provider"] = local
        save_config(cfg)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500,
            detail=_i18n.ui("写入官方配置失败: %s: %s",
                            "Failed to write the official config: %s: %s")
                   % (type(exc).__name__, exc))
    return {"ok": True, "module": module, "provider": local,
            "probe_ms": probe_ms, "activation": _activation_state()}


class _SpeakIn(BaseModel):
    text: str = _i18n.ui("你好，这是一段语音合成试听，听听这个声音怎么样。",
                         "Hello! This is Yache Voice reading a preview for you.")
    timeout_s: int = 120


@router.post("/test/speak")
def test_speak(payload: _SpeakIn) -> Dict[str, Any]:
    """Audition: spawn a short-lived worker (--server), synthesize once,
    return wav as base64 (zero-disk: the panel plays a data: URI).

    Uses the registry default voice; first call pays model load (~5s CUDA).
    """
    text = (payload.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="empty text")
    timeout_s = max(10, min(int(payload.timeout_s or 120), 300))

    py = _p.engine_python()
    if not py.is_file():
        raise HTTPException(status_code=503, detail=f"engine python missing: {py}")

    env = dict(os.environ)
    env.setdefault("HERMES_VOICES_DIR", str(_p.voices_registry_path().parent))

    try:
        proc = subprocess.Popen(
            [str(py), str(_p.worker_script()), "--server"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding="utf-8",
            errors="replace", env=env, cwd=str(_p.worker_script().parent),
            creationflags=_CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"spawn failed: {exc}")

    chunks = []
    err_lines = []
    t0 = time.time()
    try:
        # wait READY (model preloaded)
        ready = False
        while time.time() - t0 < min(timeout_s, 90):
            line = proc.stdout.readline()
            if not line:
                break
            if line.strip() == "READY":
                ready = True
                break
        if not ready:
            raise HTTPException(
                status_code=502,
                detail="worker never reached READY (model load failed?)")

        proc.stdin.write(json.dumps({"text": text, "id": 1},
                                    ensure_ascii=False) + "\n")
        proc.stdin.flush()

        while time.time() - t0 < timeout_s:
            line = proc.stdout.readline()
            if not line:
                break
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = obj.get("type")
            if t == "pcm" and obj.get("id") == 1:
                chunks.append(obj.get("data", ""))
            elif t == "done" and obj.get("id") == 1:
                break
            elif t == "error":
                raise HTTPException(status_code=502,
                                    detail=f"worker error: {obj.get('msg')}")
        else:
            raise HTTPException(status_code=504,
                                detail=f"synthesis timed out after {timeout_s}s")
    finally:
        try:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
        except Exception:
            pass
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        try:
            if proc.stderr:
                err_lines = proc.stderr.read()[-800:].splitlines()
                proc.stderr.close()
        except Exception:
            pass

    pcm = b"".join(base64.b64decode(c) for c in chunks if c)
    if not pcm:
        raise HTTPException(
            status_code=502,
            detail="no audio produced" +
                   (f" | stderr: {err_lines[-1]}" if err_lines else ""))

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(_SAMPLE_RATE)
        wf.writeframes(pcm)
    wav_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return {
        "ok": True,
        "audio_b64": wav_b64,
        "mime": "audio/wav",
        "seconds": round(len(pcm) / (_SAMPLE_RATE * 2), 2),
        "ms": round((time.time() - t0) * 1000),
        "registration": _st.snapshot(),
    }


# ---------------------------------------------------------------------------
# Voice-conversation stop phrases
# ---------------------------------------------------------------------------
# The stock desktop ships a hardcoded default phrase table with no
# settings UI for it, so a newcomer would otherwise have to hand-edit
# config.yaml. These routes are that missing entry point: saving writes
# ``voice.stop_phrases`` through the official config writer (same
# normalize/merge path as the settings page itself), and the plugin panel
# then invalidates the frontend ``['hermes-config-record']`` query so the
# host re-reads the table right away — hot-apply, no restart. Matching
# semantics stay stock: a phrase ends the conversation only when the
# whole transcript equals it verbatim, so a casual mention inside a
# longer sentence never fires it.

_STOP_PHRASE_MAX_LEN = 40


class _StopPhraseIn(BaseModel):
    phrase: str = ""


def _stop_phrases_current() -> Dict[str, Any]:
    """Current table straight from config.yaml (the single source of
    truth), plus the stock defaults so the panel can label the
    un-customized state instead of hardcoding them."""
    try:
        from hermes_cli.config import read_raw_config

        raw = read_raw_config() or {}
    except Exception:
        raw = {}
    voice = raw.get("voice")
    phrases = voice.get("stop_phrases") if isinstance(voice, dict) else None
    # Stock default shipped here as a static label so the panel can show
    # the un-customized state without importing a private core module
    # (single-source truth lives in the voice transcript pipeline; if it
    # ever changes, this cosmetic label is updated to match).
    defaults: list[str] = ["stop"]
    return {
        "configured": isinstance(phrases, list) and bool(phrases),
        "phrases": [str(p) for p in phrases] if isinstance(phrases, list) else [],
        "default_phrases": defaults,
    }


@router.get("/stop-phrase")
def stop_phrase_get() -> Dict[str, Any]:
    return _stop_phrases_current()


@router.post("/stop-phrase")
def stop_phrase_set(body: _StopPhraseIn) -> Dict[str, Any]:
    """Store the table as ``voice.stop_phrases``. One input box; several
    phrases allowed, separated by comma/semicolon (ASCII or fullwidth)."""
    import re

    parts = [
        p.strip()
        for p in re.split(r"[,，;；\n]", body.phrase or "")
        if p.strip()
    ]
    if not parts:
        raise HTTPException(
            status_code=400, detail="stop phrase must not be empty"
        )
    for p in parts:
        if len(p) > _STOP_PHRASE_MAX_LEN:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"each stop phrase must be at most "
                    f"{_STOP_PHRASE_MAX_LEN} characters"
                ),
            )
        if any(ord(c) < 32 for c in p):
            raise HTTPException(
                status_code=400, detail="control character in stop phrase"
            )
    try:
        from hermes_cli.config import load_config, save_config

        cfg = load_config()
        voice = cfg.get("voice")
        if not isinstance(voice, dict):
            voice = {}
        voice["stop_phrases"] = parts
        cfg["voice"] = voice
        save_config(cfg)
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"config write failed: {exc}"
        )
    return {"ok": True, "phrases": parts, **_stop_phrases_current()}


# ---------------------------------------------------------------------------
# Wake-up conversation mode (official key wake_word.start_new_session)
# ---------------------------------------------------------------------------
# ``start_new_session`` is an upstream first-class setting (the wake event
# payload carries it; desktop wiring and the CLI mixin both honour it) that
# no settings page ever exposes. The stock default (true) opens a fresh chat
# window on every wake — the Hey-Siri model. Setting it to false keeps one
# conversation: wake -> talk -> stop phrase pauses -> next wake resumes the
# same window with the history intact. We only surface the switch; reads and
# writes go through the official config writer and semantics stay stock.
# One honest caveat, straight from the host code (tui_gateway/methods_voice
# freezes this value when it ARMS the wake listener): a saved change applies
# at the next (re)arming, not instantly mid-listen — the panel copy says
# exactly that instead of promising a hot switch.
#
# hermes_cli.config is on the official public module whitelist (_PUBLIC in
# plugin_validate_private_imports), so this stays sanctioned ground.

class _WakeSessionIn(BaseModel):
    start_new_session: bool


def _wake_session_current() -> Dict[str, Any]:
    """Read the mode straight from config.yaml. The key is optional
    upstream; absence means the stock default (true = new session)."""
    try:
        from hermes_cli.config import read_raw_config

        raw = read_raw_config() or {}
    except Exception:
        raw = {}
    wake = raw.get("wake_word")
    if not isinstance(wake, dict):
        wake = {}
    return {
        "configured": "start_new_session" in wake,
        "start_new_session": bool(wake.get("start_new_session", True)),
        "stock_default": True,
    }


@router.get("/wake-session")
def wake_session_get() -> Dict[str, Any]:
    """Current wake-up conversation mode."""
    return _wake_session_current()


@router.post("/wake-session")
def wake_session_set(body: _WakeSessionIn) -> Dict[str, Any]:
    """Persist ``wake_word.start_new_session`` through the official config
    writer (the same normalize/merge path the settings page itself uses;
    every other config section is left untouched)."""
    try:
        from hermes_cli.config import load_config, save_config

        cfg = load_config()
        wake = cfg.get("wake_word")
        if not isinstance(wake, dict):
            wake = {}
        wake["start_new_session"] = bool(body.start_new_session)
        cfg["wake_word"] = wake
        save_config(cfg)
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"config write failed: {exc}"
        )
    return {"ok": True, **_wake_session_current()}


# ---------------------------------------------------------------------------
# Wake phrase (official key wake_word.phrase) — bilingual, verified save
# ---------------------------------------------------------------------------
# With the sherpa engine ``wake_word.phrase`` is DETECTION config: the engine
# tokenizes the phrase against the KWS model vocabulary at build time
# (wake_word_engines._SherpaKwsEngine), so the identical arm-time freeze rule
# as the conversation-mode section above applies — a saved phrase lands at
# the next wake arming (toggle wake off -> on once), or immediately on the
# next wake start while idle.
#
# Bilingual support falls out of the engine's two tokenizer shapes: a model
# dir carrying bpe.model spells LATIN phrases (gigaspeech); a dir without it
# has a pinyin token list and routes CJK phrases through ppinyin
# (wenetspeech). Neither model handles the other script, so on save we
# auto-select the locally available model matching the phrase script (stored
# under the official sub-key ``wake_word.sherpa.model_dir``; empty means the
# official auto-download English model) and then build the REAL engine via
# the official builder as a verification — the verify-before-write discipline
# used by the video panel's R2T2 params. A failed verification answers
# ok:false with the engine's own error text verbatim and writes nothing.
# Non-sherpa providers treat the phrase as a cosmetic label and save
# unverified. Sync routes run in FastAPI's threadpool, never on the event
# loop; the verification engine is a throwaway and is closed either way.

_WAKE_PHRASE_MAX_LEN = 40
_WAKE_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_WAKE_LATIN = re.compile(r"[A-Za-z]")
_SHERPA_PROVIDERS = frozenset({"sherpa", "sherpa-onnx", "kws", "open"})


class _WakePhraseIn(BaseModel):
    phrase: str = ""


def _wake_section() -> Dict[str, Any]:
    """The live ``wake_word`` section as a plain dict ({} when unset)."""
    try:
        from hermes_cli.config import load_config

        wake = load_config().get("wake_word")
    except Exception:
        wake = None
    return dict(wake) if isinstance(wake, dict) else {}


def _norm_dir(value: Any) -> str:
    """Compare-normalized form for Windows path equality checks (case and
    separator insensitive); never used as a real path."""
    try:
        return os.path.normcase(str(Path(str(value or "")).expanduser().resolve()))
    except Exception:
        return os.path.normcase(str(value or "").strip())


def _phrase_language(phrase: str) -> str:
    """Script routing mirroring the engine's tokenizer split."""
    cjk, latin = bool(_WAKE_CJK.search(phrase)), bool(_WAKE_LATIN.search(phrase))
    if cjk and latin:
        return "mixed"
    return "zh" if cjk else "en"


def _sherpa_local_models() -> list[Dict[str, str]]:
    """Unpacked sherpa KWS models under the official cache root, each tagged
    by tokenizer shape: bpe.model present -> "en", absent -> "zh" (pinyin)."""
    try:
        from hermes_constants import get_hermes_home

        root = Path(get_hermes_home()) / "cache" / "wakewords"
        return [
            {"path": str(d), "kind": "en" if (d / "bpe.model").is_file() else "zh"}
            for d in sorted(p for p in root.iterdir() if (p / "tokens.txt").is_file())
        ]
    except Exception:
        return []


def _wake_phrase_state() -> Dict[str, Any]:
    """Snapshot of the current phrase / model routing / armed picture."""
    wake = _wake_section()
    sub = wake.get("sherpa")
    sub = dict(sub) if isinstance(sub, dict) else {}
    model_dir = str(sub.get("model_dir") or "").strip()
    models = _sherpa_local_models()
    kind = next((m["kind"] for m in models
                 if _norm_dir(m["path"]) == _norm_dir(model_dir)), "")
    if not kind and not model_dir:
        kind = "en"  # no explicit dir => official auto-download (English) path
    phrase = str(wake.get("phrase") or "hey hermes").strip()
    provider = "unknown"
    try:
        import tools.wake_word as ww

        prov = getattr(ww, "_provider", None)
        if callable(prov):
            provider = str(prov(wake))
    except Exception:
        pass
    try:
        from tools.wake_word import is_listening as _is_listening

        listening = bool(_is_listening())
    except Exception:
        listening = False
    return {
        "phrase": phrase,
        "language": _phrase_language(phrase),
        "provider": provider,
        "model_dir": model_dir,
        "model_kind": kind,
        "local_models": models,
        "listening": listening,
    }


def _save_wake_phrase(phrase: str, model_dir: Optional[str]) -> None:
    """Persist phrase (and optionally the sherpa model_dir) via the writer.

    ``model_dir``: a path pins it, "" falls back to the official default
    (the sub-key is removed), None leaves the sub-key untouched."""
    from hermes_cli.config import load_config, save_config

    cfg = load_config()
    wake = cfg.get("wake_word")
    if not isinstance(wake, dict):
        wake = {}
    wake["phrase"] = phrase
    if model_dir is not None:
        sub = wake.get("sherpa")
        sub = dict(sub) if isinstance(sub, dict) else {}
        if model_dir:
            sub["model_dir"] = model_dir
        else:
            sub.pop("model_dir", None)
        wake["sherpa"] = sub
    cfg["wake_word"] = wake
    save_config(cfg)


def _try_build(wake: Dict[str, Any], phrase: str, model_dir: str) -> tuple:
    """True when the official builder accepts this phrase with this model
    dir ("" = official default). The engine is closed either way."""
    import tools.wake_word as ww

    builder = getattr(ww, "_build_engine", None)
    if builder is None:
        return False, "host wake builder unavailable"
    probe = dict(wake)
    probe["phrase"] = phrase
    sub = dict(probe.get("sherpa")) if isinstance(probe.get("sherpa"), dict) else {}
    if model_dir:
        sub["model_dir"] = model_dir
    else:
        sub.pop("model_dir", None)
    probe["sherpa"] = sub
    engine = None
    try:
        engine = builder(probe)
        return True, ""
    except Exception as exc:  # the engine's own error is the user-facing why
        text = " ".join(str(exc or exc.__class__.__name__).split())
        return False, (text[:300] or exc.__class__.__name__)
    finally:
        closer = getattr(engine, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:
                pass


@router.get("/wake-phrase")
def wake_phrase_get() -> Dict[str, Any]:
    """Current wake phrase, its script routing and the armed state."""
    return {"ok": True, **_wake_phrase_state()}


@router.post("/wake-phrase")
def wake_phrase_set(body: _WakePhraseIn) -> Dict[str, Any]:
    """Verify-before-write save of ``wake_word.phrase``.

    Always answers with an ``{ok: bool}`` body — an engine rejection is a
    routine outcome the panel renders from ``detail``, not an HTTP error."""
    phrase = " ".join((body.phrase or "").split())
    if not phrase:
        return {"ok": False, "detail": "wake phrase is empty"}
    if len(phrase) > _WAKE_PHRASE_MAX_LEN:
        return {"ok": False, "detail": f"keep it under {_WAKE_PHRASE_MAX_LEN} characters"}
    if any(ord(c) < 32 for c in phrase):
        return {"ok": False, "detail": "control character in phrase"}

    state = _wake_phrase_state()
    wake = _wake_section()
    armed = state["listening"]
    apply_hint = "rearm-toggle" if armed else "auto-next-start"

    if state["provider"] not in _SHERPA_PROVIDERS:
        # Label semantics on other providers: nothing engine-side to verify.
        try:
            _save_wake_phrase(phrase, None)
        except Exception as exc:
            return {"ok": False, "detail": f"config write failed: {exc}"}
        return {"ok": True, "verified": False, "attempts": [],
                "model_changed": False, "apply": apply_hint,
                **_wake_phrase_state()}

    # Model candidates, in order: the current model when its script already
    # matches the phrase, then matching-script models from the official
    # cache, then the current model anyway, then remaining cache models,
    # and last the official default path (one-time ~13 MB English download).
    want = {"zh": "zh", "en": "en"}.get(_phrase_language(phrase)) \
        or state["model_kind"] or "zh"
    current = state["model_dir"]
    models = state["local_models"]
    order: list[str] = []
    if current and state["model_kind"] == want:
        order.append(current)
    order += [m["path"] for m in models
              if m["kind"] == want and _norm_dir(m["path"]) != _norm_dir(current)]
    if current and state["model_kind"] != want:
        order.append(current)
    order += [m["path"] for m in models
              if m["kind"] != want and _norm_dir(m["path"]) != _norm_dir(current)]
    order.append("")

    attempts: list[Dict[str, str]] = []
    chosen: Optional[str] = None
    for path in order:
        ok, err = _try_build(wake, phrase, path)
        if ok:
            chosen = path
            break
        attempts.append({"model": Path(path).name if path else "default",
                         "error": err})
    if chosen is None:
        joined = "; ".join(f"[{a['model']}] {a['error']}" for a in attempts)
        return {"ok": False, "detail": joined or "engine rejected the phrase",
                "attempts": attempts}

    model_changed = _norm_dir(chosen) != _norm_dir(current)
    try:
        _save_wake_phrase(phrase, chosen)
    except Exception as exc:
        return {"ok": False, "detail": f"engine accepted, config write failed: {exc}"}
    return {"ok": True, "verified": True, "attempts": attempts,
            "model_changed": model_changed, "apply": apply_hint,
            **_wake_phrase_state()}


# ── Voice library: the clone -> TTS unlock flow lives here ─────────
#
# One read route (/voice/library) feeds the whole panel: on-disk model
# scan, the three voice kinds, the saved selection and the synthesis
# config. Two write routes (/voice/preview to hear, /voice/save to
# commit) and one enrollment route (/voice/clone). Every save probes
# the engine for real first — a config that cannot synthesize is
# never written to disk.

class _VoiceIn(BaseModel):
    spec: Dict[str, Any] = Field(default_factory=dict)
    text: str = _i18n.ui("你好，这是一段音色试听。",
                         "Hi, this is a voice preview.")
    # voice_preview() falls back to this when text is blank; without the
    # field an empty panel textarea died with AttributeError -> HTTP 500
    # (pydantic models reject unknown attrs, 2026-11-15 incident).
    preview_text: str = ""


class _VoiceTtsIn(BaseModel):
    quant: str = "auto"
    device: str = "auto"


class _CloneIn(BaseModel):
    model_size: str = ""
    name: str
    ref_audio: str
    ref_text: str = ""
    description: str = ""
    preview_text: str = _i18n.ui("嗨！这是刚保存的新音色，听起来怎么样？",
                                 "Hi! How does my brand-new cloned voice sound?")


# Worker-contract fields; anything else in a panel payload is dropped.
_SPEC_FIELDS = ("kind", "variant", "model_size", "quant", "device", "voice", "speaker",
                "instruct", "ref_audio", "ref_text", "description",
                "temperature")


def _norm_spec(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        raise HTTPException(status_code=400, detail=_i18n.ui("未知的音色类型", "Unknown voice kind"))
    spec = {k: raw[k] for k in _SPEC_FIELDS if k in raw and raw[k] not in (None, "")}
    if spec.get("kind") not in _vy.KINDS:
        raise HTTPException(status_code=400, detail=_i18n.ui("未知的音色类型", "Unknown voice kind"))
    if "variant" in spec and spec["variant"] not in _vy.VARIANTS:
        raise HTTPException(status_code=400, detail=_i18n.ui(f"未知模型变体: {spec['variant']}", f"Unknown model variant: {spec['variant']}"))
    if "model_size" in spec:
        _ms = str(spec["model_size"]).strip().lower()
        if _ms not in ("0.6b", "1.7b"):
            raise HTTPException(
                status_code=400,
                detail=_i18n.ui(f"未知模型粒度: {spec['model_size']}（可用: 0.6b / 1.7b）",
                                f"Unknown model size: {spec['model_size']} (available: 0.6b / 1.7b)"))
        spec["model_size"] = _ms
    if "device" in spec and str(spec["device"]).lower() not in ("auto", "cuda", "cpu"):
        raise HTTPException(status_code=400, detail=_i18n.ui(f"未知后端: {spec['device']}", f"Unknown backend: {spec['device']}"))
    # The worker speaks "auto" as None/"" (None = no pin, let the
    # per-request VRAM gate decide).  Map the literal string away here
    # so every route that builds a spec shares one contract.
    for _k in ("quant", "device"):
        if str(spec.get(_k, "")).strip().lower() == "auto":
            spec[_k] = None
    return spec


def _clone_ledger() -> Dict[str, Any]:
    """Clones registered by THIS panel only. A voice is trusted once the
    user picked its reference file, confirmed the transcript and saved a
    validated synthesis — nothing is inferred from the shared registry
    (same rules on the author's machine as on every user's)."""
    led = _p.load_data().get("clones")
    return led if isinstance(led, dict) else {}


def _clone_ledger_put(name: str, meta: Dict[str, Any]) -> None:
    data = _p.load_data()
    led = data.get("clones")
    if not isinstance(led, dict):
        led = {}
    led[str(name)] = meta
    data["clones"] = led
    _p.save_data(data)


def _clone_ledger_drop(name: str) -> None:
    data = _p.load_data()
    led = data.get("clones")
    if isinstance(led, dict) and str(name) in led:
        del led[str(name)]
        data["clones"] = led
        _p.save_data(data)


def _lib_view() -> Dict[str, Any]:
    base = _p.tts_base_dir()
    # A missing/moved model dir must NOT take the whole settings page down:
    # degrade to an empty scan and surface the human-readable reason; the
    # panel shows the red note and the model-dir row lets the user point at
    # the right folder from there.
    scan_error: Optional[str] = None
    try:
        scan = _vy.scan_models_dir(str(base))
    except _vy.ScanError as exc:
        scan, scan_error = {}, str(exc)
    except Exception as exc:  # noqa: BLE001
        scan, scan_error = {}, "%s: %s" % (type(exc).__name__, exc)
    by_variant = {v.get("variant"): v for v in scan.get("variants") or []}
    free = _w.cuda_free_mib()
    cuda = bool(_w._has_nvidia())
    data = _p.load_data()
    saved_voice = data.get("voice") if isinstance(data.get("voice"), dict) else None
    saved_tts = data.get("tts") if isinstance(data.get("tts"), dict) else {}
    # The recommendation anchors on the variant the selection actually
    # uses (base for clones, customvoice/voicedesign by choice).
    anchor = str((saved_voice or {}).get("variant") or "customvoice")
    if anchor not in by_variant:
        anchor = next(iter(by_variant), "base")
    anchor_rows = (by_variant.get(anchor) or {}).get("quants") or []
    anchor_size = str((saved_voice or {}).get("model_size") or "").strip().lower()
    # the recommender sees the combo's own rows — model_size rides along, so
    # a 0.6b tier gets its half-size VRAM estimate instead of the big build's
    _asized = [q for q in anchor_rows
               if not anchor_size or (str(q.get("model_size") or "").lower() or "1.7b") == anchor_size]
    rec = _vy.recommend(_asized or anchor_rows, free, cuda)
    try:
        reg = _vy.load_registry(_p.voices_registry_path()) or {}
    except Exception:
        reg = {}
    clones = []
    for name, meta in _clone_ledger().items():
        if not isinstance(meta, dict):
            continue
        ref = str(meta.get("ref_audio") or "")
        clones.append({
            "name": str(name),
            "description": str(meta.get("description") or ""),
            "ref_ok": bool(ref) and Path(ref).is_file(),
        })
    clones.sort(key=lambda c: c["name"])
    cv = by_variant.get("customvoice") or {}
    return {
        "scan": {
            name: {
                "present": name in by_variant,
                "quants": [{"quant": str(q.get("quant") or ""),
                            "model_size": str(q.get("model_size") or "").lower(),
                            "size_mib": q.get("size_mib")}
                           for q in (by_variant.get(name) or {}).get("quants") or []],
            } for name in _vy.VARIANTS
        },
        "tokenizer_ok": bool(scan.get("tokenizer")),
        "scan_error": scan_error,
        "recommend": rec,
        "anchor_variant": anchor,
        "anchor_size": anchor_size,
        "free_vram": free,
        "has_cuda": cuda,
        "clones": clones,
        "speakers": [str(s.get("name")) for s in cv.get("speakers") or []],
        "design_available": "voicedesign" in by_variant,
        "default": str(reg.get("default") or ""),
        "saved_voice": saved_voice,
        "saved_tts": {"quant": str(saved_tts.get("quant") or "auto"),
                      "device": str(saved_tts.get("device") or "auto")},
        "voices_dir": str(_p.voices_dir()),
        "active_dir": {
            "path": str(base),
            "custom": bool(str(data.get("tts_base_dir", "")).strip()),
        },
    }


def _preview_spec_enrich(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Clone specs selected by name only carry a voice id; fill the ref
    fields from the registry so the worker never has to guess."""
    spec = dict(spec)
    if spec.get("kind") == "clone" and not spec.get("ref_audio"):
        try:
            reg = _vy.load_registry(_p.voices_registry_path()) or {}
        except Exception:
            reg = {}
        voices = reg.get("voices") if isinstance(reg.get("voices"), dict) else {}
        vid = str(spec.get("voice") or "") or str(reg.get("default") or "")
        entry = voices.get(vid)
        norm = _vy.normalize_entry(vid, entry) if isinstance(entry, dict) else None
        if norm is None:
            norm = {}
        cand = str(norm.get("ref_audio") or "").strip()
        if not cand or not Path(cand).is_file():
            # The registry mirror can lag or hold an empty path; the
            # plugin's own clone ledger is what the enrollment flow
            # actually verified, so it gets the fallback vote.
            meta = _clone_ledger().get(vid) or {}
            cand = str(meta.get("ref_audio") or "").strip()
            if cand and Path(cand).is_file():
                norm = {**norm, "ref_audio": cand,
                        "ref_text": norm.get("ref_text") or
                        str(meta.get("ref_text") or "")}
        if not str((norm or {}).get("ref_audio") or "").strip():
            raise HTTPException(
                status_code=400,
                detail=_i18n.ui(f"克隆音色 '{vid or '(未选择)'}' 不在音色库，无法试听",
                                f"Cloned voice '{vid or '(none)'}' is not in the voice registry — cannot preview"))
        spec["voice"] = vid
        spec["ref_audio"] = str(norm.get("ref_audio") or "")
        spec["ref_text"] = str(norm.get("ref_text") or "")
    return spec


def _run_preview(spec: Dict[str, Any], text: str,
                 timeout_s: int = 180) -> Dict[str, Any]:
    """Spawn one worker --preview run. The env must be stripped of
    PYTHONPATH/PYTHONHOME or the host's own site-packages hijacks the
    engine venv's numpy (cp314 vs cp312 .pyd mismatch)."""
    py = _p.engine_python()
    if not py.is_file():
        raise HTTPException(status_code=503,
                            detail=_i18n.ui(f"引擎 Python 缺失: {py}", f"Engine Python not found: {py}"))
    spec = _preview_spec_enrich(spec)
    env = {k: v for k, v in os.environ.items()
           if k not in ("PYTHONPATH", "PYTHONHOME")}
    env["HERMES_VOICES_DIR"] = str(_p.voices_dir())
    env["VA_VOICE_JSON"] = json.dumps(spec, ensure_ascii=False)
    workdir = str(_p.worker_script().parent)
    try:
        proc = subprocess.Popen(
            [str(py), str(_p.worker_script()), "--preview"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding="utf-8",
            errors="replace", env=env, cwd=workdir,
            creationflags=_CREATE_NO_WINDOW if os.name == "nt" else 0)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=_i18n.ui(f"启动试听进程失败: {exc}", f"Failed to start the preview process: {exc}"))
    try:
        out = proc.communicate(json.dumps({"text": text}, ensure_ascii=False),
                               timeout=timeout_s)[0]
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        raise HTTPException(status_code=504,
                            detail=_i18n.ui(f"试听超过 {timeout_s} 秒未完成", f"The preview did not finish within {timeout_s} seconds"))
    payload = None
    for line in (out or "").splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                pass
    if payload is None:
        raise HTTPException(status_code=502, detail=_i18n.ui("试听进程没有回传结果", "The preview worker returned no result"))
    return payload


@router.get("/voice/library")
def voice_library() -> Dict[str, Any]:
    """Everything the voice panel needs, from one verified source."""
    try:
        return _lib_view()
    except Exception as exc:
        raise HTTPException(status_code=500,
                            detail=_i18n.ui(f"扫描音色库失败: {type(exc).__name__}: {exc}",
                f"Failed to scan the voice registry: {type(exc).__name__}: {exc}"))


@router.post("/voice/preview")
def voice_preview(payload: _VoiceIn) -> Dict[str, Any]:
    # Empty text is legitimate: the panel placeholder promises a sample
    # sentence, so fall back to it instead of rejecting the request.
    text = (payload.text or "").strip() or (payload.preview_text or "").strip() \
        or _i18n.ui("嗨！这是刚保存的新音色，听起来怎么样？",
                    "Hi! How does my brand-new cloned voice sound?")
    spec = _norm_spec(payload.spec)
    res = _run_preview(spec, text)
    lib = _lib_view()
    if not res.get("ok"):
        return {"ok": False, "error": res.get("error") or _i18n.ui("试听失败", "Preview failed"),
                "library": lib}
    return {"ok": True, "seconds": res.get("seconds"),
            "elapsed": res.get("elapsed"), "wav_b64": res.get("wav_b64"),
            "library": lib}


@router.post("/voice/save")
def voice_save(payload: _VoiceIn) -> Dict[str, Any]:
    """Commit the selected voice: probe-synthesize first, write second.
    Clone entries are mirrored into the registry voice library, so
    anything reading registry.json sees them too."""
    spec = _norm_spec(payload.spec)
    kind = spec["kind"]
    if kind == "speaker":
        lib_scan = _vy.scan_models_dir(str(_p.tts_base_dir()))
        valid = [str(s.get("name"))
                 for v in lib_scan.get("variants") or []
                 if v.get("variant") == "customvoice"
                 for s in v.get("speakers") or []]
        if str(spec.get("speaker") or "") not in valid:
            raise HTTPException(
                status_code=400,
                detail=_i18n.ui(f"内置音色 '{spec.get('speaker') or '(未选)'}' 不在本机模型清单（可用: {', '.join(valid) or '无'}）",
                                f"Built-in voice '{spec.get('speaker') or '(none)'}' is not in this model's manifest (available: {', '.join(valid) or 'none'})"))
    if kind == "design" and not str(spec.get("instruct") or "").strip():
        raise HTTPException(status_code=400, detail=_i18n.ui("造声需要先填写声音描述", "Voice design needs a voice description first"))
    text = (payload.text or "").strip() or _i18n.ui("你好，这是一段音色试听。",
                                                    "Hi, this is a voice preview.")
    res = _run_preview(spec, text, timeout_s=240)
    lib = _lib_view()
    if not res.get("ok"):
        return {"ok": False, "error": res.get("error") or _i18n.ui("保存前真实合成失败", "Real synthesis failed before saving"),
                "library": lib}
    data = _p.load_data()
    # Store the enriched spec (ref resolved): a bare clone pointer must
    # stay self-contained even if the registry mirror later goes stale.
    data["voice"] = _preview_spec_enrich(spec)
    _p.save_data(data)
    if kind == "clone" and spec.get("voice"):
        reg_path = _p.voices_registry_path()
        try:
            reg = _vy.load_registry(reg_path) or {}
        except Exception:
            reg = {}
        voices = reg.get("voices") if isinstance(reg.get("voices"), dict) else {}
        name = str(spec["voice"])
        entry = dict(voices.get(name) or {})
        entry.setdefault("description", str(spec.get("description") or ""))
        # A bare name selection carries no ref path; NEVER blank an
        # existing good path (or its transcript) with the empty string,
        # or the next synthesis loses its reference.
        new_ref = str(spec.get("ref_audio") or "")
        if new_ref:
            entry["ref_audio"] = new_ref
        new_text = str(spec.get("ref_text") or "")
        if new_text:
            entry["ref_text"] = new_text
        entry["engine"] = "qwen3-tts"
        voices[name] = entry
        reg["voices"] = voices
        reg["default"] = name
        _vy.save_registry(reg_path, reg)
    return {"ok": True, "seconds": res.get("seconds"),
            "elapsed": res.get("elapsed"), "wav_b64": res.get("wav_b64"),
            "library": lib}


class DesignExportIn(BaseModel):
    wav_b64: str
    path: str = ""
    ref_text: str = ""


@router.post("/voice/design/export")
def voice_design_export(payload: DesignExportIn) -> dict[str, Any]:
    """Persist the wav of a design preview.

    A design voice is re-invented randomly on every synthesis, so the clip
    the user just heard is the only concrete instance of it — exporting is
    what makes it keepable. The panel may hand a user-chosen save path
    (native save dialog); with none the clip lands in the plugin's own
    voices/design folder under a timestamped name. The saved file doubles
    as a clone reference: feeding it back through the clone enrollment is
    how a designed voice becomes a permanent, registrable one.
    """
    blob = str(payload.wav_b64 or "").strip()
    if not blob:
        return {"ok": False,
                "error": _i18n.ui("缺音频数据。", "No audio data.")}
    try:
        data = base64.b64decode(blob)
    except Exception:
        return {"ok": False, "error": _i18n.ui(
            "音频数据已损坏，请重新试听。",
            "The audio data is corrupted — preview again.")}
    if not 1024 <= len(data) <= 20 * 1024 * 1024:
        return {"ok": False, "error": _i18n.ui(
            "试听音频过小或过大，请重新试听后再导出。",
            "The clip is too small or too large — preview again "
            "and retry.")}
    dest = str(payload.path or "").strip()
    if dest:
        parent = os.path.dirname(os.path.abspath(dest))
        if not os.path.isdir(parent):
            return {"ok": False, "error": _i18n.ui(
                "保存位置的文件夹不存在。", "That folder does not exist.")}
        if not dest.lower().endswith(".wav"):
            dest += ".wav"
    else:
        folder = _p.voices_dir() / "design"
        folder.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        dest = str(folder / f"designed-{stamp}.wav")
        n = 1
        while os.path.exists(dest):
            dest = str(folder / f"designed-{stamp}-{n}.wav")
            n += 1
    try:
        with open(dest, "wb") as fh:
            fh.write(data)
    except OSError as exc:
        _log.warning("design export failed: %s", exc)
        return {"ok": False, "error": _i18n.ui(
            f"写入文件失败：{exc}", f"Writing the file failed: {exc}")}
    _log.info("[panel] design voice clip exported: %s (%d bytes)",
                dest, len(data))
    return {"ok": True, "path": dest}


@router.post("/voice/tts")
def voice_tts(payload: _VoiceTtsIn) -> Dict[str, Any]:
    """Save the synthesis config (quant + device). The save probes the
    CURRENT selection under the NEW config, so a quant the engine
    cannot load under the present VRAM never reaches disk."""
    quant = (payload.quant or "auto").strip().lower()
    device = (payload.device or "auto").strip().lower()
    if quant not in ("auto", "bf16", "q8_0", "q4_k_m"):
        raise HTTPException(status_code=400, detail=_i18n.ui(f"未知量化档: {quant}", f"Unknown quantization tier: {quant}"))
    if device not in ("auto", "cuda", "cpu"):
        raise HTTPException(status_code=400, detail=_i18n.ui(f"未知后端: {device}", f"Unknown backend: {device}"))
    data = _p.load_data()
    base_spec = data.get("voice") if isinstance(data.get("voice"), dict) else {}
    spec = _norm_spec({**base_spec, "quant": quant, "device": device})
    if quant != "auto":
        anchor = str(spec.get("variant") or "base")
        asize = str(spec.get("model_size") or "").strip().lower()
        lib_scan = _vy.scan_models_dir(str(_p.tts_base_dir()))
        # scanner rows carry (quant, model_size): the tier the user picked
        # is only real for the model size their voice combo pins
        avail = [{"quant": str(q.get("quant") or "").upper(),
                  "model_size": str(q.get("model_size") or "").lower(),
                  "size_mib": q.get("size_mib")}
                 for v in lib_scan.get("variants") or []
                 if v.get("variant") == anchor
                 for q in v.get("quants") or []]
        if asize:
            sized = [a for a in avail if (a["model_size"] or "1.7b") == asize]
            avail = sized or avail
        if _vy.check_quant(quant.upper(), avail) is None:
            shown = ", ".join(sorted({
                a["quant"] + (" · " + a["model_size"].upper()
                              if a["model_size"] else "")
                for a in avail})) or "none"
            return {"ok": False,
                    "error": _i18n.ui(
                        f"变体 {anchor}"
                        + (f" · {asize.upper()}" if asize else "")
                        + f" 磁盘上没有 {quant.upper()} 量化文件（实际有: {shown}）",
                        f"Variant {anchor}"
                        + (f" · {asize.upper()}" if asize else "")
                        + f" has no {quant.upper()} quant file on disk (found: {shown})"),
                    "library": _lib_view()}
    res = _run_preview(spec, _i18n.ui("合成配置验证。", "Configuration verification."), timeout_s=240)
    lib = _lib_view()
    if not res.get("ok"):
        return {"ok": False, "error": res.get("error") or _i18n.ui("新配置验证失败", "The new configuration failed verification"),
                "library": lib}
    data = _p.load_data()
    data["tts"] = {"quant": quant, "device": device,
                   "updated_at": datetime.now().isoformat(timespec="seconds")}
    _p.save_data(data)
    return {"ok": True, "seconds": res.get("seconds"),
            "elapsed": res.get("elapsed"), "library": lib}


@router.post("/voice/clone")
def voice_clone(payload: _CloneIn) -> Dict[str, Any]:
    """Enroll a brand-new clone: validate the reference, probe-synthesize
    it for real, then write the registry entry and make it the current
    selection. A failed enrollment writes nothing at all."""
    name = (payload.name or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_\u4e00-\u9fff]{1,32}", name):
        raise HTTPException(
            status_code=400,
            detail=_i18n.ui("音色名只可用字母、数字、下划线或中文，最长 32 位",
                            "Voice names may only use letters, digits, underscores or Chinese characters, up to 32 characters"))
    ref = Path((payload.ref_audio or "").strip().strip('"')).expanduser()
    if not ref.is_absolute():
        ref = _p.voices_dir() / ref.name
    if not ref.is_file():
        raise HTTPException(status_code=400,
                            detail=_i18n.ui(f"参考音频不存在: {payload.ref_audio}", f"Reference audio not found: {payload.ref_audio}"))
    if ref.suffix.lower() not in (".wav", ".mp3", ".flac", ".ogg"):
        raise HTTPException(status_code=400,
                            detail=_i18n.ui(f"不支持的参考音频格式: {ref.suffix}", f"Unsupported reference audio format: {ref.suffix}"))
    # Validate the reference where it lives: a user's audio sits in
    # wherever the file dialog happened to point until the synthesis
    # check actually passes; only then is it pulled into the voices
    # dir and registered. Failed attempts leave zero traces.
    size = (payload.model_size or "").strip().lower()
    if size and size not in ("0.6b", "1.7b"):
        raise HTTPException(
            status_code=400,
            detail=_i18n.ui(f"未知模型粒度: {payload.model_size}（可用: 0.6b / 1.7b）",
                            f"Unknown model size: {payload.model_size} (available: 0.6b / 1.7b)"))
    spec = {
        "kind": "clone", "variant": "base", "voice": name,
        "model_size": size or "1.7b",
        "ref_audio": str(ref), "ref_text": payload.ref_text.strip(),
    }
    res = _run_preview(spec, payload.preview_text.strip()
                       or _i18n.ui("你好呀，这就是刚刚保存的新克隆音色，你觉得听起来怎么样？",
                                  "Hi! This is the new cloned voice we just saved — how does it sound?"),
                       timeout_s=240)
    lib = _lib_view()
    if not res.get("ok"):
        return {"ok": False, "error": res.get("error") or _i18n.ui("克隆验证失败", "Clone verification failed"),
                "library": lib}
    stored = ref
    vdir = _p.voices_dir()
    if not str(ref).lower().startswith(str(vdir).lower()):
        vdir.mkdir(parents=True, exist_ok=True)
        stored = vdir / f"{name}{ref.suffix.lower()}"
        shutil.copyfile(ref, stored)
    spec["ref_audio"] = str(stored)
    reg_path = _p.voices_registry_path()
    try:
        reg = _vy.load_registry(reg_path) or {}
    except Exception:
        reg = {}
    voices = reg.get("voices") if isinstance(reg.get("voices"), dict) else {}
    voices[name] = {
        "engine": "qwen3-tts",
        "description": payload.description.strip() or _i18n.ui(f"克隆音色 {name}", f"Cloned voice {name}"),
        "ref_audio": str(stored),
        "ref_text": payload.ref_text.strip(),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    reg["voices"] = voices
    reg["default"] = name
    _vy.save_registry(reg_path, reg)
    _clone_ledger_put(name, {
        "ref_audio": str(spec.get("ref_audio") or ""),
        "ref_text": str(spec.get("ref_text") or ""),
        "description": str(payload.description or "").strip(),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    })
    data = _p.load_data()
    data["voice"] = dict(spec)
    _p.save_data(data)
    return {"ok": True, "name": name, "seconds": res.get("seconds"),
            "elapsed": res.get("elapsed"), "wav_b64": res.get("wav_b64"),
            "library": lib}


class _RefIn(BaseModel):
    path: str = ""


@router.post("/voice/ref/transcribe")
def voice_ref_transcribe(payload: _RefIn) -> Dict[str, Any]:
    """Transcribe a picked reference audio with the plugin's LOCAL ASR
    (faster-whisper ladder, zero network) so the panel can print the
    transcript for the user to verify. On failure the panel asks the
    user to type the words by hand — both paths end in a manual edit
    box anyway."""
    raw = str(payload.path or "").strip().strip('"')
    if not raw:
        return {"ok": False, "error": _i18n.ui("请先选择参考音频文件", "Pick a reference audio file first")}
    ref = Path(raw).expanduser()
    if not ref.is_file():
        return {"ok": False, "error": _i18n.ui("文件不存在: %s", "File not found: %s") % ref}
    try:
        from yache_engine import stt as _stt
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": _i18n.ui("转录引擎加载失败: %s: %s",
                                              "Transcription engine failed to load: %s: %s")
                % (type(exc).__name__, exc)}
    # Drive the ASR worker directly (not via LocalSTTProvider): the
    # provider is the chat pipeline's front door and its transcribe()
    # also trips the wake-word gate for 180 s — borrowing it from a
    # settings-panel probe would silently mute wake-up. Same interpreter
    # ladder, same worker, minus the chat-side side effects.
    payload = None
    last_err = "no usable ASR interpreter configured"
    for py in _p.stt_python_candidates():
        if not Path(py).is_file():
            continue
        try:
            payload = _stt._run_worker(
                str(py), ["--wav", str(ref),
                          "--model", _stt_saved()["model"]], 120.0)
            break
        except Exception as exc:  # noqa: BLE001
            last_err = "%s: %s" % (type(exc).__name__, exc)
    if payload is None:
        return {"ok": False, "error": _i18n.ui("转录失败: %s", "Transcription failed: %s") % last_err}
    # worker contract: {"transcript": ..., "language": ..., ...}
    text = str(payload.get("transcript") or "").strip()
    if not text:
        return {"ok": False, "error": _i18n.ui("没有识别出语音内容", "No speech was recognized")}
    return {"ok": True, "text": text,
            "language": str(payload.get("language") or "")}


class _ScanDirBody(BaseModel):
    path: Optional[str] = None
    reset: bool = False


@router.post("/voice/models/scan")
def voice_models_scan(payload: _ScanDirBody) -> Dict[str, Any]:
    """Point the whole plugin at a model directory (design doc: the FIRST
    row of the clone panel). The directory is validated by a REAL scan
    before anything is saved; the engine re-reads tts_base_dir per request,
    so a new directory is hot with no restart. reset=true restores the
    platform default location. Stock voice-route semantics: HTTP 200 with
    ok/error so the panel never has to parse HTTP codes."""
    try:
        if payload.reset:
            data = _p.load_data()
            data.pop("tts_base_dir", None)
            _p.save_data(data)
            return {"ok": True, "reset": True}
        raw = (payload.path or "").strip().strip('"')
        if not raw:
            return {"ok": False, "error": _i18n.ui("请先选择或填写模型目录", "Pick or type a model folder first")}
        root = Path(raw).expanduser()
        if not root.is_dir():
            return {"ok": False, "error": _i18n.ui("目录不存在: %s", "Folder not found: %s") % root}
        # The scanner accepts the models/ subdir the folder picker tends to
        # land in, but every engine path (venv, site-packages, dlls) hangs
        # off the ROOT dir. Normalise before persisting, otherwise
        # tts_base_dir=<...>\models makes engine_python() chase
        # <...>\models\.venv and every preview/synthesis dies with
        # "engine Python missing" (real incident, 2026-11-15).
        if not (root / ".venv").is_dir():
            parent = root.parent
            if (parent / ".venv").is_dir():
                root = parent
        scan = _vy.scan_models_dir(str(root))  # ScanError -> ok:false below
        variants = scan.get("variants") or []
        if not variants:
            return {"ok": False,
                    "error": _i18n.ui("这里没找到任何 Qwen3-TTS 变体模型", "No Qwen3-TTS variant model was found in this folder")}
        data = _p.load_data()
        data["tts_base_dir"] = str(root)
        _p.save_data(data)
        return {"ok": True,
                "found_variants": [str(v.get("variant")) for v in variants]}
    except _vy.ScanError as exc:
        return {"ok": False, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}


# ── download wizard (model + GPU accelerator; dashboard/download.py) ─────
# Thin wrappers over the download state machine. Stock panel semantics: HTTP
# 200 with ok/error so the frontend never has to parse HTTP codes; the one
# exception is /wizard/status which is polled and returns the raw state.


def _dl_or_none():
    return _dl


class _WizardModelBody(BaseModel):
    combo: str = Field(..., description="e.g. 0.6b-customvoice / 1.7b-base")
    quant: str = Field(..., description="q4_k_m | q8_0 | bf16 | f32")
    target: str = Field(..., description="directory picked by the user")


@router.get("/wizard/catalog")
def wizard_catalog():
    """The two dropdowns flattened: model (size x variant) x quantization,
    each combo carrying its real download size for the live estimate line."""
    d = _dl_or_none()
    if d is None:
        return {"ok": False, "error": _i18n.ui("下载模块未装载", "download module unavailable")}
    return {"ok": True, "combos": d.catalog()}


@router.get("/wizard/source")
def wizard_source():
    """Fresh per-call probe of huggingface.co vs hf-mirror.com (never cached,
    so a VPN toggle is picked up on the next click)."""
    d = _dl_or_none()
    if d is None:
        return {"ok": False, "error": _i18n.ui("下载模块未装载", "download module unavailable")}
    return {"ok": True, **d.probe_endpoint()}


@router.post("/wizard/model")
def wizard_model_start(payload: _WizardModelBody):
    d = _dl_or_none()
    if d is None:
        return {"ok": False, "error": _i18n.ui("下载模块未装载", "download module unavailable")}
    r = d.start_model_download(payload.combo, payload.quant, payload.target)
    if not r.get("ok"):
        return {"ok": False, "error": r.get("detail") or _i18n.ui("无法启动下载", "cannot start")}
    return r


@router.get("/wizard/gpu")
def wizard_gpu_status():
    """Installed-ness + the backend-computed gate (a model pair scans clean
    from the loaded dir) that unlocks the GPU pack button."""
    d = _dl_or_none()
    if d is None:
        return {"ok": False, "error": _i18n.ui("下载模块未装载", "download module unavailable")}
    return {"ok": True, **d.gpu_status()}


@router.post("/wizard/gpu")
def wizard_gpu_start():
    """The gate is re-checked server-side; a frontend flag is never trusted."""
    d = _dl_or_none()
    if d is None:
        return {"ok": False, "error": _i18n.ui("下载模块未装载", "download module unavailable")}
    g = d.gpu_status()
    r = d.start_gpu_download(bool(g.get("gate_ok")), str(g.get("gate_reason") or ""))
    if not r.get("ok"):
        return {"ok": False, "error": r.get("detail") or _i18n.ui("无法启动下载", "cannot start")}
    return r


@router.get("/wizard/status")
def wizard_status():
    """Poll target for both progress bars: state/pct/mbps/eta_s/step_label."""
    d = _dl_or_none()
    if d is None:
        return {"state": "unavailable"}
    return d.download_status()


@router.post("/wizard/resume")
def wizard_resume():
    d = _dl_or_none()
    if d is None:
        return {"ok": False, "error": _i18n.ui("下载模块未装载", "download module unavailable")}
    r = d.resume_download()
    if not r.get("ok"):
        return {"ok": False, "error": r.get("detail", "cannot resume")}
    return r


@router.post("/wizard/cancel")
def wizard_cancel():
    d = _dl_or_none()
    if d is None:
        return {"ok": False, "error": _i18n.ui("下载模块未装载", "download module unavailable")}
    r = d.cancel_download()
    if not r.get("ok"):
        return {"ok": False, "error": r.get("detail", "nothing to cancel")}
    return r


@router.delete("/voice/clone/{name}")
def voice_clone_delete(name: str):
    """Remove a clone entry from the voices registry. The reference wav
    stays on disk — deleting recordings is the user's call, not a side
    effect of this button. If the plugin's saved selection pointed at this
    clone, the selection is cleared so the panels unlock consistently."""
    key = (name or "").strip()
    if not key:
        raise HTTPException(status_code=400, detail=_i18n.ui("音色名为空", "Voice name is empty"))
    try:
        reg_path = _p.voices_registry_path()
        reg = json.loads(Path(reg_path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500,
                            detail=_i18n.ui("读音色注册表失败: %s", "Failed to read the voice registry: %s") % exc)
    voices = reg.get("voices") or {}
    if key not in voices:
        raise HTTPException(status_code=404, detail=_i18n.ui("本机没有克隆音色 '%s'", "no cloned voice '%s' on this machine") % key)
    voices.pop(key)
    if reg.get("default") == key:
        reg.pop("default", None)
    Path(reg_path).write_text(json.dumps(reg, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    _clone_ledger_drop(key)
    pdata = _p.load_data()
    sv = pdata.get("voice")
    if isinstance(sv, dict) and sv.get("kind") == "clone" \
            and str(sv.get("voice") or "") == key:
        pdata.pop("voice", None)
        _p.save_data(pdata)
    return _lib_view()


# ---------------------------------------------------------------------------
# STT transcription settings (see the STT panel design notes)
#
# One official path, no mirror detours: model + language live in the host's
# own stt.local.* keys (the host feeds them into transcribe() live, zero
# drift); device has no official key, so it stays in the plugin data file.
# Saving first PROVES the new settings: the worker really loads the chosen
# model (--verify = genuine WhisperModel load + a 1 s silent transcribe,
# official-source download included on first use). Nothing persists unless
# the engine itself says yes; its error comes back verbatim.
# ---------------------------------------------------------------------------

_WHISPER_SIZES = ["tiny", "base", "small", "medium", "large-v3"]
_WHISPER_SIZE_GB = {"tiny": 0.08, "base": 0.15, "small": 0.48,
                    "medium": 1.53, "large-v3": 3.1}
_WHISPER_LANGS = ["", "zh", "en", "ja", "ko", "es", "fr", "de", "ru"]
# CTranslate2 fingerprints. faster-whisper's local-directory mode only
# accepts CT2 snapshots, so checking these up front turns "user picked
# whisper.cpp's .ggml or a raw safetensors folder" into an honest message
# instead of a confusing engine crash at first real use.
_CT2_MARKERS = ("model.bin", "tokenizer.json", "vocabulary.txt")


def _stt_saved() -> Dict[str, str]:
    """Current STT knobs: model/language from the official config (the
    live source the host feeds to transcribe()), device from the plugin
    data file. Never raises — an unreadable config just means defaults."""
    model, language = "small", ""
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        stt_cfg = cfg.get("stt") if isinstance(cfg, dict) else None
        local = stt_cfg.get("local") if isinstance(stt_cfg, dict) else None
        if isinstance(local, dict):
            model = str(local.get("model") or "small").strip() or "small"
            language = str(local.get("language") or "").strip()
    except Exception:
        pass
    try:
        device = str(((_p.load_data().get("stt") or {})
                      .get("device")) or "auto")
    except Exception:
        device = "auto"
    return {"model": model, "language": language, "device": device}


def _hub_of() -> Path:
    """The ONE hub-root rule shared by cache scan, delete and download
    progress. Environment variables win when present; otherwise the
    official config's huggingface_hub.HF_HOME decides — the desktop app
    lets the user point the model library anywhere and feeds that to
    official children via config, but the gateway process itself may
    never carry HF_HOME in its environment."""
    try:
        from yache_engine import stt as _stt
        return _stt._hub_root()
    except Exception:  # noqa: BLE001  (mirror stt's rule as fallback)
        home = Path(os.environ.get("HF_HOME")
                    or (Path.home() / ".cache" / "huggingface"))
        return Path(os.environ.get("HF_HUB_CACHE") or (home / "hub"))


def _whisper_cache() -> Dict[str, float]:
    """Systran faster-whisper sizes really on disk (GB per snapshot).
    A size counts ONLY when a snapshot holds model.bin — half-finished
    downloads once masqueraded as complete caches and bit this family
    before (video-xray), so a bare directory name is not enough."""
    hub = _hub_of()
    found: Dict[str, float] = {}
    try:
        for d in sorted(hub.glob("models--Systran--faster-whisper-*")):
            size = d.name.rsplit("faster-whisper-", 1)[-1]
            snaps = d / "snapshots"
            if not snaps.is_dir():
                continue
            for s in snaps.iterdir():
                if (s / "model.bin").is_file():
                    total = sum(f.stat().st_size for f in s.rglob("*")
                                if f.is_file())
                    found[size] = round(total / 2 ** 30, 2)
                    break
    except OSError:
        pass
    return found


@router.get("/stt/state")
def stt_state() -> Dict[str, Any]:
    """The first step is the health check: engine availability up front
    (red screen stops the user before they fill anything), then where the
    knobs currently stand and which sizes the cache really holds, so the
    dropdown can label each one cached vs download-on-save."""
    cur = _stt_saved()
    try:
        from yache_engine import stt as _stt
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": _i18n.ui("转录引擎模块加载失败: %s: %s",
                                              "Transcription engine module failed to import: %s: %s")
                % (type(exc).__name__, exc)}
    pys = [str(x) for x in _p.stt_python_candidates() if Path(x).is_file()]
    engine_ok, version, py_used, err = False, "", "", ""
    for py in pys:
        try:
            out = _stt._run_worker(py, ["--probe"], 120.0)
            engine_ok = True
            version = str(out.get("version") or "")
            py_used = py
            break
        except Exception as exc:  # noqa: BLE001
            err = "%s: %s" % (type(exc).__name__, exc)
    cached = _whisper_cache()
    sizes = [{"name": s, "gb": cached.get(s, _WHISPER_SIZE_GB.get(s, 0.5)),
              "cached": s in cached} for s in _WHISPER_SIZES]
    for s in sorted(cached):
        if s not in _WHISPER_SIZES:
            # Extra CT2 variants the user cached on their own (e.g.
            # large-v3-turbo): selectable without any new list here.
            sizes.append({"name": s, "gb": cached[s], "cached": True})
    return {"ok": True, "engine_ok": engine_ok, "version": version,
            "python": py_used, "error": err, "model": cur["model"],
            "language": cur["language"], "device": cur["device"],
            "sizes": sizes, "langs": _WHISPER_LANGS}


class _SttIn(BaseModel):
    # Builtin-union annotations on purpose: with `from __future__ import
    # annotations` + a single-file import (how web_server_dashboard.py
    # loads this module), pydantic resolves names against builtins and a
    # bare `Optional` fails there. `str | None` needs no import at all.
    model: str | None = None
    language: str | None = None
    device: str | None = None


@router.post("/stt/save")
def stt_save(payload: _SttIn) -> Dict[str, Any]:
    """Save = the engine must accept the new settings before anything
    persists. Local directories are fingerprint-checked for the CT2
    markers first (instant honest error for the wrong format), then the
    worker really loads the model — an uncached size downloads from the
    official source right here, so the gate stays patient (~30 min)."""
    model = str(payload.model or "").strip()
    if not model:
        return {"ok": False, "error": _i18n.ui("请选择模型尺寸或本地模型目录", "Pick a model size or a local model folder first")}
    device = str(payload.device or "auto").strip().lower()
    if device not in ("auto", "cpu", "cuda"):
        return {"ok": False, "error": _i18n.ui("设备只认 自动 / CPU / GPU，收到: %r",
                                              "Device must be auto / CPU / GPU, got: %r")
                % device}
    language = str(payload.language or "").strip().lower()
    if language and language not in _WHISPER_LANGS:
        return {"ok": False, "error": _i18n.ui("不支持的语言代码: %r", "Unsupported language code: %r") % language}

    # A path has a backslash, a drive prefix, or starts with / or ~; an
    # org-scoped repo name ("Systran/faster-whisper-small") is only ever
    # forward-slashes in the middle — don't misread it as a missing dir.
    looks_path = ("\\" in model) or \
                 re.match(r"^(?:[A-Za-z]:[/\\]|[/~])", model) is not None
    if looks_path:
        p = Path(model).expanduser()
        if not p.is_dir():
            return {"ok": False, "error": _i18n.ui("目录不存在: %s", "Folder not found: %s") % p}
        miss = [m for m in _CT2_MARKERS if not (p / m).is_file()]
        if miss:
            return {"ok": False, "error":
                    _i18n.ui("这个目录不是 faster-whisper (CTranslate2) 格式，缺 %s。whisper.cpp 的 .ggml/.bin 和 HF safetensors 快照需要先转换成 CTranslate2 才能装载",
                             "This folder is not in faster-whisper (CTranslate2) format — missing %s. whisper.cpp .ggml/.bin files and HF safetensors snapshots must first be converted to CTranslate2 before they can be loaded")
                    % ", ".join(miss)}
    # Non-path names are size names (whitelist) or any CT2 repo the engine
    # can resolve (e.g. distil variants); "repo not found" comes back from
    # the engine verbatim below — better than us guessing repo lists here.
    try:
        from yache_engine import stt as _stt
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": _i18n.ui("转录引擎模块加载失败: %s: %s",
                                              "Transcription engine module failed to import: %s: %s")
                % (type(exc).__name__, exc)}
    worker_args = ["--model", model, "--device", device, "--verify"]
    last_err = "no usable ASR interpreter configured"
    out = None
    for py in _p.stt_python_candidates():
        if not Path(py).is_file():
            continue
        try:
            out = _stt._run_worker(py, worker_args, 1800.0)
            break
        except Exception as exc:  # noqa: BLE001
            last_err = "%s: %s" % (type(exc).__name__, exc)
    if out is None:
        return {"ok": False, "error": _i18n.ui("引擎加载验证失败: %s", "Engine load verification failed: %s") % last_err}

    # Green engine -> persist. model/language go to the host's own keys
    # (same atomic official write as the stop-phrase card), device to the
    # plugin data file. A config-write failure leaves the old settings
    # untouched, which is the whole point of the gate.
    try:
        from hermes_cli.config import load_config, save_config
        cfg = load_config()
        if not isinstance(cfg, dict):
            cfg = {}
        stt_cfg = cfg.get("stt") if isinstance(cfg.get("stt"), dict) else {}
        cfg["stt"] = stt_cfg
        local = stt_cfg.get("local") if isinstance(stt_cfg.get("local"),
                                                   dict) else {}
        stt_cfg["local"] = local
        local["model"] = model
        if language:
            local["language"] = language
        else:
            local.pop("language", None)
        save_config(cfg)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": _i18n.ui("写入官方配置失败: %s: %s",
                                              "Failed to write the official config: %s: %s")
                % (type(exc).__name__, exc)}
    try:
        data = _p.load_data()
        block = data.get("stt") if isinstance(data.get("stt"), dict) else {}
        data["stt"] = block
        block["device"] = device
        _p.save_data(data)
    except Exception as exc:  # noqa: BLE001
        _log.warning("[panel] stt device persist failed: %s", exc)
    return {"ok": True, "model": model, "language": language,
            "device": device, "engine": str(out.get("version") or ""),
            "device_used": str(out.get("device") or ""),
            "load_s": out.get("load_s")}


_DL_LOCK = threading.Lock()
_DL: Dict[str, Any] = {}   # the one in-flight download job (state/model/error)


@router.post("/stt/download")
def stt_download(payload: _SttIn) -> Dict[str, Any]:
    """Explicit, user-triggered model download: the ONLY network exit of
    this whole chain (saves/verifies stay strictly offline). Loads
    nothing in-request: spawns the engine worker (verify = download +
    real load + one silence pass) in a background thread and answers at
    once, so the panel can poll /stt/download/status and paint a live
    green progress bar. A failed download leaves nothing set — the
    model stays uncached, the user sees the raw engine error and may
    retry; a success lands as a warm cache, which is exactly what
    unlocks Verify & save."""
    model = str(payload.model or "").strip()
    if not model:
        return {"ok": False, "error": _i18n.ui("先选择一个模型尺寸再下载", "Pick a model size first, then download")}
    if ("\\" in model) or re.match(r"^(?:[A-Za-z]:[/\\]|[/~])",
                                    model) is not None:
        return {"ok": False,
                "error": _i18n.ui("本地目录里的模型就在机器上，不需要在线下载",
                                  "The model in a local folder is already on this machine — no online download needed")}
    if model not in _WHISPER_SIZES:
        return {"ok": False,
                "error": _i18n.ui("在线下载只接受白名单尺寸: %r", "Online download only accepts whitelisted sizes: %r") % model}
    if model in _whisper_cache():
        return {"ok": True, "model": model, "state": "done",
                "already_cached": True}
    try:
        from yache_engine import stt as _stt
    except Exception as exc:  # noqa: BLE001
        return {"ok": False,
                "error": _i18n.ui("转录引擎模块没装载: %s: %s",
                                "Transcription engine module is not loaded: %s: %s")
                         % (type(exc).__name__, exc)}
    pys = [py for py in _p.stt_python_candidates() if Path(py).is_file()]
    if not pys:
        return {"ok": False, "error": "no usable ASR interpreter configured"}
    with _DL_LOCK:
        if _DL.get("state") == "running":
            return {"ok": True, "model": _DL.get("model", model),
                    "state": "running", "started": False}
        _DL.clear()
        _DL.update({"state": "running", "model": model,
                    "started_at": time.time(), "error": ""})

    def _run() -> None:
        err = "no usable ASR interpreter configured"
        for py in pys:  # first interpreter that finishes wins
            try:
                _stt._run_worker(py, ["--model", model, "--device",
                                      "cpu", "--verify"], 1800.0,
                                 online=True)
                err = ""
                break
            except Exception as exc:  # noqa: BLE001  (raw error travels)
                err = "%s: %s" % (type(exc).__name__, exc)
        with _DL_LOCK:
            _DL["state"] = "error" if err else "done"
            _DL["error"] = err
            _DL["finished_at"] = time.time()

    threading.Thread(target=_run, name="yache-stt-download",
                     daemon=True).start()
    return {"ok": True, "model": model, "state": "running",
            "started": True, "size_gb": _WHISPER_SIZE_GB[model]}


@router.post("/stt/delete")
def stt_delete(payload: _SttIn) -> Dict[str, Any]:
    """Explicitly forget one downloaded whisper size from disk — the
    Download button's counterpart. Guards: user-placed directory models
    are never touched (that folder is the user's property), the CURRENT
    active engine is refused (deleting it would silence voice mode on
    the spot), and a size that was never downloaded has nothing to
    free. rmtree can race a worker that just finished verify (its
    process may still linger a moment holding model.bin), so retry a
    few beats; a truly locked file returns the raw OS error rather
    than pretending a half-deleted cache is gone."""
    model = str(payload.model or "").strip()
    if not model or ".." in model:
        return {"ok": False, "error": _i18n.ui("先选择一个要删除的模型尺寸", "Pick a model size to delete first")}
    if re.match(r"^(?:[A-Za-z]:[/\\]|[/~])", model) is not None:
        return {"ok": False, "error": _i18n.ui("本地目录里的模型是您自己放的，请自行管理该目录",
                                              "Models in a local folder are your own files — please manage that folder yourself")}
    try:
        from yache_engine import stt as _stt
    except Exception as exc:  # noqa: BLE001
        return {"ok": False,
                "error": _i18n.ui("转录引擎模块没装载: %s: %s",
                                "Transcription engine module is not loaded: %s: %s")
                         % (type(exc).__name__, exc)}
    if _stt_saved().get("model") == model:
        return {"ok": False,
                "error": _i18n.ui("%s 正在生效 — 先切换到别的模型并保存，再删它",
                                "%s is currently active — switch to a different model and save first, then delete it")
                         % model}
    if model not in _whisper_cache():
        return {"ok": False, "error": _i18n.ui("本机没有该模型的缓存，无需删除", "no cached copy of this model on this machine — nothing to delete")}
    # Same root rule as _whisper_cache(). HF names cache folders after
    # the FULL repo: a bare size name lives in
    # models--Systran--faster-whisper-<name>; an explicit "org/repo"
    # spelling resolves to its own folder. Guessing from the bare name
    # alone once "deleted" a model that was never touched.
    repo = model if "/" in model else "Systran/faster-whisper-" + model
    rdir = _hub_of() / ("models--" + repo.replace("/", "--"))
    if not rdir.is_dir():
        # Cache scan says yes, disk says no: already gone — be idempotent.
        return {"ok": True, "model": model, "freed_gb": 0.0}
    freed = 0
    try:
        for p in rdir.rglob("*"):
            if p.is_file():
                freed += p.stat().st_size
    except OSError:
        pass
    err = ""
    for _ in range(3):
        try:
            shutil.rmtree(rdir)
            err = ""
            break
        except OSError as exc:  # a lingering verify worker may hold it
            err = "%s: %s" % (type(exc).__name__, exc)
            time.sleep(0.6)
    if err:
        return {"ok": False, "error": _i18n.ui("删除失败（模型可能正被占用）: %s",
                                              "delete failed (the model may be in use): %s")
                                      % err}
    return {"ok": True, "model": model,
            "freed_gb": round(freed / 1_000_000_000.0, 2)}


@router.get("/stt/download/status")
def stt_download_status() -> Dict[str, Any]:
    """Poll target while the Download button is waiting. Percent/speed/
    ETA come from bytes already on disk (HF stages growing
    *.incomplete blobs), so the bar tells the truth even while the
    engine itself stays silent; no job (or a gateway restart) reads as
    idle and the partial cache simply resumes on the next click."""
    with _DL_LOCK:
        job = dict(_DL)
    state = str(job.get("state") or "")
    if state not in ("running", "done", "error"):
        return {"state": "idle"}
    model = str(job.get("model") or "")
    out: Dict[str, Any] = {"state": state, "model": model,
                           "error": job.get("error", "")}
    have, total = 0, 1
    try:
        from yache_engine import stt as _stt
        prog = _stt.download_progress(model,
                                      _WHISPER_SIZE_GB.get(model, 0.0))
        have = int(prog["have_bytes"])
        total = max(int(prog["total_bytes"]), 1)
    except Exception:  # noqa: BLE001  (status must never explode)
        pass
    out["have_bytes"], out["total_bytes"] = have, total
    if state == "running":
        out["pct"] = round(min(have / total * 100.0, 99.5), 1)
        span = max(time.time() - float(job.get("started_at")
                                       or time.time()), 1.0)
        mbps = round(have / span / 1_000_000.0, 2)
        out["mbps"] = mbps
        out["eta_s"] = (round((total - have) / (mbps * 1e6), 1)
                        if mbps > 0.05 and total > have else None)
    elif state == "done":
        out["pct"] = 100.0
    return out


# ── ASR environment: a venv the plugin owns, built on one click ──────────
#
# Why this exists at all: faster-whisper and PyAV are NOT in Hermes'
# dependency lock, so they cannot live inside the gateway venv — an
# official environment rebuild regenerates that venv from the stock lock
# and silently drops anything added by hand (this exact wipe silenced
# local STT on 2026-11-14). The durable home is a venv the plugin owns,
# under HERMES_HOME: official upgrades rewrite the source checkout and
# the gateway environment, and never touch HERMES_HOME/stt.
#
# Users are not expected to know what a venv is, so ONE button builds it:
#   1. python -m venv <HERMES_HOME>/stt/.venv — off the interpreter the
#      host is literally running (sys.executable), so no Hermes-internal
#      path is hardcoded and nothing depends on how Hermes was installed;
#   2. pip install the two pinned dependencies;
#   3. the worker's --probe, i.e. the real import, not a wish.
# If that base interpreter later moves (an official upgrade can replace
# the bundled Python), the venv stops answering and the panel says
# "broken" — one more click rebuilds it. Models live in the user's own
# model library and survive every rebuild untouched.
#
# Network is touched only while this button is running — the same single
# exit policy as the model Download button.

_ASR_PKGS = ("faster-whisper==1.2.1", "av<19")
# Versions verified together on the shipping box. The index used for the
# first attempt is whatever the user's own pip config says (a local mirror
# is faster for most users); only if that mirror has nothing do we fall
# back to the official index — a real failure seen during development: a
# popular CN mirror carries no faster-whisper at all.
_ASR_INDEX_FALLBACK = "https://pypi.org/simple"
_ENV_TAIL = 25          # log lines handed to the panel
_ENV_STEP_TIMEOUT = {   # per-step ceilings, seconds
    "wipe": 120.0, "venv": 300.0, "pip": 1800.0, "verify": 150.0,
}
_ENV_LOCK = threading.Lock()
_ENV: Dict[str, Any] = {}   # the one install job this process is running


def _env_proc_env() -> Dict[str, str]:
    """Sanitized environment for the installer's subprocesses.

    Same hygiene the transcription worker uses: the gateway exports a
    PYTHONPATH pointing at its own site-packages, which would shadow the
    fresh venv's packages, and HF_HOME must come from config so the model
    library stays wherever the user pointed it.
    """
    from yache_engine import stt as _stt
    env = _stt._worker_env(online=True)
    # the child must not format its output in the machine codepage
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def _env_win_flags() -> int:
    from yache_engine import stt as _stt
    return _stt._win_flags()


def _env_free_gb() -> float:
    """Free gigabytes where the environment will land (-1 = unknown)."""
    try:
        return round(shutil.disk_usage(str(_p.hermes_home()))[0] / 1024 ** 3, 1)
    except OSError:
        return -1.0


def _env_py_version(py: str) -> str:
    """Ask the venv interpreter for its version; '' means it cannot start.

    A Windows venv is a thin shim over the install it was built from, so a
    moved/removed base interpreter shows up here as a silent death — the
    honest signal for "needs a rebuild" rather than "missing".
    """
    try:
        out = subprocess.run(
            [py, "-c", "import sys; print(sys.version.split()[0])"],
            capture_output=True, text=True, timeout=20.0,
            encoding="utf-8", errors="replace",
            env=_env_proc_env(), creationflags=_env_win_flags())
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def _env_probe(py: str) -> tuple[bool, str]:
    """Real faster-whisper import through the worker ('' or the error)."""
    try:
        from yache_engine import stt as _stt
        out = _stt._run_worker(py, ["--probe"], 150.0)
        return True, str(out.get("version") or "")
    except Exception as exc:  # noqa: BLE001
        return False, "%s: %s" % (type(exc).__name__, exc)


def _env_state() -> Dict[str, Any]:
    """Honest status of the plugin-owned ASR environment.

    States are ordered by what the panel can do about them:
      missing — no environment yet; the Install button creates one
      broken  — folder exists but its interpreter won't start (base
                Python moved, or an install was killed halfway); Rebuild
      deps    — interpreter starts, faster-whisper does not import; the
                same button tops the packages back up
      ready   — interpreter AND import both verified on this box
    """
    env_dir = _p.stt_env_dir()
    py = str(_p.stt_env_python())
    has_exe = Path(py).is_file()
    ver = _env_py_version(py) if has_exe else ""
    probe_ok, detail = (False, "")
    whisper = ""
    if not has_exe:
        state = "missing" if not env_dir.exists() else "broken"
    elif not ver:
        state = "broken"
    else:
        probe_ok, detail = _env_probe(py)
        state = "ready" if probe_ok else "deps"
        if probe_ok:
            whisper = detail
    # Which interpreter the runtime will actually pick — the first rung of
    # the documented ladder, no probing (liveness is /stt/state's job).
    try:
        chosen = str(_p.stt_python_candidates()[0])
    except Exception:  # noqa: BLE001
        chosen = py
    with _ENV_LOCK:
        running = dict(_ENV) if _ENV.get("state") == "running" else None
    if running is not None:
        running["log"] = list(running.get("log") or [])[-_ENV_TAIL:]
    return {
        "state": state,
        "env_dir": str(env_dir),
        "env_python": py,
        "env_version": ver,
        "whisper_version": whisper,
        "detail": detail if not probe_ok else "",
        "base_python": sys.executable,
        "in_use_python": chosen,
        "in_use_is_env": chosen == py,
        "packages": list(_ASR_PKGS),
        "free_gb": _env_free_gb(),
        "install": running,
    }


def _env_log(job: Dict[str, Any], line: str) -> None:
    line = str(line).strip()[:400]
    if not line:
        return
    with _ENV_LOCK:
        log = job.setdefault("log", [])
        log.append(line)
        if len(log) > 400:
            del log[:len(log) - 400]


def _env_step(job: Dict[str, Any], key: str) -> None:
    with _ENV_LOCK:
        job["step"] = (job.get("steps") or []).index(key)
        job["label"] = key


def _env_run(job: Dict[str, Any], cmd: list, label: str) -> subprocess.CompletedProcess:
    """Run one installer step, streaming its tail into the panel log."""
    _env_log(job, "$ " + " ".join(str(c) for c in cmd))
    try:
        # encoding pinned on purpose: pip/venv emit UTF-8 even on a
        # GBK-locale Windows box, where text mode alone takes the ANSI
        # codepage, the reader thread dies on an undecodable byte and
        # hands back stdout=None.
        out = subprocess.run([str(c) for c in cmd], capture_output=True, text=True,
                             timeout=_ENV_STEP_TIMEOUT.get(label, 300.0),
                             encoding="utf-8", errors="replace",
                             env=_env_proc_env(), creationflags=_env_win_flags())
    except subprocess.TimeoutExpired:
        raise RuntimeError(_i18n.ui("步骤超时（%s 秒）: %s",
                                    "Step timed out after %ss: %s")
                           % (_ENV_STEP_TIMEOUT.get(label, 300.0), label))
    except OSError as exc:
        raise RuntimeError(_i18n.ui("无法启动 %s: %s", "Could not start %s: %s") % (label, exc))
    keep = [l.strip() for l in ((out.stdout or "") + "\n" + (out.stderr or ""))
            .splitlines() if l.strip()]
    for line in keep[-10:]:
        _env_log(job, line)
    if out.returncode != 0:
        raise RuntimeError("%s (exit %s)" % (label, out.returncode))
    return out


def _env_rmtree(path, attempts: int = 8) -> None:
    """Delete a directory tree, tolerating Windows' delete-pending race.

    A file that some other handle still holds open keeps its parent directory
    "not empty" for a moment after deletion, so rmtree can raise WinError 145
    on a tree that really is going away — measured live on 2026-11-16 while two
    installs of the speech-to-text environment overlapped.  Scanners and
    antivirus also touch freshly written files, so a transient failure here is
    normal and must not block an install that would otherwise work: retry with
    a growing pause, then accept the result when the directory is gone or
    empty, and only report an error if real content is still left.
    """
    last: Exception = FileNotFoundError()
    for attempt in range(attempts):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return                      # already gone: nothing to wipe
        except OSError as exc:          # noqa: BLE001  (transient by nature)
            last = exc
            time.sleep(0.3 * (attempt + 1))
            if not os.path.isdir(path):
                return
    if not os.path.isdir(path):
        return
    try:
        if not os.listdir(path):        # emptied late: finish the job
            os.rmdir(path)
            return
    except OSError:
        pass
    raise last


def _env_worker(job: Dict[str, Any]) -> None:
    """Build/repair the environment, then pin the ladder onto it."""
    py = str(_p.stt_env_python())
    env_dir = _p.stt_env_dir()
    try:
        if "wipe" in job["steps"]:
            _env_step(job, "wipe")
            if env_dir.exists():
                _env_log(job, "removing %s" % env_dir)
                _env_rmtree(env_dir)   # retry-tolerant, see its docstring
        if "venv" in job["steps"]:
            _env_step(job, "venv")
            # stt_env_dir() *is* the venv (it already ends in .venv), so the
            # environment is created exactly where stt_env_python() looks for it.
            _env_run(job, [sys.executable, "-m", "venv", str(env_dir)], "venv")
            if not Path(py).is_file():
                raise RuntimeError(_i18n.ui("venv 建好了但找不到解释器: %s",
                                            "venv was created but its interpreter is missing: %s") % py)
            # Some slim runtimes ship without the bundled pip wheel; say so
            # and fix it with ensurepip instead of failing on a vague error.
            pip_check = subprocess.run([py, "-m", "pip", "-V"], capture_output=True,
                                       text=True, timeout=60.0,
                                       encoding="utf-8", errors="replace",
                                       env=_env_proc_env(),
                                       creationflags=_env_win_flags())
            if pip_check.returncode != 0:
                _env_log(job, _i18n.ui("这个 Python 不带 pip，用 ensurepip 补一个",
                                       "this Python ships without pip — adding it via ensurepip"))
                _env_run(job, [py, "-m", "ensurepip", "--upgrade"], "venv")
        if "pip" in job["steps"]:
            _env_step(job, "pip")
            base = [py, "-m", "pip", "install", "--no-input",
                    "--disable-pip-version-check"]
            try:
                _env_run(job, base + list(_ASR_PKGS), "pip")
                index = "user"
            except Exception as exc:  # noqa: BLE001
                # A mirror that simply does not carry the package looks
                # identical to "no such package at all"; retry ONCE on the
                # official index and say which one worked.
                _env_log(job, _i18n.ui("当前 pip 源里没有这些包，改用官方源重试: %s",
                                       "the configured pip index has nothing for these packages — retrying on the official index: %s")
                         % exc)
                _env_run(job, base + ["--index-url", _ASR_INDEX_FALLBACK] + list(_ASR_PKGS), "pip")
                index = "official"
            with _ENV_LOCK:
                job["index"] = index
        _env_step(job, "verify")
        ok, detail = _env_probe(py)
        if not ok:
            raise RuntimeError(_i18n.ui("装完了但引擎仍加载不起来: %s",
                                        "installed, but the engine still will not load: %s") % detail)
        # The user clicked this button precisely so this environment gets
        # used: pin the ladder to it, and drop any stale probe verdict so
        # the lamps refresh on the next look.
        data = _p.load_data()
        replaced = str(data.get("stt_python", "")).strip()
        data["stt_python"] = py
        _p.save_data(data)
        try:
            from yache_engine import stt as _stt
            _stt._probe_cache.pop(py, None)
        except Exception:  # noqa: BLE001
            pass
        with _ENV_LOCK:
            job.update(state="done", error="", replaced=replaced,
                       whisper_version=detail, finished_at=time.time())
            job["step"] = len(job.get("steps") or [])
        _env_log(job, _i18n.ui("环境就绪 (faster-whisper %s)", "environment ready (faster-whisper %s)") % detail)
    except Exception as exc:  # noqa: BLE001
        with _ENV_LOCK:
            job.update(state="error", error=str(exc), finished_at=time.time())
        _env_log(job, "ERROR: %s" % exc)


@router.get("/stt/env")
def stt_env_state() -> Dict[str, Any]:
    """ASR environment status (see _env_state for the state machine).

    Probes the plugin environment once per call (a real import, ~2-4 s),
    which is fine because the panel only asks when the dialog opens and
    after an install finishes.
    """
    try:
        return _env_state()
    except Exception as exc:  # noqa: BLE001
        return {"state": "error", "detail": "%s: %s" % (type(exc).__name__, exc),
                "env_dir": "", "env_python": "", "env_version": "",
                "whisper_version": "", "base_python": sys.executable,
                "in_use_python": "", "in_use_is_env": False,
                "packages": list(_ASR_PKGS), "free_gb": -1.0, "install": None}


class _EnvInstallIn(BaseModel):
    rebuild: bool = False   # wipe the folder first (fresh from scratch)


@router.post("/stt/env/install")
def stt_env_install(payload: Optional[_EnvInstallIn] = None) -> Dict[str, Any]:
    """Build or repair the plugin-owned ASR venv. The panel's only pip exit.

    Refuses to run a second installer over the same folder, and refuses
    upfront when the target disk is too small instead of dying halfway
    through a download.
    """
    with _ENV_LOCK:
        if _ENV.get("state") == "running":
            return {"ok": False, "state": "running",
                    "error": _i18n.ui("环境正在安装中，请等它装完再点",
                                      "An install is already running — wait for it to finish before clicking again")}
    st = _env_state()
    free = st.get("free_gb", -1.0)
    if 0 <= free < 1.5:
        return {"ok": False, "state": "error",
                "error": _i18n.ui("目标磁盘只剩 %s GB：环境加模型至少需要 1.5 GB。请先清出空间，或把模型库挪到大盘后重试",
                                  "Only %s GB free on the target disk: the environment plus models need at least 1.5 GB. Free up space, or point the model library at a bigger drive and try again")
                        % free}
    rebuild = bool(payload.rebuild) if payload is not None else False
    if rebuild or st.get("state") != "deps":
        steps = ["wipe", "venv", "pip", "verify"]
    else:
        steps = ["pip", "verify"]
    job = {"state": "running", "steps": steps, "step": 0, "label": steps[0],
           "log": [], "error": "", "index": "", "rebuild": rebuild,
           "started_at": time.time(), "finished_at": None, "replaced": "",
           "whisper_version": ""}
    with _ENV_LOCK:
        _ENV.clear()
        _ENV.update(job)
        # The worker publishes into this shared dict itself.  Handing it a
        # private snapshot instead froze the panel at 0% for the whole install
        # (only the log list was shared, so neither the step counter nor the
        # state ever moved) — caught live on 2026-11-16.
        published = _ENV
    threading.Thread(target=_env_worker, args=(published,), daemon=True,
                     name="yache-asr-env").start()
    return {"ok": True, "state": "running", "steps": steps}


@router.get("/stt/env/install/status")
def stt_env_install_status() -> Dict[str, Any]:
    """Poll the install job: which real step it is on, and its log tail.

    Progress is completed steps over total steps — no invented percentage
    from byte counts. The log tail carries the real pip lines so a failure
    is readable instead of a spinner that never ends.
    """
    with _ENV_LOCK:
        if not _ENV:
            return {"state": "idle"}
        job = dict(_ENV)
        job["log"] = list(job.get("log") or [])[-_ENV_TAIL:]
    steps = job.get("steps") or ["verify"]
    done = job.get("step", 0) if job.get("state") != "done" else len(steps)
    job["progress"] = round(100.0 * max(0, min(done, len(steps))) / max(len(steps), 1), 1)
    job["elapsed_s"] = round((job.get("finished_at") or time.time())
                             - job.get("started_at", time.time()), 1)
    return job


# ---------------------------------------------------------------------------
# Plugin-owned error trail: wrap every route endpoint so any 500 leaves a full
# traceback in <plugin>/logs/yache-plugin.log (host logs only show the one
# generic uvicorn line). The panel detail carries the exception class + message;
# the route name is logged so a screenshot alone pinpoints the failing handler.
def _wrap_endpoints() -> None:
    import functools
    import inspect

    def _sync(fn):
        @functools.wraps(fn)
        def inner(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except HTTPException:
                raise
            except Exception as exc:
                _log.exception("route %s failed", fn.__name__)
                raise HTTPException(
                    status_code=500,
                    detail=f"{fn.__name__}: {type(exc).__name__}: {exc} "
                           f"— see plugins/{_lx._NAME}/logs/yache-plugin.log")
        return inner

    def _async(fn):
        @functools.wraps(fn)
        async def inner(*args, **kwargs):
            try:
                return await fn(*args, **kwargs)
            except HTTPException:
                raise
            except Exception as exc:
                _log.exception("route %s failed", fn.__name__)
                raise HTTPException(
                    status_code=500,
                    detail=f"{fn.__name__}: {type(exc).__name__}: {exc} "
                           f"— see plugins/{_lx._NAME}/logs/yache-plugin.log")
        return inner

    for _route in router.routes:
        _ep = getattr(_route, "endpoint", None)
        if _ep is None or getattr(_ep, "_yache_wrapped", False):
            continue
        _w = _async(_ep) if inspect.iscoroutinefunction(_ep) else _sync(_ep)
        _w._yache_wrapped = True
        _route.endpoint = _w


try:
    _wrap_endpoints()
    _log.info("dashboard api loaded: %d routes wrapped", len(router.routes))
except Exception:
    _log.exception("dashboard api: endpoint wrapping failed")
