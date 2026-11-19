"""local-stt — plugin-ized faster-whisper STT provider (worker subprocess).

Survives official Hermes upgrades because all logic lives in the plugin
directory, not in the core tools/transcription_tools.py.

The model runs in a short-lived child process under an EXTERNAL venv
interpreter (see paths.stt_python_candidates), never inside the gateway:

  * no VRAM/RAM held between utterances (the worker exits immediately),
  * immunity to gateway environment rebuilds — the official dependency
    manager regenerates gateway environments from the stock lock, and
    faster-whisper is not in that lock, so an in-gateway install silently
    vanished after the 2026-11-14 rebuild ("STT plugin 'local-stt' is
    not available" incident),
  * no leaked PYTHONPATH hijacks of the gateway's own imports.

Public contract (signatures, result envelope, wake-gate handoff) is
identical to the previous in-process implementation.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from agent.transcription_provider import TranscriptionProvider

from .paths import asr_worker_script, load_data, stt_python_candidates

logger = logging.getLogger(__name__)

_PROBE_TTL_S = 120.0
_PROBE_FAIL_TTL_S = 20.0
_PROBE_TIMEOUT_S = 25.0
_TRANSCRIBE_TIMEOUT_S = 180.0

_WIN_CREATE_NO_WINDOW = 0x08000000  # GUI host: console children need this
_WIN_BELOW_NORMAL = 0x00004000

_probe_cache: Dict[str, tuple] = {}
_probe_lock = threading.Lock()


def _win_flags() -> int:
    return _WIN_CREATE_NO_WINDOW | _WIN_BELOW_NORMAL if os.name == "nt" else 0


def _worker_env(online: bool = False) -> dict:
    """Sanitized env: a PYTHONPATH leaked by the gateway (harness
    site-packages) would shadow the worker venv's own packages; forcing
    UTF-8 stdio keeps the JSON pipe codepage-independent.

    Offline-by-default is the deliberate policy: the host injects
    HF_HUB_OFFLINE=1 and every normal transcription/verify only ever
    loads from the local cache. The panel's Download button is the one
    single exit to the network (online=True lifts that flag just for
    that spawn), so download failures are unambiguously network
    failures and nothing else can reach the network behind the user's
    back."""
    env = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME"):
        env.pop(key, None)
    env["PYTHONIOENCODING"] = "utf-8"
    # Quarantine a classic desktop-config disease: values stored WITH
    # literal enclosing double quotes (e.g. HF_ENDPOINT set to
    # '"https://hf-mirror.com"' via a quoting slip). Offline mode never
    # parses these and stays silent for years; the first online spawn
    # then dies on httpx's "Request URL is missing an http:// or
    # https:// protocol" and users blame the VPN. We repair only
    # fully-wrapped values (a quote inside a value is left alone) and
    # only for the URL-shaped keys that break this way.
    for key in ("HF_ENDPOINT", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
                "WSS_PROXY"):
        val = env.get(key)
        if (val and len(val) >= 2 and val[0] == val[-1]
                and val[0] in "\"'"):
            env[key] = val[1:-1].strip()
    # Same config fallback everywhere else in this plugin uses: when the
    # gateway environment never learned where the user pointed the model
    # library, the worker must still download INTO that library, not the
    # per-user default — otherwise the cache scan keeps missing every
    # "successful" download.
    if not env.get("HF_HOME") and not env.get("HF_HUB_CACHE"):
        cfg_home = _official_hf_home()
        if cfg_home:
            env["HF_HOME"] = cfg_home
    if online:
        env["HF_HUB_OFFLINE"] = "0"
    return env


def _official_hf_home() -> str:
    """The desktop app stores the model library wherever the USER chose
    it (official config huggingface_hub.HF_HOME). The gateway process
    may never carry that in its own environment; one shared reader
    keeps worker, cache scan and progress probe pointed at the SAME
    disk."""
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        home = str(((cfg.get("huggingface_hub") or {})
                    .get("HF_HOME")) or "").strip()
        if home:
            return home
    except Exception:
        pass
    return ""


# ---- live download progress (panel green bar) ------------------------
# huggingface_hub 2.x offers no public progress callback, so we measure
# the download the honest way: HF stages every file as a growing
# <blob>.incomplete inside the repo's blobs dir. Summing those bytes
# (plus already-finished blobs) against the repo's declared file sizes
# yields a truthful percentage without touching the engine at all.
_DL_TOTAL_CACHE: dict[str, int] = {}


def _hub_root() -> Path:
    hub = os.environ.get("HF_HUB_CACHE")
    if hub:
        return Path(hub)
    home = os.environ.get("HF_HOME") or _official_hf_home()
    if home:
        return Path(home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _hf_repo_for(model: str) -> str:
    """faster-whisper's repo mapping is literal concatenation — "small.en"
    is its OWN repo (Systran/faster-whisper-small.en), not an alias; on-disk
    cache dirs proved it."""
    return "Systran/faster-whisper-" + model


def download_progress(model: str, fallback_gb: float = 0.0) -> dict:
    """Bytes-based snapshot of an in-flight model download. The declared
    total comes from model_info when the network cooperates, else from
    the caller's advertised size (the caller caps percent below 100
    while running, so an underestimated total can never fake a
    finished download)."""
    repo = _hf_repo_for(model)
    blobs = _hub_root() / ("models--" + repo.replace("/", "--")) / "blobs"
    have = 0
    if blobs.is_dir():
        try:
            for p in blobs.iterdir():
                if (p.is_file()
                        and not p.name.endswith((".lock", ".metadata"))):
                    try:
                        have += p.stat().st_size
                    except OSError:
                        pass
        except OSError:
            pass
    total = _DL_TOTAL_CACHE.get(repo, 0)
    if not total:
        try:
            from huggingface_hub import model_info
            info = model_info(repo, files_metadata=True, etag_timeout=8)
            total = sum(int(getattr(s, "size", 0) or 0)
                        for s in (getattr(info, "siblings", None) or []))
            if total > 0:
                _DL_TOTAL_CACHE[repo] = total
        except Exception:
            total = 0
    if not total:
        total = int(max(float(fallback_gb), 0.0) * 1_000_000_000 * 1.05)
    return {"have_bytes": int(have), "total_bytes": max(int(total), 1)}


def _run_worker(python: str, args: list, timeout: float,
                online: bool = False) -> dict:
    """Run one worker invocation; return the parsed final-line JSON.

    Raises RuntimeError on spawn/timeout/protocol failure.
    ``online=True`` is reserved for the panel's Download button (the
    only sanctioned network exit; see _worker_env).
    """
    cmd = [python, "-I", str(asr_worker_script())] + list(args)
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=_worker_env(online),
            creationflags=_win_flags(),
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"ASR worker timed out after {timeout:.0f}s")
    except OSError as exc:
        raise RuntimeError(f"ASR worker spawn failed: {exc}")

    payload = None
    for line in reversed((proc.stdout or "").splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                payload = json.loads(line)
            except ValueError:
                payload = None
            break
    if not isinstance(payload, dict):
        tail = " | ".join((proc.stderr or "").splitlines()[-3:])
        raise RuntimeError(tail or f"ASR worker exited with code {proc.returncode}")
    if payload.get("error"):
        raise RuntimeError(str(payload["error"]))
    return payload


def _probe(python: str) -> bool:
    """Cached live probe: can THIS interpreter import faster_whisper?"""
    now = time.time()
    with _probe_lock:
        hit = _probe_cache.get(python)
        if hit and now - hit[1] < (_PROBE_TTL_S if hit[0] else _PROBE_FAIL_TTL_S):
            return hit[0]
    ok = False
    try:
        ok = bool(_run_worker(python, ["--probe"], _PROBE_TIMEOUT_S).get("ok"))
    except Exception:
        ok = False
    with _probe_lock:
        _probe_cache[python] = (ok, now)
    return ok


def _dead_interpreter(err: str) -> bool:
    """Worker error meaning 'this interpreter lacks the ASR stack'."""
    low = err.lower()
    return "no module named" in low or "modulenotfound" in low or "importerror" in low


class LocalSTTProvider(TranscriptionProvider):
    """Plugin-ized local faster-whisper transcription provider."""

    @property
    def name(self) -> str:
        return "local-stt"

    @property
    def display_name(self) -> str:
        return "Local STT (faster-whisper)"

    def is_available(self) -> bool:
        """Live probe of the worker-interpreter ladder (TTL-cached)."""
        for py in stt_python_candidates():
            python = str(py)
            if not os.path.exists(python):
                continue
            if _probe(python):
                return True
        return False

    def transcribe(
        self,
        file_path: str,
        *,
        model: Optional[str] = None,
        language: Optional[str] = None,
        **extra: Any,
    ) -> Dict[str, Any]:
        """Transcribe ``file_path`` via the faster-whisper worker.

        Tries each configured interpreter in order; one whose worker
        reports a missing dependency is marked probe-dead (short TTL) so
        the ladder self-heals after an environment rebuild.
        """
        try:
            args = ["--wav", file_path, "--model", model or "small"]
            if language:
                args += ["--language", str(language)]
            if extra.get("prompt"):
                args += ["--prompt", str(extra["prompt"])]
            # Device is a plugin-owned knob (the host's stt config has no
            # such key): yache-voice.json stt.device, set by the settings
            # panel behind its real-load gate. "auto" stays unwritten —
            # the worker decides (cuda first, fall back to cpu).
            try:
                dev = str(((load_data().get("stt") or {}).get("device"))
                          or "").strip().lower()
            except Exception:
                dev = ""
            if dev in ("cpu", "cuda"):
                args += ["--device", dev]

            last_error = "no usable ASR interpreter configured"
            payload = None
            for py in stt_python_candidates():
                python = str(py)
                if not os.path.exists(python):
                    continue
                try:
                    payload = _run_worker(python, args, _TRANSCRIBE_TIMEOUT_S)
                    break
                except RuntimeError as exc:
                    last_error = str(exc)
                    if _dead_interpreter(last_error):
                        with _probe_lock:
                            _probe_cache[python] = (False, time.time())
                    logger.warning("[local-stt] worker %s failed: %s", python, last_error)
            if payload is None:
                raise RuntimeError(last_error)

            transcript = str(payload.get("transcript") or "").strip()
            logger.info("[local-stt] transcribed %s (%.1fs audio, lang=%s)",
                        Path(file_path).name,
                        float(payload.get("duration") or 0.0),
                        payload.get("language"))

            # ── Wake-word half-duplex gate (same design as the playback
            # side; default OFF via the wake_gate switch) ──
            # The moment the transcript is about to post into the chat,
            # the detector goes on vacation: it covers the gap between
            # posting, thinking, and the first TTS chunk (the stock
            # semantics of "wake switch turns itself off once the spoken
            # sentence has been sent"). The playback gate in tts.py then
            # takes over seamlessly and resumes after the last sentence;
            # for a text-only reply with no playback, the watchdog
            # re-arms on timeout. Borrow, never seize — see tts.py.
            try:
                try:
                    from yache_engine.tts import _wake_gate_busy
                except ImportError:
                    from .tts import _wake_gate_busy
                # Long reasoning on thinking models is covered too; once
                # TTS starts the gate is handed over, no re-wake needed.
                _wake_gate_busy(time.monotonic() + 180.0)
            except Exception:
                logger.debug("[local-stt] wake gate handoff skipped", exc_info=True)

            return {
                "success": True,
                "transcript": transcript,
                "provider": self.name,
            }
        except Exception as exc:
            logger.error("[local-stt] transcription failed: %s", exc, exc_info=True)
            return {
                "success": False,
                "transcript": "",
                "provider": self.name,
                "error": str(exc),
            }


def register(ctx) -> None:
    """Plugin entry point — register the local-stt transcription provider."""
    ctx.register_transcription_provider(LocalSTTProvider())
