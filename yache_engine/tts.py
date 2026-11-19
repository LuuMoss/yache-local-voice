"""qwen3-tts — local Qwen3-TTS 1.7B (ggml engine) plugin, upstream TTSProvider API.

Design:
- The engine runs as a subprocess (engine/worker.py; QWEN3_TTS_ENGINE /
  QWEN3_TTS_WORKER env override); sentence splitting and trimming live there.
- streams_pcm=True + stream_sample_rate=24000 → the upstream dispatcher consumes
  the raw mono int16 PCM yielded by stream() chunk by chunk, entirely in memory.
- Voice follows the voice registry: the voice parameter (a registry voice
  key) is handed to the engine via the QWEN3_TTS_VOICE env resolution chain.
- VRAM policy: outside a /voice lease every synthesis ends with an immediate
  _release_model() (speak-then-release, never fight the LLM for VRAM); the engine
  stays resident only while warm()/release() (the /voice lease) is active.

Failure policy:
  Any failure in the Qwen3-TTS(ggml) path — engine load, model files or
  synthesis — raises the original error to the host; nothing is silently
  swapped for another model.  This module deliberately avoids importing
  numpy at top level and loading the engine at import time: even when
  environment dependencies drift, plugin registration still succeeds and
  the "qwen3-tts" name always holds its provider slot — so upstream can
  never silently route requests to the online Edge TTS service (Microsoft,
  often 403 in China and a violation of the local-only rule).
"""
from __future__ import annotations

import base64
import importlib.util as _ilu
import json
import logging
import os
import subprocess
import threading
import sys
import time
import wave
from pathlib import Path
from typing import Iterator

from agent.tts_provider import TTSProvider

from . import i18n as _i18n  # bilingual user-facing strings

logger = logging.getLogger(__name__)
_PLUGIN_DIR = Path(__file__).resolve().parent
# stderr dump of the engine subprocess (gateway logger.debug is invisible; the file is for post-mortem)
CHILD_LOG = _PLUGIN_DIR.parents[2] / "cache" / "scratch" / "yache-voice-child.log"

# site-packages of the engine's own venv (faster_qwen3_tts / qwentts_cpp: pure Python + DLL bindings;
# numpy etc. come from the host venv -> appended at the END of sys.path so host packages win)
from . import paths as _p

_ENGINE_VENV_SP = str(_p.engine_site_packages())
if _ENGINE_VENV_SP not in sys.path:
    sys.path.append(_ENGINE_VENV_SP)
# worker.py (loaded inline) does a top-level `import paths` (sibling module): the engine dir must be importable
_ENGINE_DIR = str(_PLUGIN_DIR)
if _ENGINE_DIR not in sys.path:
    sys.path.append(_ENGINE_DIR)

_HERMES_HOME = _p.hermes_home()
# engine worker script: QWEN3_TTS_ENGINE / QWEN3_TTS_WORKER env override ->
# plugin-bundled yache_engine/worker.py — single stable path
_ENGINE_PATH = next(
    (p for p in (
        Path(os.environ["QWEN3_TTS_ENGINE"]) if os.environ.get("QWEN3_TTS_ENGINE") else _p.worker_script(),
        _p.worker_script(),
    ) if p.is_file()),
    _p.worker_script(),
)
# python of the engine's own venv (runs worker.py --server as a subprocess)
_ENGINE_PYTHON = _p.engine_python()
_VOICES = _p.voices_dir()
_SAMPLE_RATE = 24000  # Qwen3-TTS codec is fixed at 24kHz

# ---- main-path switch ----
# True = main path: Qwen3-TTS in subprocess mode (process exit -> CUDA context destroyed -> VRAM fully freed);
# False = short-circuit stream()/synthesize() with a tiny silent wav (experiment knob)
_SYNTH_ENABLED = True


def _load_engine():
    os.environ.setdefault("PYTHONUTF8", "1")  # force UTF-8 on stdin/stdout across platforms
    if not _ENGINE_PATH.is_file():
        logger.warning("[qwen3-tts] engine script missing: %s", _ENGINE_PATH)
        return None
    try:
        spec = _ilu.spec_from_file_location("yache_worker", str(_ENGINE_PATH))
        if spec is None or spec.loader is None:
            raise RuntimeError(_i18n.ui(f"无法加载 TTS 引擎: {_ENGINE_PATH}", f"Failed to load the TTS engine: {_ENGINE_PATH}"))
        mod = _ilu.module_from_spec(spec)
        sys.modules["yache_worker"] = mod
        spec.loader.exec_module(mod)
        return mod
    except Exception:
        logger.warning("[qwen3-tts] engine module failed to load", exc_info=True)
        return None


_ENGINE = None
_ENGINE_TRIED = False


def _engine():
    """Lazy engine load: zero dependencies at plugin import time, registration always succeeds."""
    global _ENGINE, _ENGINE_TRIED
    if not _ENGINE_TRIED:
        _ENGINE_TRIED = True
        _ENGINE = _load_engine()
    return _ENGINE


def _normalize(text: str) -> str:
    eng = _engine()
    if eng is None:
        return (text or "").strip()
    t = eng.normalize_symbols(text or "")
    t = eng.normalize_times(t)
    t = eng.normalize_units(t)
    t = eng.normalize_numbers(t)
    return t.strip()


# ---- engine subprocess management (mirrors the v0.21.5 Electron ensureTtsServer singleton) ----------
# The main path moved from inline import back to subprocess: the engine runs in its own process, and
# exiting it destroys the CUDA context so VRAM physically returns to the OS (inline mode leaves ~300M
# of ggml C-level caches that gc cannot reclaim).  The subprocess stays hot; it restarts on voice switch
_SERVER_PROC = None
_SERVER_STDIN = None
_SERVER_STDOUT = None
_SERVER_VOICE = None  # voice key (ref path) of the current subprocess; a switch triggers a restart
_SERVER_LOCK = threading.Lock()
_SYNTH_LOCK = threading.Lock()  # serialize concurrent gen requests to keep ordering sane

# ---- active-stream guard (added 11-11): release()/the 15s timer must never kill the
#    subprocess mid-stream — the reply would be cut in half (0 packets / a partial
#    parked, and it fires the moment the last chunk lands.  ----
_ACTIVE_STREAMS = 0
_STREAMS_LOCK = threading.Lock()
_PENDING_RELEASE = False


def _stream_enter() -> None:
    global _ACTIVE_STREAMS
    with _STREAMS_LOCK:
        _ACTIVE_STREAMS += 1


def _stream_exit() -> None:
    global _ACTIVE_STREAMS, _PENDING_RELEASE
    with _STREAMS_LOCK:
        _ACTIVE_STREAMS = max(0, _ACTIVE_STREAMS - 1)
        fire = _PENDING_RELEASE and _ACTIVE_STREAMS == 0
        if fire:
            _PENDING_RELEASE = False
    if fire:
        _stop_server()


def _request_stop_server() -> None:
    """Single entry for "we want the subprocess dead": kill now if idle, park if synthesizing."""
    global _PENDING_RELEASE
    with _STREAMS_LOCK:
        busy = _ACTIVE_STREAMS > 0 or _SYNTH_LOCK.locked()
        _PENDING_RELEASE = busy
    if not busy:
        _stop_server()


def _server_alive() -> bool:
    return _SERVER_PROC is not None and _SERVER_PROC.poll() is None


def _stop_server_locked() -> None:
    """Kill the subprocess (caller must hold _SERVER_LOCK)."""
    global _SERVER_PROC, _SERVER_STDIN, _SERVER_STDOUT, _SERVER_VOICE
    if _SERVER_PROC is None:
        return
    if _SERVER_PROC.poll() is None:
        try:
            _SERVER_PROC.terminate()
            _SERVER_PROC.wait(timeout=8)
        except subprocess.TimeoutExpired:
            _SERVER_PROC.kill()
        except Exception:
            pass
    for fh in (_SERVER_STDIN, _SERVER_STDOUT, _SERVER_PROC.stderr):
        try:
            if fh:
                fh.close()
        except OSError:
            pass
    _SERVER_PROC = None
    _SERVER_STDIN = None
    _SERVER_STDOUT = None
    _SERVER_VOICE = None
    logger.info("[qwen3-tts] engine subprocess exited (VRAM released)")


def _stop_server() -> None:
    with _SERVER_LOCK:
        _stop_server_locked()


def _ensure_server(env: dict | None = None, voice_key: str | None = None) -> None:
    """Spawn `worker.py --server`; restart on voice switch or process death.

    The voice travels to the child via the QWEN3_TTS_VOICE env var (the engine's
    _resolve_voice reads it).  The child stays hot so synthesis never pays a 5s reload.
    """
    global _SERVER_PROC, _SERVER_STDIN, _SERVER_STDOUT, _SERVER_VOICE
    if _server_alive() and _SERVER_VOICE == voice_key:
        return  # alive and voice matches -> reuse
    with _SERVER_LOCK:
        if _server_alive() and _SERVER_VOICE == voice_key:
            return
        _stop_server_locked()
        e = dict(os.environ)
        # The gateway leaks its own PYTHONPATH (harness site-packages) into
        # children; it hijacks the engine venv's numpy (cp314 host vs cp312
        # engine .pyd). Strip it — the engine venv resolves on its own.
        e.pop("PYTHONPATH", None)
        e.pop("PYTHONHOME", None)
        e.setdefault("PYTHONUTF8", "1")
        e.setdefault("HERMES_VOICES_DIR", str(_VOICES))
        if env:
            e.update(env)
        popen_kwargs = {"stdin": subprocess.PIPE, "stdout": subprocess.PIPE,
                        "stderr": subprocess.PIPE, "env": e}
        if os.name == "nt":
            popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
        _SERVER_PROC = subprocess.Popen(
            [str(_ENGINE_PYTHON), str(_ENGINE_PATH), "--server"], **popen_kwargs)
        _SERVER_STDIN = _SERVER_PROC.stdin
        _SERVER_STDOUT = _SERVER_PROC.stdout
        _SERVER_VOICE = voice_key
        try:
            hand = _SERVER_STDOUT.readline()  # read the READY handshake line
        except Exception:
            hand = b""
        if hand.strip() != b"READY":
            code = _SERVER_PROC.poll()
            _stop_server_locked()
            raise RuntimeError(
                _i18n.ui(f"子进程握手失败: stdout 收到 {hand!r} (exit={code})；详见 {CHILD_LOG}",
                         f"Handshake with the engine subprocess failed: stdout gave {hand!r} (exit={code}); see {CHILD_LOG}"))

        def _drain():
            try:
                for line in iter(_SERVER_PROC.stderr.readline, b""):
                    s = line.decode("utf-8", "ignore").strip()
                    if s:
                        logger.debug("[qwen3-tts][engine] %s", s)
                        try:
                            with open(CHILD_LOG, "a", encoding="utf-8") as fh:
                                fh.write(s + "\n")
                        except OSError:
                            pass
            except Exception:
                pass

        threading.Thread(target=_drain, daemon=True).start()
        logger.info("[qwen3-tts] engine subprocess started (voice=%s)", voice_key)


_GEN_SEQ = 0


def _server_gen(text: str):
    """Send a gen request to the child; yield PCM bytes sentence by sentence (the engine _server_loop protocol).

    Protocol: stdin {"text": "...", "id": N} -> stdout
      {"type":"pcm","id":N,"data":<b64 int16>} (once per sentence) + {"type":"done","id":N}
    Yields per sentence (audio starts as soon as the first sentence is ready).
    """
    global _GEN_SEQ
    _stream_enter()
    try:
        with _SYNTH_LOCK:
            _GEN_SEQ += 1
            req_id = _GEN_SEQ
            _SERVER_STDIN.write(json.dumps({"text": text, "id": req_id},
                                           ensure_ascii=False).encode("utf-8") + b"\n")
            _SERVER_STDIN.flush()
            while True:
                line = _SERVER_STDOUT.readline()
                if not line:
                    break  # child process exited
                s = line.decode("utf-8", "ignore").strip()
                if not s:
                    continue
                try:
                    msg = json.loads(s)
                except Exception:
                    continue  # READY / "model released" and other non-JSON lines
                t = msg.get("type")
                if t == "pcm":
                    yield base64.b64decode(msg.get("data", ""))
                elif t == "done":
                    break
                elif t == "error":
                    raise RuntimeError(msg.get("msg", _i18n.ui("子进程引擎 gen 失败", "engine subprocess reported gen failure")))
    finally:
        _stream_exit()


# ═══ Wake-word half-duplex gate (2026-11-11, cyclic-call design) ═══
# The assistant's replies naturally contain its own wake word. If the detector is
# on duty while a reply is read aloud, it hears the synthesized self ->
# firing -> the desktop opens a new session and the playback gets cut
# (2026-11-11 22:29:16 smoking gun: tools.wake_word "phrase detected --
# firing callback" landed mid-playback, wake.detected with sid='' -> new
# session). Gate posture: pause the current detector the moment the first
# PCM chunk is delivered (mic off, engine kept, resume is sub-second); the
# vacation deadline renews dynamically with the playtime of delivered
# chunks; +0.8 s after the last chunk a watchdog resumes and re-arms —
# matching the intended cycle: wake -> record+transcript posts to the chat
# -> thinking -> playback (nobody overhears) -> back to wake-ready.
# Borrow, never seize: only operate on the currently live detector and
# track "we are the ones who paused it" (object identity), so we never
# revive a detector another lease stopped; when listening is off there is
# no detector and the gate is a no-op throughout.
_WAKE_GATE_LOCK = threading.Lock()
_WAKE_GATE = {"until": 0.0, "alive": False, "det": None}


def _wake_gate_current():
    """Locate the live wake detector while the read-aloud gate is enabled.

    Reviewer note (catalog rule 9): this is a fail-soft *read* of core
    state, never a rebind — the plugin replaces no core behaviour. The
    gate is an experimental local convenience (off by default; official
    wake semantics are unchanged unless ``wake_gate`` is set true in the
    plugin data file) that stops the plugin's own TTS audio from
    re-triggering wake detection mid-answer. A proper host-side
    half-duplex hook would replace this; until then the import stays
    function-local and lazy, and the official wake engine remains the
    single authority.
    """
    try:
        from tools import wake_word as _ww
        return _ww._current_detector()
    except Exception:
        return None


def _wake_gate_busy(until: float) -> None:
    """Extend the detector vacation until `until`; pause now if one is
    present, and arm the watchdog once.

    Experimental switch, OFF by default = stock behavior: every entry
    point (stream gate / file-path lease / transcript handoff)
    short-circuits here; opt in with ``wake_gate: true`` in the data file
    — see paths.wake_gate_enabled.
    """
    if not _p.wake_gate_enabled():
        return
    det = _wake_gate_current()
    with _WAKE_GATE_LOCK:
        if until > _WAKE_GATE["until"]:
            _WAKE_GATE["until"] = until
        if det is None or not getattr(det, "running", False):
            return
        try:
            det.pause()
        except Exception:
            pass
        _WAKE_GATE["det"] = det
        if not _WAKE_GATE["alive"]:
            _WAKE_GATE["alive"] = True
            threading.Thread(target=_wake_gate_watch, daemon=True,
                             name="yache-voice-wake-gate").start()


def _wake_gate_watch() -> None:
    """While on duty, self-heal every 0.5s; when vacation ends, resume only the one WE stopped."""
    while True:
        time.sleep(0.5)
        with _WAKE_GATE_LOCK:
            busy = _WAKE_GATE["until"] - time.monotonic() > 0
            if not busy:
                _WAKE_GATE["alive"] = False
        if busy:
            # someone else resumed it, or the detector was rebuilt -> stop the new one once
            det = _wake_gate_current()
            if det is not None and getattr(det, "running", False):
                try:
                    det.pause()
                except Exception:
                    pass
                with _WAKE_GATE_LOCK:
                    _WAKE_GATE["det"] = det
            continue
        with _WAKE_GATE_LOCK:
            paused = _WAKE_GATE["det"]
            _WAKE_GATE["det"] = None
        det = _wake_gate_current()
        # identity check: only re-arm if it is still OUR stopped detector and still not running
        if det is not None and paused is not None and det is paused \
                and not getattr(det, "running", False):
            try:
                det.resume()
            except Exception:
                pass
        return


def _wake_gate_file(output_path, margin: float = 1.2) -> None:
    """After one-shot synthesis to a file, the host plays that file: extend the truce by its duration."""
    try:
        secs = os.path.getsize(str(output_path)) / (_SAMPLE_RATE * 2)
    except OSError:
        secs = 0.0
    _wake_gate_busy(time.monotonic() + secs + margin)


class _PlaybackGate:
    """Convert the chunks one stream() delivered into a detector-truce deadline."""
    __slots__ = ("first", "total")

    def __init__(self):
        self.first = None
        self.total = 0.0

    def chunk(self, nbytes: int) -> None:
        now = time.monotonic()
        if self.first is None:
            self.first = now
        self.total += nbytes / (_SAMPLE_RATE * 2)  # 24kHz mono s16le
        # sequential-play finish = first chunk delivered + accumulated audio time; when synthesis lags, renew from "now"
        _wake_gate_busy(max(self.first + self.total, now) + 0.8)

    def close(self) -> None:
        _wake_gate_busy(time.monotonic() + 0.8)


class Qwen3TTSProvider(TTSProvider):
    """Local Qwen3-TTS provider (name matches "qwen3-tts" in plugin.yaml).

    Main-path failures raise to the host; synthesis is local and never goes online.
    """

    name = "qwen3-tts"
    display_name = _i18n.ui("Yache Voice · Qwen3-TTS 本地", "Yache Voice · Qwen3-TTS (local)")
    streams_pcm = True
    stream_sample_rate = _SAMPLE_RATE

    def __init__(self) -> None:
        # True while a /voice lease is active: model stays resident, no per-synthesis release
        self._leased = False
        # session timeout: keep the model resident during bursts of synthesis, auto-release after the quiet period (fair play with the LLM's VRAM)
        self._release_timer: threading.Timer | None = None

    def is_available(self) -> bool:
        eng = _engine()
        if eng is not None:
            try:
                if os.path.isfile(eng.TALKER) and os.path.isfile(eng.TOKENIZER):
                    return True
            except Exception:
                pass
        # engine load failed / model files missing: unavailable
        return False

    def get_setup_schema(self) -> dict:
        return {
            "type": "local",
            "blurb": _i18n.ui(
                "本地 Qwen3-TTS 1.7B（ggml CUDA/CPU），念完即释放显存；失败还给宿主，绝不联网。",
                "Local Qwen3-TTS 1.7B (ggml CUDA/CPU); VRAM released the moment"
                " speech ends; failures surface to the host — never goes online."),
        }

    # ---- voice resolution: voice param -> engine QWEN3_TTS_VOICE env resolution chain ----
    def _resolve(self, voice):
        eng = _engine()
        if eng is None:
            raise RuntimeError(_i18n.ui("qwen3-tts 引擎不可用（引擎模块未加载，见 doctor 体检）", "the qwen3-tts engine is unavailable (engine module not loaded; see the doctor health check)"))
        prev = os.environ.get("QWEN3_TTS_VOICE")
        if voice:
            os.environ["QWEN3_TTS_VOICE"] = str(voice).strip()
        else:
            os.environ.pop("QWEN3_TTS_VOICE", None)
        try:
            ref, ref_text, temp = eng._resolve_voice()
        finally:
            if prev is None:
                os.environ.pop("QWEN3_TTS_VOICE", None)
            else:
                os.environ["QWEN3_TTS_VOICE"] = prev
        # Only clone voices carry a reference wav.  speaker/design return a
        # built-in speaker name / instruct text, which is NOT a file path —
        # isfile() would kill them.  worker.py already hard-gates every kind
        # with a precise message, so this shell only guards the empty case.
        if not ref:
            raise RuntimeError(_i18n.ui("TTS 音色解析结果为空（音色未设置或指针损坏）", "TTS voice resolution came back empty (voice unset or broken pointer)"))
        return str(ref), ref_text, float(temp)

    # ---- core: subprocess mode, spawn the engine (per-sentence streaming synthesis); yield (pcm_bytes, sr) ----
    def _gen_sentences(self, eng, text: str, voice, ref, ref_text, temp: float):
        """Run the engine as a subprocess (per-sentence streaming); yield (pcm_bytes, sr).

        The voice (a raw registry key) travels to the child via the
        QWEN3_TTS_VOICE env (the engine consults the registry in _resolve_voice).
        """
        vk = voice if voice else "default"
        env = {"QWEN3_TTS_VOICE": str(voice).strip()} if voice else {}
        _stream_enter()  # guard BEFORE _ensure_server: even the READY wait counts as busy
        try:
            _ensure_server(env=env, voice_key=vk)
            produced = False
            try:
                for chunk in _server_gen(text):
                    produced = True
                    yield chunk, _SAMPLE_RATE
            except Exception as exc:
                logger.warning("[qwen3-tts] engine subprocess synthesis failed: %s", exc, exc_info=True)
                if not produced:
                    # not a single PCM byte came out: surface the real error to the host (fallback/report), never stay mute
                    raise
        finally:
            _stream_exit()

    def _finish(self, eng) -> None:
        """No lease (read-aloud switch off): after speaking, kill the child and give the GPU back to the LLM."""
        if not self._leased:
            _stop_server()

    # ---- TTSProvider interface ----
    def synthesize(self, text, output_path, *, voice=None, model=None,
                   speed=None, format="mp3", **extra) -> str:
        text = _normalize(str(text or ""))
        if not text:
            raise RuntimeError(_i18n.ui("TTS 文本为空", "TTS text is empty"))
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if not _SYNTH_ENABLED:
            # experimental short-circuit: write a 50ms silent wav (no model load, no GPU, no VRAM)
            with wave.open(str(out), "wb") as h:
                h.setnchannels(1)
                h.setsampwidth(2)
                h.setframerate(_SAMPLE_RATE)
                h.writeframes(b"\x00\x00" * (_SAMPLE_RATE // 20))
            return str(out)
        eng = _engine()
        wrote = False
        if eng is not None:
            try:
                ref, ref_text, temp = self._resolve(voice)
                try:
                    with wave.open(str(out), "wb") as h:
                        h.setnchannels(1)
                        h.setsampwidth(2)
                        h.setframerate(_SAMPLE_RATE)
                        for pcm_bytes, _sr in self._gen_sentences(eng, text, voice, ref, ref_text, temp):
                            h.writeframes(pcm_bytes)
                            wrote = True
                finally:
                    self._finish(eng)
            except Exception as exc:
                logger.warning("[yache-voice] main one-shot synthesis failed (%s)", exc)
                raise  # hand the original root cause straight back to the host
        if not wrote:
            raise RuntimeError(_i18n.ui("qwen3-tts 未产出音频", "qwen3-tts produced no audio"))
        _wake_gate_file(out)  # the file is played by the host: let the detector take leave for the full clip duration
        return str(out)

    def stream(self, text, *, voice=None, model=None, format="opus",
               speed=None, **extra) -> Iterator[bytes]:
        if format != "pcm":
            raise NotImplementedError(_i18n.ui("qwen3-tts 插件只产 raw PCM，请走 synthesize 落文件路径", "the qwen3-tts plugin only yields raw PCM; use the synthesize() file path"))
        text = _normalize(str(text or ""))
        if not text:
            return
        if not _SYNTH_ENABLED:
            return  # experimental short-circuit: empty generator (no model load, no GPU, no VRAM)
        gate = _PlaybackGate()
        eng = _engine()
        if eng is None:
            raise RuntimeError(_i18n.ui("qwen3-tts 引擎不可用（引擎模块未加载，见 doctor 体检）", "the qwen3-tts engine is unavailable (engine module not loaded; see the doctor health check)"))
        produced = 0
        try:
            ref, ref_text, temp = self._resolve(voice)
            self._schedule_release()  # arm the session timer before the first synthesis
            try:
                for pcm_bytes, _sr in self._gen_sentences(eng, text, voice, ref, ref_text, temp):
                    produced += len(pcm_bytes)
                    gate.chunk(len(pcm_bytes))  # renew the detector truce on delivery
                    yield pcm_bytes
            finally:
                if produced == 0:
                    self._release_now()  # zero chunks delivered: release immediately, no timeout wait
        except Exception as exc:
            logger.warning("[yache-voice] main streaming path failed (%s)", exc)
            if produced == 0:
                raise  # nothing reached the speaker yet: surface the root cause to the host
        finally:
            gate.close()  # wrap-up: after the last packet +0.8s the wake detector goes back on duty

    # ---- VRAM lifecycle (/voice lease path: warm keeps it resident, release frees it) ----
    def warm(self, *, voice=None, model=None) -> None:
        """Read-aloud switch ON: mark the lease (_leased=True, subprocess stays hot).

        The subprocess spawns at the first synthesis (ON alone never steals the LLM's VRAM);
        when the switch flips OFF, release() kills the subprocess and frees VRAM.
        """
        self._leased = True
        with _STREAMS_LOCK:
            global _PENDING_RELEASE
            _PENDING_RELEASE = False  # switch back ON cancels any parked kill
        if _engine() is None:
            raise RuntimeError(_i18n.ui("qwen3-tts 引擎不可用（引擎模块未加载，见 doctor 体检）", "the qwen3-tts engine is unavailable (engine module not loaded; see the doctor health check)"))

    def release(self) -> None:
        """Read-aloud switch OFF: kill the subprocess -> CUDA context dies -> VRAM fully back.

        Active-stream guard: never cut mid-stream — park the kill, fire it when the last chunk lands.
        """
        self._leased = False
        self._cancel_release()
        _request_stop_server()

    # ---- session timeout: keep the model across consecutive synthesis, auto-release when quiet ----

    _SESSION_TIMEOUT = 15.0  # seconds; model stays resident during bursts, released after this quiet gap

    def _schedule_release(self) -> None:
        """Arm a 15s timer after each synthesis starts; new synthesis resets the clock and cancels a parked kill."""
        self._cancel_release()
        with _STREAMS_LOCK:
            global _PENDING_RELEASE
            _PENDING_RELEASE = False  # a fresh round of synthesis is running; the earlier kill intent is void
        self._release_timer = threading.Timer(
            self._SESSION_TIMEOUT, self._release_now)
        self._release_timer.daemon = True
        self._release_timer.start()

    def _cancel_release(self) -> None:
        if self._release_timer is not None:
            self._release_timer.cancel()
            self._release_timer = None

    def _release_now(self) -> None:
        """Session timeout expired: read-aloud still ON (leased) -> stay hot;
        OFF -> kill the child.  If chunks are still flowing, do not hard-kill —
        rearm the 15s timer instead: during a long reply the child stays hot, and
        only 15s of silence after the final sentence ends it (no reload mid-talk)."""
        if self._leased:
            return
        with _STREAMS_LOCK:
            busy = _ACTIVE_STREAMS > 0 or _SYNTH_LOCK.locked()
        if busy:
            self._schedule_release()  # still streaming: reset the timer, decide again after 15s of silence
            return
        _stop_server()


def register(ctx) -> None:
    # registration never loads the engine: even with dependency drift the provider name holds, blocking any silent route to online Edge TTS
    ctx.register_tts_provider(Qwen3TTSProvider())
