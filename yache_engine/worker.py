#!/usr/bin/env python3
"""Qwen3-TTS 1.7B engine worker for the Yache Voice plugin.

Runs under the *engine* venv interpreter as a standalone child process
spawned by ``tts.py`` (main path: ``python worker.py --server``), but can
also be imported / invoked directly for debugging:

1. Server mode (main path — streaming PCM over stdio, zero disk):
   python worker.py --server
   stdin : {"text": "...", "id": N}
   stdout: {"type": "pcm", "id": N, "sample_rate": 24000, "data": "<b64 int16>"}
           {"type": "done", "id": N}
   Model is loaded lazily and RELEASED after every request (VRAM goes back
   to the main LLM); the process self-exits after an idle timeout.

2. Preview mode (dashboard draft test):
   python worker.py --preview   with env VA_VOICE_JSON = full draft spec JSON
   stdin : {"text": "<preview sentence>"}
   stdout: READY, then one result JSON line
           {"ok": true, "wav_b64": "...", "sample_rate": 24000, "seconds": N}
           {"ok": false, "error": "<the engine's original error>"}
   Nothing is ever written to disk; the wav is assembled in memory.

3. Inline mode (debug): ``synthesize("text")`` plays via sounddevice.

The voice is a *spec*: variant (base/customvoice/voicedesign) + quantization
file + hardware backend + kind payload (reference audio / built-in speaker
name / design description). Spec sources, highest first: env VA_VOICE_JSON
(the panel's draft), then QWEN3_TTS_VOICE resolved through
voices/registry.json. One resolution source (yache_voice scanner, GGUF
metadata verified) so panel and service can never disagree about which
model file actually loads. Device "cuda" load/generate failures retry once
on the CPU backend and *stick* for this process (no flapping); if CPU fails
too, the original engine error surfaces (official degrade).

All filesystem paths resolve through ``paths.py`` (env -> data file ->
platform default); never a personal absolute path.
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
import re
import subprocess
import sys
import time
import json
import base64
import threading
import queue as _q
import gc

# Path resolution must work in all three load modes: run as a script under
# the engine venv (sys.path[0]=this dir), imported as engine.worker by the
# agent half, or loaded as a bare module by the web half (dashboard reuses
# cuda_free_mib — pure stdlib, no torch pulled in).
try:
    import paths as _p                              # direct-script mode
except ModuleNotFoundError:
    try:
        from . import paths as _p                   # package mode
    except ImportError:
        import importlib.util as _ilu
        import pathlib as _pl
        _spec = _ilu.spec_from_file_location(
            "yache_voice_paths", str(_pl.Path(__file__).with_name("paths.py")))
        _p = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_p)                # external bare-module mode

# Same three-mode dance for the scanner/registry helper (stdlib only).
try:
    import yache_voice as _vz                       # direct-script mode
except ModuleNotFoundError:
    try:
        from . import yache_voice as _vz            # package mode
    except ImportError:
        import importlib.util as _ilu3
        import pathlib as _pl3
        _sp3 = _ilu3.spec_from_file_location(
            "yache_voice", str(_pl3.Path(__file__).with_name("yache_voice.py")))
        _vz = _ilu3.module_from_spec(_sp3)
        _sp3.loader.exec_module(_vz)

# Bilingual user-facing strings: same three-mode load dance as above; a
# standalone debug run outside the plugin tree keeps the Chinese original.
try:
    import i18n as _i18n                              # direct-script mode
except ModuleNotFoundError:
    try:
        from . import i18n as _i18n                   # package mode
    except ImportError:
        try:
            import importlib.util as _ilu5
            import pathlib as _pl5
            _sp5 = _ilu5.spec_from_file_location(
                "yache_voice_i18n", str(_pl5.Path(__file__).with_name("i18n.py")))
            _i18n = _ilu5.module_from_spec(_sp5)
            _sp5.loader.exec_module(_i18n)            # external bare-module mode
        except Exception:
            class _i18n:                              # standalone fallback: zh only
                @staticmethod
                def ui(zh, en):
                    return zh
_ui = _i18n.ui


BASE_DIR = str(_p.tts_base_dir())
TALKER, TOKENIZER = (str(x) for x in _p.model_paths())
TEMPERATURE = float(os.environ.get("QWEN3_TTS_TEMPERATURE", "0.3"))

# ---- Voice registry ----
HERMES_VOICES_DIR = str(_p.voices_dir())


def _resolve_voice_path(p):
    if p.startswith("~/.hermes/"):
        return str(_p.hermes_home() / p[len("~/.hermes/"):])
    return os.path.expanduser(p)

# ---- Text normalization ----
_CN_DIGITS = "零一二三四五六七八九"


def _int_to_cn(n):
    if n == 0: return "零"
    if n < 10: return _CN_DIGITS[n]
    if n < 20: return "十" + (_CN_DIGITS[n % 10] if n % 10 else "")
    if n < 100: return _CN_DIGITS[n // 10] + "十" + (_CN_DIGITS[n % 10] if n % 10 else "")
    if n < 1000:
        s = _CN_DIGITS[n // 100] + "百"
        rem = n % 100
        if rem == 0: return s
        if rem < 10: return s + "零" + _CN_DIGITS[rem]
        return s + _int_to_cn(rem)
    if n < 10000: return _CN_DIGITS[n // 1000] + "千" + (_int_to_cn(n % 1000) if n % 1000 else "")
    if n < 100000000:
        wan = n // 10000; rem = n % 10000
        return _int_to_cn(wan) + "万" + (_int_to_cn(rem) if rem else "")
    return _int_to_cn(n // 100000000) + "亿" + _int_to_cn(n % 100000000)


def _per_digit(s):
    return "".join(_CN_DIGITS[int(c)] for c in s)

_NUM_RE = re.compile(r"\d+(?:\.\d+)?")

# Non-Chinese context gate (2026-11-20, owner: the plugin ships worldwide and
# foreign users read aloud in English). normalize_times/units/numbers INJECT
# hanzi ("12" -> "十二"); inside a latin sentence that breaks the reading — the
# engine pronounces injected hanzi with Chinese phonemes, while Qwen3-TTS's own
# text frontend already reads latin-context digits correctly ("5ms" -> "five
# milliseconds", "2026" -> "twenty twenty-six"). So every numeric rewrite is
# gated on CJK adjacency: only digits whose nearest non-space neighbour on
# either side is a CJK ideograph get spoken in Chinese. Per-digit gating (not
# whole-text language sniffing) is deliberate: mixed prose like "GDP 增长 5.2%"
# is the everyday reality. normalize_symbols is NOT gated — markdown/quotation/
# lone-surrogate scrubbing is script-neutral and protects English too.
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


def _cjk_adjacent(text, start, end):
    left = text[:start].rstrip()
    right = text[end:].lstrip()
    return bool((left and _CJK_RE.match(left[-1]))
                or (right and _CJK_RE.match(right[0])))


def normalize_numbers(text):
    def repl(m):
        if not _cjk_adjacent(text, m.start(), m.end()):
            return m.group(0)  # latin neighbourhood: the model reads it itself
        tok = m.group(0)
        if "." in tok:
            int_part, dec_part = tok.split(".", 1)
            return _int_to_cn(int(int_part)) + "点" + _per_digit(dec_part)
        n = int(tok)
        digits = tok.lstrip("0") or "0"
        if 1900 <= n <= 2099 and len(digits) == 4: return _per_digit(digits)
        if re.fullmatch(r"[1-9]0{4,}", digits): return _int_to_cn(n)
        if len(digits) >= 5: return _per_digit(digits)
        return _int_to_cn(n)
    return _NUM_RE.sub(repl, text)

_UNIT_MAP = {"ms": "毫秒", "min": "分钟", "s": "秒", "h": "小时"}
_UNIT_RE = re.compile(r"(\d+)(ms|min|s|h)(?![a-zA-Z])")


def normalize_units(text):
    def repl(m):
        if not _cjk_adjacent(text, m.start(), m.end()):
            return m.group(0)  # "took 5ms" stays latin for the English frontend
        return m.group(1) + _UNIT_MAP[m.group(2)]
    return _UNIT_RE.sub(repl, text)

_TIME_RE = re.compile(r"\b(\d{1,2}):(\d{2})\b(?::(\d{2})\b)?")


def _hour_cn(h):
    if h < 10: return _CN_DIGITS[h]
    if h < 20: return "十" + (_CN_DIGITS[h % 10] if h % 10 else "")
    return _CN_DIGITS[h // 10] + "十" + (_CN_DIGITS[h % 10] if h % 10 else "")


# Latin-context time spelling (2026-11-20 international fix): the engine has
# no English text normalization of its own, and whisper-verified A/B against
# R2T2 (Chinese-lean ASR) proved the digit form "4:45" gets read with Chinese
# syllables even under an explicit English language tag ("It is 四四級五
# a.m."), while plain latin digits ("13 test cases", "a.m.") read as English
# fine. So times must be spelled into English WORDS before synthesis.
_EN_ONES = ("zero one two three four five six seven eight nine ten eleven "
            "twelve thirteen fourteen fifteen sixteen seventeen eighteen "
            "nineteen").split()
_EN_TENS = {2: "twenty", 3: "thirty", 4: "forty", 5: "fifty"}


def _en_num59(n):
    if n < 20:
        return _EN_ONES[n]
    t, r = divmod(n, 10)
    return _EN_TENS[t] + ("-" + _EN_ONES[r] if r else "")


def _time_en(h, mm, ss):
    if mm == 0 and ss in (None, 0):
        return _en_num59(h) + " o'clock"
    # 11-20 主人亲耳终审: "oh five" 念成汉字(穿帮)、裸 "o five" 念对英文——白名单只留 o。
    s = _en_num59(h) + (" o " if 0 < mm < 10 else " ") + _en_num59(mm)
    if ss:
        s += " " + _en_num59(ss)
    return s


def normalize_times(text):
    def repl(m):
        h, mm = int(m.group(1)), int(m.group(2))
        ss = int(m.group(3)) if m.group(3) else None
        if not (0 <= h <= 23 and 0 <= mm <= 59) or (ss is not None and not 0 <= ss <= 59):
            return m.group(0)
        if not _cjk_adjacent(text, m.start(), m.end()):
            return _time_en(h, mm, ss)  # "at 12:30 pm" -> "twelve thirty pm"
        s = _hour_cn(h) + "点"
        if mm == 0:
            s += "零分" if ss else "整"
        else:
            s += _int_to_cn(mm) + "分"
        if ss is not None and ss != 0: s += _int_to_cn(ss) + "秒"
        return s
    return _TIME_RE.sub(repl, text)

# A lone UTF-16 surrogate cannot be encoded to UTF-8 on its own and would
# crash synthesis at the encode step (surrogates not allowed); properly
# paired astral chars (emoji etc.) fall outside the \ud800-\udfff single-codepoint range and are never stripped by mistake.
_LONE_SURROGATE_RE = re.compile('[\ud800-\udfff]')


def normalize_symbols(text):
    text = _LONE_SURROGATE_RE.sub("", text)
    text = re.sub(r"—{2,}", "，", text)
    text = re.sub(r"–{2,}", "，", text)
    text = re.sub(r"[~～]+", "", text)
    text = re.sub(r"[/|]+", "，", text)
    text = re.sub(r"[（(【\[]", "，", text)
    text = re.sub(r"[）)】\]]", "，", text)
    text = re.sub(r"[*_`#+>]+", "", text)
    text = re.sub(r'["""\u2018\u2019\u300c\u300d\u300e\u300f]', "", text)
    text = re.sub(r"，{2,}", "，", text)
    text = re.sub(r"^[，\s]+", "", text)
    return text.strip()

# ---- Voice spec resolution (single source; shared with the dashboard) ----
_scan_cache = {}
_scan_lock = threading.Lock()


def _scan_cached(mdir):
    with _scan_lock:
        r = _scan_cache.get(mdir)
    if r is None:
        r = _vz.scan_models_dir(mdir)
        with _scan_lock:
            _scan_cache[mdir] = r
    return r


def _load_spec():
    """Build the voice spec for this process. VA_VOICE_JSON (panel draft)
    wins; otherwise QWEN3_TTS_VOICE -> voices/registry.json entry, else the
    registry default. Clone registry entries (mode=clone + ref_audio +
    ref_text_file) resolve through normalize_entry."""
    raw = os.environ.get("VA_VOICE_JSON", "").strip()
    if raw:
        try:
            spec = json.loads(raw)
        except ValueError as exc:
            raise ValueError(_ui("VA_VOICE_JSON 不是合法 JSON: %s", "VA_VOICE_JSON is not valid JSON: %s") % exc)
        if not isinstance(spec, dict):
            raise ValueError(_ui("VA_VOICE_JSON 必须是 JSON 对象", "VA_VOICE_JSON must be a JSON object"))
    else:
        # Panel-saved selection (dashboard /voice/save writes "voice" into
        # the plugin data file) sits between the env draft and the
        # registry path, so tts.py needs zero changes to follow the panel.
        try:
            saved = _p.load_data().get("voice")
        except Exception:
            saved = None
        if isinstance(saved, dict) and saved.get("kind"):
            spec = dict(saved)
        else:
            voice_id = os.environ.get("QWEN3_TTS_VOICE", "").strip()
            reg_path = str(_p.voices_registry_path())
            if not os.path.exists(reg_path):
                raise RuntimeError(
                    f"voice registry not found at {reg_path} — enroll a voice first")
            with open(reg_path, encoding="utf-8") as rf:
                reg = json.load(rf)
            voice_id = voice_id or reg.get("default", "")
            entry = (reg.get("voices") or {}).get(voice_id)
            if entry is None:
                available = ", ".join(sorted(reg.get("voices") or {})) or _ui("（空）", "(empty)")
                raise RuntimeError(_ui(
                    f"no usable voice: '{voice_id or '(unset)'}' 不在音色库里 — 可用: {available}（{reg_path}）",
                    f"no usable voice: '{voice_id or '(unset)'}' is not in the voice registry — available: {available} ({reg_path})"))
            spec = dict(_vz.normalize_entry(voice_id, entry))
            spec["name"] = voice_id
    kind = str(spec.get("kind") or spec.get("mode") or "clone").lower()
    spec["kind"] = kind if kind in ("clone", "speaker", "design") else "clone"
    spec.setdefault("variant", "base")
    # model size (0.6b / 1.7b) is the third axis of a build: variant alone
    # no longer identifies a file. "" = unpinned -> _resolve_files keeps the
    # historical default (the biggest quant file on disk wins).
    spec.setdefault("model_size", "")
    spec.setdefault("models_dir", None)
    spec.setdefault("quant", None)
    spec.setdefault("device", None)
    if spec["kind"] == "clone" and not str(spec.get("ref_audio") or "").strip():
        # A bare clone name (what the panel saves) resolves against the
        # voice registry HERE — the single choke point every mode shares,
        # so the server path and the dashboard preview cannot drift apart.
        try:
            reg = _vz.load_registry(str(_p.voices_registry_path())) or {}
        except Exception:
            reg = {}
        voices = reg.get("voices") if isinstance(reg.get("voices"), dict) else {}
        vid = str(spec.get("voice") or "").strip() or str(reg.get("default") or "")
        meta = voices.get(vid)
        norm = _vz.normalize_entry(vid, meta) if isinstance(meta, dict) else None
        if norm is not None:
            spec["voice"] = vid
            spec["name"] = vid
            spec["ref_audio"] = str(norm.get("ref_audio") or "")
            spec["ref_text"] = str(norm.get("ref_text") or "")
    if spec["kind"] == "clone" and spec.get("ref_audio"):
        spec["ref_audio"] = _resolve_voice_path(spec["ref_audio"])
    # Panel synthesis config (dashboard /voice/tts writes quant+device into
    # the plugin data file) fills a spec's tier fields when the spec itself
    # does not pin them. "auto" means "no pin" and falls through to the
    # per-request VRAM gate in _resolve_files/_get_model.
    if spec.get("quant") in (None, "") or spec.get("device") in (None, ""):
        try:
            cfg = _p.load_data().get("tts") or {}
        except Exception:
            cfg = {}
        if spec.get("quant") in (None, ""):
            q = str(cfg.get("quant") or "").strip().lower()
            if q and q != "auto":
                spec["quant"] = q.upper()
        if spec.get("device") in (None, ""):
            d = str(cfg.get("device") or "").strip().lower()
            if d and d != "auto":
                spec["device"] = d
    return spec


def _resolve_files(spec):
    """(talker_path, tokenizer_path, quant) from spec — the scanner walks
    GGUF metadata (never filename guesses) to confirm variant identity.
    The quant menu picks (tier, model_size); an unpinned size keeps the
    pre-size default by taking the scanner's first match, which sorts the
    1.7b build before the 0.6b one. Raises ValueError with human-readable
    text on anything missing."""
    mdir = str(spec.get("models_dir") or BASE_DIR)
    variant = str(spec.get("variant") or "base").lower()
    size = str(spec.get("model_size") or "").strip().lower() or None
    try:
        scan = _scan_cached(mdir)
    except _vz.ScanError as exc:
        raise ValueError(str(exc))
    var = next((v for v in scan["variants"] if v["variant"] == variant), None)
    if var is None:
        have = ", ".join(v["variant"] for v in scan["variants"])
        raise ValueError(_ui("该模型目录里没有 %s 变体（目录里实际有: %s）",
                             "this model directory has no %s variant (it actually has: %s)")
                         % (variant, have))
    quant = str(spec.get("quant") or "").strip().upper() or None
    if quant is None:
        rec = _vz.recommend(var["quants"], cuda_free_mib(), _has_nvidia())
        quant = rec["quant"]
        size = size or (str(rec.get("model_size") or "").lower() or None)
        print(f"[yache-voice] auto-quant: {quant}"
              f"{(' · ' + size) if size else ''} ({rec['reason']})",
              file=sys.stderr, flush=True)
    qf = next((q for q in var["quants"]
               if q["quant"] == quant
               and (size is None
                    or (q.get("model_size") or "1.7b") == size)), None)
    if qf is None:
        # name the sizes in the hint: with two builds sharing a tier label,
        # "it has: Q4_K_M" alone would still be a riddle
        have = ", ".join(sorted({
            q["quant"] + ((" · " + q["model_size"].upper())
                          if q.get("model_size") else "")
            for q in var["quants"]}))
        raise ValueError(_ui("该变体磁盘上没有 %s%s 量化文件（实际有: %s）",
                             "this variant has no %s%s quant file on disk (it has: %s)")
                         % (quant, (" · " + size.upper()) if size else "", have))
    device = spec.get("device")
    if device not in (None, "", "cpu", "cuda"):
        raise ValueError(_ui("device 只认 cpu / cuda / 自动，收到: %r", "device accepts only cpu / cuda / auto, got: %r") % device)
    base = str(scan["dir"])
    return (os.path.join(base, qf["file"]),
            os.path.join(base, scan["tokenizer"]["preferred"]), quant)


def _has_nvidia():
    try:
        import shutil
        return shutil.which("nvidia-smi") is not None
    except Exception:
        return False

# ---- Model cache (load per spec, release after each request) ----
_model = None
_model_key = None
_LAST_BACKEND = None      # backend actually loaded last ("cpu"/"cuda"/"auto")
_CPU_STICKY = False       # after a GPU failure stay on CPU for this process
_model_lock = threading.Lock()


def _get_model(spec=None):
    """Load (or swap for a different spec) the engine. Returns (model, quant)."""
    global _model, _model_key, _LAST_BACKEND
    if spec is None:
        spec = _load_spec()
    talker, tokenizer, quant = _resolve_files(spec)
    device = spec.get("device") or None
    if _CPU_STICKY or device == "cpu":
        backend = "cpu"
    elif device == "cuda":
        backend = "cuda"
    else:
        backend = resolve_backend()
    key = (talker, tokenizer, backend)
    with _model_lock:
        if _model is not None and _model_key != key:
            _release_model_locked()
        if _model is None:
            t = time.time()
            # Add engine venv site-packages to sys.path for faster_qwen3_tts
            _venv_sp = str(_p.engine_site_packages())
            if _venv_sp not in sys.path:
                sys.path.insert(0, _venv_sp)
            _dll_dir = str(_p.dll_search_dir())
            if os.path.isdir(_dll_dir):
                os.add_dll_directory(_dll_dir)
            from faster_qwen3_tts import FasterQwen3TTS
            if backend == "cpu":
                os.environ["GGML_BACKEND"] = "CPU"
            elif backend == "cuda":
                os.environ["GGML_BACKEND"] = "CUDA0"
            else:
                os.environ.pop("GGML_BACKEND", None)
            _free = cuda_free_mib()
            print(f"[yache-voice] backend gate: {backend}"
                  + (f" (free VRAM {_free} MiB)" if _free is not None
                     else " (no nvidia-smi info)")
                  + f" quant={quant}", file=sys.stderr, flush=True)
            _model = FasterQwen3TTS.from_pretrained(
                os.path.dirname(talker) or ".",
                backend="ggml",
                gguf_talker_path=talker,
                gguf_codec_path=tokenizer,
                local_files_only=True,
            )
            _model_key = key
            _LAST_BACKEND = backend
            print(f"[yache-voice] model loaded in {time.time()-t:.1f}s",
                  file=sys.stderr, flush=True)
    return _model, quant


def _release_model():
    """Release TTS model + GPU memory back to the main LLM after playback.

    Closes the C-level runtime (qt_free) and drops our reference so gc can
    reclaim the Python objects. Next synthesize() re-loads (few seconds).
    """
    with _model_lock:
        _release_model_locked()


def _release_model_locked():
    global _model, _model_key
    if _model is None:
        return
    t = time.time()
    try:
        # GGMLQwen3TTS holds self.runtime (a QwenTTS). Close it -> qt_free.
        rt = getattr(_model, "runtime", None)
        if rt is not None and hasattr(rt, "close"):
            rt.close()
    except Exception as e:
        print(f"[yache-voice] close warn: {e}", file=sys.stderr)
    _model = None
    _model_key = None
    gc.collect()
    print(f"[yache-voice] model released in {time.time()-t:.1f}s", file=sys.stderr)


def _quality_ok(audio, sr):
    """Synthesis quality gate (audio is normalized float32 in -1.0..1.0).
    Returns False on a bad take -> the caller retries one more time."""
    import numpy as np
    if len(audio) == 0:
        return False
    if len(audio) < sr * 0.05:  # <50ms -> nothing was synthesized
        return False
    # over 85% of samples with abs value < 0.005 -> essentially all silence
    if float(np.mean(np.abs(audio) < 0.005)) > 0.85:
        return False
    # RMS sanity range
    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms < 0.003:   # near-silent
        return False
    if rms > 0.8:     # near clipping
        return False
    # clipping check: >30% of samples pinned to the ceiling
    if float(np.mean(np.abs(audio) > 0.95)) > 0.3:
        return False
    return True

# ---- Sentence splitting & silence trimming ----
# Sentence ends: CJK punctuation splits anywhere; Latin .!?; only splits when
# followed by whitespace/EOL (so the dots inside "a.m." / "e.g." are not cut
# mid-word — the old regex split "4:45 a.m." into "4:45 a" + "m", which made
# the model read the orphaned time fragment with Chinese habits), and even
# then a ".x." single-letter abbreviation tail (.m. of a.m.) suppresses it.
_SENT_RE = re.compile(r'(?<=[。！？；…])\s*|(?<=[.!?\;])(?<!\.[a-z]\.)(?=\s|$)')


_TIMEY_RE = re.compile(
    r"\d{1,2}[:：]\d{2}"                                  # 数字钟点 4:45
    r"|[〇零一二两三四五六七八九十百]{1,4}点"                # 汉字钟点 四点/十二点
    r"|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
    r"thirty|forty|fifty|o'clock|o)\b", re.I)             # 英文时间词 nine/o five(o=11-20 白名单写法)

def _split_sentences(text):
    """Sentence-level split; anything past MAX_LEN chars gets a second comma/pause split.
    Keeps one giant blob from blowing past max_new_tokens and being truncated mid-read.
    时间密集门(11-20 主人亲耳实锤)：1.7B talker 在一句里遇到两个及以上钟点会漂——
    英文时间词念成汉字("nine oh five"→九点零五)、汉字时间吞念("四点零分三十秒"→四点点半)，
    单独喂短句实测全对。故两个时间块绝不许合并进同一次生成，就地切开各自合成。"""
    MAX_LEN = 60
    parts = _SENT_RE.split(text)
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if len(p) <= MAX_LEN and len(_TIMEY_RE.findall(p)) < 2:
            out.append(p)
            continue
        # oversized / time-dense: second-pass split on commas/pauses/colons/newlines
        sub = re.split(r"(?<=[，,、：:;\n])", p)
        buf = ""
        for s in sub:
            # flush before merge when: 会超长, 或 buf 与 s 各含时间块（两个钟点不进同一句）
            if buf and (len(buf) + len(s) > MAX_LEN or
                        (_TIMEY_RE.search(buf) and _TIMEY_RE.search(s))):
                out.append(buf)
                buf = ""
            buf += s
        if buf.strip():
            out.append(buf.strip())
    return [s for s in out if s.strip()]


def _trim_leading_silence(audio, sr):
    import numpy as np
    if len(audio) > sr * 0.5:
        rms_win = int(sr * 0.05)
        rms = np.array([np.abs(audio[i:i+rms_win]).mean() for i in range(0, len(audio)-rms_win, rms_win)])
        voiced = np.where(rms > 0.005)[0]
        if len(voiced) > 0:
            start = int(voiced[0] * rms_win)
            if start > 0:
                return audio[start:]
    return audio

# ---- VRAM pre-flight gate (health check before making any sound) ----


def cuda_free_mib():
    """Free VRAM (MiB) of the first NVIDIA GPU.

    Returns None when there is no NVIDIA GPU here, nvidia-smi is missing,
    or the driver is too busy to answer — fail-open: None means "cannot
    tell" and the engine's own best-device probe still runs. Never blocks
    the box: two queries max, each capped at 5 s, no polling loop (a
    VRAM-pressured driver blocks nvidia-smi, and hammering it is how
    probes strangle the whole machine).

    Windows landmine (measured 2026-11-11): this worker runs as a
    subprocess of a GUI process, so nvidia-smi.exe allocates its own
    console and its stdout vanishes into a hidden window — an empty
    capture, not an error. CREATE_NO_WINDOW keeps the pipe attached.
    """
    cmd, err = ["nvidia-smi", "--query-gpu=memory.free",
                "--format=csv,noheader,nounits"], None
    for _ in range(2):
        try:
            flags = 0x08000000 if os.name == "nt" else 0
            r = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=5, creationflags=flags)
            if r.returncode == 0:
                line = r.stdout.strip().splitlines()[0] if r.stdout.strip() else ""
                if line:
                    return int(line)
            else:
                err = (r.stderr or "").strip()[:120] or f"exit {r.returncode}"
        except FileNotFoundError:
            return None            # no NVIDIA toolchain at all
        except Exception as exc:   # timeout etc: driver busy, don't insist
            err = str(exc)[:120]
    if err:
        print(f"[yache-voice] nvidia-smi unavailable: {err}", file=sys.stderr)
    return None


def resolve_backend():
    """Decide which ggml backend an auto-device load should use.

    Measured on a 24 GB box with the chat engine resident (free ≈ 1.9 G):
    the 1.7 B Q8 model still *loads* onto the GPU (WDDM page-faults it in,
    load 3.8 s -> 10.0 s, GPU 98 %) and synthesis then crawls so hard that
    one short sentence never finished in 150 s. A silent crawl is worse
    than an honest CPU run, so below the gate we pin the engine to its CPU
    backend through GGML_BACKEND — the native force-device switch in
    qwentts.cpp backend.h (works on the stock DLL, no code change).

    Called from _get_model(), i.e. once per load: the server releases the
    model after every request, so each reload re-checks current free VRAM
    and follows the chat engine coming and going.

    VA_TTS_BACKEND=cuda|cpu overrides the probe outright; the 3800 MiB
    gate (measured 1528 MiB model footprint + activations + headroom) is
    tunable via VA_TTS_MIN_FREE_MIB.
    """
    # Hard guard first: without the CUDA DLLs (GPU accelerator pack not
    # installed) every path, even an explicit VA_TTS_BACKEND=cuda, must pin
    # to the CPU backend — one safe path beats a crash or a silent crawl.
    try:
        dll_dir = _p.dll_search_dir()
        has_cuda = ((dll_dir / "ggml-cuda.dll").is_file()
                    and (dll_dir / "cublas64_12.dll").is_file())
    except OSError:
        has_cuda = False
    if not has_cuda:
        return "cpu"
    forced = os.environ.get("VA_TTS_BACKEND", "").strip().lower()
    if forced == "cpu":
        return "cpu"
    if forced == "cuda":
        return "cuda"
    free = cuda_free_mib()
    if free is None:
        return "auto"  # no NVIDIA / no nvidia-smi: let ggml pick its best
    gate = int(os.environ.get("VA_TTS_MIN_FREE_MIB", "3800"))
    return "cuda" if free >= gate else "cpu"

# ---- Voice resolution (ref triple for the tts.py inline path) ----


def _resolve_voice(spec=None):
    """Back-compat accessor used by tts.py inline mode: returns
    (ref_audio_or_speaker_or_instruct, ref_text, temperature). The hard gate
    stays: a clone voice needs an existing ref_audio, else fail fast."""
    if spec is None:
        spec = _load_spec()
    temp = float(spec.get("temperature") or TEMPERATURE)
    kind = spec.get("kind")
    if kind == "speaker":
        speaker = (spec.get("speaker") or "").strip()
        if not speaker:
            raise RuntimeError(_ui("speaker 音色缺少 speaker 名字", "speaker-type voice is missing its speaker name"))
        return speaker, "", temp
    if kind == "design":
        instruct = (spec.get("instruct") or "").strip()
        if not instruct:
            raise RuntimeError(_ui("design 音色缺少声音描述 instruct", "design-type voice is missing its instruct description"))
        return instruct, "", temp
    ref_audio = spec.get("ref_audio")
    if not ref_audio or not os.path.isfile(ref_audio):
        raise RuntimeError(_ui(
            f"no usable clone voice: '{spec.get('name', '')}' 缺参考音频 （{ref_audio or '(未设置)'}）— 请先在音色克隆面板保存一个克隆音色",
            f"no usable clone voice: '{spec.get('name', '')}' has no reference audio ({ref_audio or '(unset)'}) — save a cloned voice in the voice-cloning panel first"))
    return ref_audio, (spec.get("ref_text") or "").strip() or None, temp

# ---- Generation dispatch (three kinds, shared sampling knobs) ----

# (11-20 摘除 _ref_spk_emb_cache：缓存路从未生效+已被耳朵判死，见 _gen_sentence 注释)


def _gen_sentence(M, spec, sent, quant=None):
    """Generate one sentence per spec kind; returns (audio_float32, sr)."""
    temp = float(spec.get("temperature") or TEMPERATURE)
    # Language tag follows the sentence's script (2026-11-20 international
    # fix, owner: foreign users read aloud English). The old hardcoded
    # language="Chinese" pinned EVERY sentence — even pure-English ones —
    # to the Chinese language id, so the model read "4:45 a.m." as
    # 四点四十五. Probe on this machine (whisper-transcribed A/B/C, see
    # scratch/tts_probe_asr.py output): explicit English tag -> clean English
    # reading; "auto" -> first half English, second half degenerates into
    # gibberish on long sentences; Chinese tag -> full Chinese-isms. So:
    # CJK-bearing sentences keep the explicit Chinese tag (mixed prose reads
    # best in Chinese mode, owner-validated daily path); latin-only sentences
    # get the explicit English tag.
    lang = "Chinese" if _CJK_RE.search(sent) else "English"
    common = dict(language=lang, temperature=temp, do_sample=True,
                  top_p=0.85, max_new_tokens=480, repetition_penalty=1.05)
    kind = spec.get("kind")
    if kind == "speaker":
        result = M.generate_custom_voice(
            text=sent, speaker=(spec.get("speaker") or "").strip(), **common)
        return result[0][0], result[1]
    if kind == "design":
        result = M.generate_voice_design(
            text=sent, instruct=(spec.get("instruct") or "").strip(), **common)
        return result[0][0], result[1]
    # clone: 每句 raw ICL = 11-20 主人亲耳全对定格形态(ab-icl-* 组)。原 spk-embedding
    # 缓存分支传的是引擎签名里不存在的 spk_emb/spk/spk_id(真名 ref_spk_emb 系)，缓存
    # 命中必 TypeError 被阶梯静默吞掉、从未真正生效；ref_spk_emb+xvec_only 官方缓存路
    # 同晚被主人耳朵判死(英文后半卡壳)。一并摘除，单路到底。
    ref_audio = spec.get("ref_audio")
    if not ref_audio or not os.path.isfile(ref_audio):
        raise ValueError(_ui("克隆音色需要参考音频文件，当前: %r（文件不存在？）",
                             "a clone voice needs a reference audio file, got: %r (file missing?)")
                         % (ref_audio or ""))
    ref_text = (spec.get("ref_text") or "").strip() or None
    result = M.generate_voice_clone(
        text=sent, ref_audio=ref_audio, ref_text=ref_text, **common)
    return result[0][0], result[1]


def _gen_with_fallback(spec, sent):
    """One sentence with the design-doc §4 ladder: any engine-side failure
    retries once on the CPU backend and sticks there for this process (no
    flapping); CPU failure or data errors re-raise the original."""
    global _CPU_STICKY
    for attempt in (1, 2):
        M, quant = _get_model(spec)
        try:
            return _gen_sentence(M, spec, sent, quant)
        except ValueError:
            raise                    # spec/data problems: retry changes nothing
        except Exception as exc:
            if attempt == 1 and _LAST_BACKEND and _LAST_BACKEND != "cpu":
                _CPU_STICKY = True
                with _model_lock:
                    _release_model_locked()
                print(f"[yache-voice] GPU generation failed ({exc}); this process fell back"
                      f"to the CPU backend and will retry", file=sys.stderr, flush=True)
                continue
            raise
    raise RuntimeError("unreachable")


def _est_speech_seconds(sent):
    """粗时长预算: 汉字 ~0.30s/字、拉丁 ~0.07s/字符，地板 1.5s。"""
    han = sum(1 for ch in sent if "\u4e00" <= ch <= "\u9fff")
    return max(1.5, han * 0.30 + (len(sent) - han) * 0.07)


def _gen_guarded(spec, sent):
    """时长守门(11-20 主人亲耳实锤 44s 复读案): penalty 归位 1.05 防崩后，时间句
    偶发复读循环把音频灌到 30-45s。超预算 2 倍+1.5s 判为循环，重抽一次取较短者——
    正常念读永远碰不到这条线，零误杀。"""
    audio, sr = _gen_with_fallback(spec, sent)
    import numpy as np
    dur = np.asarray(audio).size / max(sr, 1)
    cap = _est_speech_seconds(sent) * 2.0 + 1.5
    if dur > cap:
        print("[yache-voice] loop-guard: %.1fs 超预算 cap %.1fs，重抽一次: %r"
              % (dur, cap, sent[:50]), file=sys.stderr, flush=True)
        a2, sr2 = _gen_with_fallback(spec, sent)
        if np.asarray(a2).size / max(sr2, 1) < dur:
            audio, sr = a2, sr2
    return audio, sr

# ---- Main synthesize function (inline, zero disk I/O) ----


def synthesize(text: str) -> None:
    """Synthesize and play text. Zero disk, all in memory."""
    # Ensure engine venv deps (numpy/sounddevice/etc) are importable BEFORE
    # any of them — inline mode runs under the backend Python which lacks them.
    _venv_sp = str(_p.engine_site_packages())
    if _venv_sp not in sys.path:
        sys.path.insert(0, _venv_sp)
    import numpy as np
    import sounddevice as _sd

    t0 = time.time()

    text = normalize_symbols(text)
    text = normalize_times(text)
    text = normalize_units(text)
    text = normalize_numbers(text)

    if not text:
        print(f"[yache-voice] empty text, skip")
        return

    spec = _load_spec()

    sentences = _split_sentences(text)
    _is_pipeline = len(sentences) > 1
    t_gen = time.time()

    if _is_pipeline:
        print(f"[yache-voice] PIPELINE: {len(sentences)} sentences")
        play_queue = _q.Queue(maxsize=4)
        sentinel = object()
        gen_errors = 0

        def _play_worker():
            while True:
                item = play_queue.get()
                if item is sentinel:
                    break
                audio, sr, idx = item
                try:
                    _sd.play(audio, sr)
                    _sd.wait()
                except Exception as e:
                    print(f"[yache-voice] play err sent{idx}: {e}", file=sys.stderr)
                    try:
                        _sd.stop()
                    except Exception:
                        pass
                finally:
                    del audio

        player = threading.Thread(target=_play_worker, daemon=True)
        player.start()

        for i, sent in enumerate(sentences):
            try:
                audio, sr = _gen_guarded(spec, sent)
                audio = _trim_leading_silence(audio, sr)
                play_queue.put((audio, sr, i))
                dur = len(audio) / sr
                print(f"[yache-voice] sent{i+1}/{len(sentences)}: {dur:.1f}s")
            except Exception as e:
                gen_errors += 1
                print(f"[yache-voice] sent{i+1} gen err: {e}", file=sys.stderr)

        play_queue.put(sentinel)
        player.join(timeout=300)
        if player.is_alive():
            print(f"[yache-voice] WARNING: player timeout, forcing stop", file=sys.stderr)
            try:
                _sd.stop()
            except Exception:
                pass

        gen_time = time.time() - t_gen
        print(f"[yache-voice] pipeline done: {len(sentences)} sents, {gen_errors} errs, {time.time()-t0:.1f}s")

    else:
        try:
            audio, sr = _gen_with_fallback(
                spec, sentences[0] if sentences else text)
            audio = _trim_leading_silence(audio, sr)
            _sd.play(audio, sr)
            _sd.wait()
            dur = len(audio) / sr
            gen_time = time.time() - t_gen
            rtf = gen_time / dur if dur > 0 else 0
            print(f"[yache-voice] done: {dur:.1f}s, RTF={rtf:.2f}, total={time.time()-t0:.1f}s")
        except Exception as e:
            print(f"[yache-voice] ERROR: {e}", file=sys.stderr)

    # Release TTS model + GPU memory back after playback is done
    _release_model()

# ---- Server mode (persistent, streaming PCM via stdout) ----
_TTS_IDLE_TIMEOUT = float(os.environ.get("QWEN3_TTS_IDLE_TIMEOUT", "300"))  # seconds


def _server_loop():
    """Persistent TTS server: reads JSON from stdin, writes PCM JSON to stdout.

    Protocol (newline-delimited JSON):
      stdin:  {"text": "...", "id": N}
      stdout: {"type": "pcm", "id": N, "sample_rate": 24000, "data": "<base64 int16>"}
              {"type": "done", "id": N}

    Exits automatically after _TTS_IDLE_TIMEOUT seconds of inactivity.
    """
    import numpy as np
    try:
        import librosa
    except ModuleNotFoundError:
        # librosa (and its scipy/numba tail) only exist to stretch the
        # stream 0.95x — a timbre nicety, never a reason to kill the whole
        # streaming read-aloud chain on a lean install (2026-11-19 fresh-
        # machine audit).
        librosa = None

    # Model load is lazy now (happens on the first request): READY comes
    # fast even when the spec is broken, and a bad spec yields a clean
    # error chunk instead of killing the worker at startup.
    print("READY", flush=True)

    # Idle watchdog: exit if no request for _TTS_IDLE_TIMEOUT seconds
    _last_activity = [time.time()]

    def _idle_watchdog():
        while True:
            time.sleep(10)
            if time.time() - _last_activity[0] > _TTS_IDLE_TIMEOUT:
                print("[yache-voice] idle timeout, exiting", file=sys.stderr)
                os._exit(0)
    threading.Thread(target=_idle_watchdog, daemon=True).start()

    for raw_line in sys.stdin.buffer:
        _last_activity[0] = time.time()
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            print(json.dumps({"type": "error", "msg": "bad json"}), flush=True)
            continue

        text = req.get("text", "")
        req_id = req.get("id", 0)
        text = normalize_symbols(text)
        text = normalize_times(text)
        text = normalize_units(text)
        text = normalize_numbers(text)

        if not text:
            print(json.dumps({"type": "done", "id": req_id}), flush=True)
            continue

        # Re-read the spec every request: the model is released and
        # re-loaded per request anyway, so a voice or tier saved in the
        # panel takes effect on the very next reply — no restart, and no
        # stale-spec cache to explain away later.
        try:
            spec = _load_spec()
            print(f"[yache-voice] voice spec: kind={spec['kind']} "
                  f"variant={spec.get('variant')} name={spec.get('name', '')}",
                  file=sys.stderr, flush=True)
        except Exception as exc:
            print(json.dumps({"type": "error", "id": req_id,
                              "msg": str(exc)}), flush=True)
            continue

        req_spec = spec

        # ---- per-sentence streaming: print one PCM chunk (same req_id) the moment a sentence is ready ----
        # the renderer chains chunks by arrival order and the line reader happily takes many chunks per id,
        # so a long reply starts playing at its first sentence (the old build buffered all_audio to the end).
        sentences = _split_sentences(text) or [text]
        _gen_err = 0
        for _si, sent in enumerate(sentences):
            try:
                audio, sr = _gen_with_fallback(req_spec, sent)
                audio = _trim_leading_silence(audio, sr)
                # quality gate: retry once on a bad take (happens before print, so playback never doubles)
                if not _quality_ok(audio, sr):
                    print(f"[yache-voice] sent{_si+1} quality bad, retry", file=sys.stderr, flush=True)
                    audio, sr = _gen_with_fallback(req_spec, sent)
                    audio = _trim_leading_silence(audio, sr)
                # ease the pace ~5% (time-stretch, pitch preserved: keeps the soft feel without sounding hollow)
                if librosa is not None and len(audio) > sr * 0.1:  # >100ms only; and librosa is optional
                    audio = librosa.effects.time_stretch(audio, rate=0.95)
                pcm16 = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
                del audio
                _pcm_bytes = pcm16.tobytes()
                pcm_b64 = base64.b64encode(_pcm_bytes).decode("ascii")
                del pcm16, _pcm_bytes
                print(json.dumps({"type": "pcm", "id": req_id,
                                  "sample_rate": int(sr), "data": pcm_b64}), flush=True)
                del pcm_b64
                print(f"[yache-voice] A sent{_si+1}/{len(sentences)}", file=sys.stderr, flush=True)
            except Exception as e:
                _gen_err += 1
                print(json.dumps({"type": "error", "id": req_id, "msg": str(e)}), flush=True)

        print(json.dumps({"type": "done", "id": req_id}), flush=True)

        # Release TTS model + GPU memory back after each request. Prevents the
        # resident server from pinning VRAM while the user keeps chatting
        # (starving the main LLM). Next request re-loads via _get_model().
        _release_model()

# ---- Preview mode (dashboard draft: one text, wav in memory, nothing on disk) ----


def _preview_loop():
    """--preview with env VA_VOICE_JSON draft spec:
    stdin one {"text": ...} line -> stdout one result JSON line."""
    print("READY", flush=True)
    raw = sys.stdin.readline()
    try:
        text = (json.loads(raw or "{}").get("text") or "").strip()
        if not text:
            raise ValueError("empty text")
        spec = _load_spec()
        text = normalize_numbers(normalize_units(normalize_times(
            normalize_symbols(text))))
        t0 = time.time()
        audio, sr = _gen_guarded(spec, text)
        audio = _trim_leading_silence(audio, sr)
        import numpy as np
        a = np.asarray(audio, dtype=np.float32)
        peak = float(np.max(np.abs(a))) if a.size else 0.0
        if peak > 0:
            a = a * (0.95 / peak)
        pcm16 = np.clip(a * 32767.0, -32768, 32767).astype(np.int16)
        import io as _io
        import wave as _wave
        buf = _io.BytesIO()
        with _wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(int(sr))
            wf.writeframes(pcm16.tobytes())
        print(json.dumps({"ok": True,
                          "wav_b64": base64.b64encode(buf.getvalue()).decode("ascii"),
                          "sample_rate": int(sr),
                          "seconds": round(len(a) / float(sr), 2),
                          "elapsed": round(time.time() - t0, 1)},
                         ensure_ascii=False), flush=True)
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)},
                         ensure_ascii=False), flush=True)
    finally:
        _release_model()

# ---- Entry point ----
if __name__ == "__main__":
    if "--server" in sys.argv:
        _server_loop()
    elif "--preview" in sys.argv:
        _preview_loop()
    elif len(sys.argv) >= 2 and sys.argv[1] == "-":
        text = sys.stdin.read().strip()
        synthesize(text)
        print("[yache-voice] stdin mode done")
    else:
        print("Usage: worker.py [--server | --preview | - ]")
        sys.exit(1)

