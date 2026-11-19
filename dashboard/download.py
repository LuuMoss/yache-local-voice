r"""Model + GPU-accelerator download centre for the Yache local voice plugin.

Stdlib only, self-contained. One disk JSON state machine (atomic writes, the
frontend polls it), resumable Range downloads, per-session endpoint probing
(never cached, so VPN toggles are noticed), SHA-256 verification.

  * model : the 5 combos x 4 quants of Serveurperso/Qwen3-TTS-GGUF (talker +
            shared tokenizer, nothing else — the repo has no config sidecars).
            Files land in <chosen dir>\models\ (or the chosen dir itself when
            it IS named "models"); on completion the target is auto-loaded
            into the settings (tts_base_dir persisted + pair verified with
            the same GGUF metadata scanner the manual directory-load path
            uses). One step: download -> installed.
  * gpu   : the CUDA accelerator zip hosted on our GitHub release. Gated by
            the backend on "a model pair currently loads from tts_base_dir".
            Download -> SHA-256 -> auto-extract into <HERMES_HOME>\yache-cuda
            (upgrade-safe, shared by every model dir). The engine picks it up
            automatically through paths.dll_search_dir().
  * cancel: one marker file next to the state JSON; workers poll it.

Endpoints: model downloads race huggingface.co against hf-mirror.com (TLS +
first-128-KB throughput); the GPU pack prefers direct GitHub and falls back
to a public gh-proxy mirror.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import threading
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urljoin
from urllib.request import Request, urlopen

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from yache_engine import paths as _p  # noqa: E402
from yache_engine import yache_voice as _vy  # noqa: E402
from yache_engine import i18n as _i18n  # noqa: E402  (bilingual user-facing strings)

_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) YacheLocalVoice/1.0"
_CHUNK = 262144

HF_REPO = "Serveurperso/Qwen3-TTS-GGUF"
HF_ENDPOINTS = ["https://huggingface.co", "https://hf-mirror.com"]

# Our GitHub release hosting the CUDA accelerator zip. metadata.json lives in
# the same release and carries {asset, sha256, bytes}: the plugin never trusts
# an archive without the digest, and the byte count gives the progress bar an
# honest denominator before the first MB has moved.
GPU_REPO = "LuuMoss/yache-local-voice"
GPU_TAG = "cuda-pack-v1"
GPU_ASSET = "yache-cuda-windows-x64.zip"
GPU_META = "metadata.json"
GPU_ENDPOINTS = ["https://github.com", "https://gh-proxy.com"]

# ---- catalog: real values from the HF model card (0.6b has no voicedesign) ----
QUANTS = ["q4_k_m", "q8_0", "bf16", "f32"]
QUANT_LABELS = {"q4_k_m": "Q4_K_M", "q8_0": "Q8_0", "bf16": "BF16", "f32": "F32"}
TOK_MB = {"q4_k_m": 255, "q8_0": 291, "bf16": 359, "f32": 647}
TALKER_MB = {
    ("0.6b", "base"):        {"q4_k_m": 629, "q8_0": 993, "bf16": 1800, "f32": 3700},
    ("0.6b", "customvoice"): {"q4_k_m": 629, "q8_0": 993, "bf16": 1800, "f32": 3700},
    ("1.7b", "base"):        {"q4_k_m": 1200, "q8_0": 2100, "bf16": 3900, "f32": 7700},
    ("1.7b", "customvoice"): {"q4_k_m": 1200, "q8_0": 2100, "bf16": 3900, "f32": 7700},
    ("1.7b", "voicedesign"): {"q4_k_m": 1200, "q8_0": 2100, "bf16": 3900, "f32": 7700},
}
_L = _i18n.ui  # bilingual labels; the tables below store (zh, en) tuples and
# catalog() resolves them per request — an import-time _L(...) would freeze the
# language at gateway boot and ignore later display.language switches.
MODEL_LABELS = {
    ("0.6b", "base"):        ("0.6B · 音色克隆（参考音频）", "0.6B · voice cloning (reference audio)"),
    ("0.6b", "customvoice"): ("0.6B · 内置音色+方言", "0.6B · built-in preset voices + dialects"),
    ("1.7b", "base"):        ("1.7B · 音色克隆（参考音频）", "1.7B · voice cloning (reference audio)"),
    ("1.7b", "customvoice"): ("1.7B · 内置音色+方言", "1.7B · built-in preset voices + dialects"),
    ("1.7b", "voicedesign"): ("1.7B · 描述造声", "1.7B · design voices from text"),
}
MODEL_HINTS = {
    "0.6b": ("体积最小、最省显存的新手档", "smallest, lightest on VRAM, beginner tier"),
    "1.7b": ("推荐：本插件就是在这一档上开发和调校的", "recommended: the tier this plugin was developed and tuned on"),
}
VARIANT_HINTS = {
    "base": ("克隆底座：喂一段参考音频就复刻那个声音，本身不带内置音色", "the cloning base: give it a reference clip, it copies the voice; no built-in voices of its own"),
    "customvoice": ("官方内置音色合集：具名预置音色+中文方言，从列表里挑", "the official preset-voice lineup: named built-in voices + Chinese dialects, pick from the list"),
    "voicedesign": ("用一段文字描述凭空造一个声音（仅 1.7B）", "invent a voice from one text description (1.7B only)"),
}


def combo_files(combo: str, quant: str) -> Optional[Tuple[str, str]]:
    """(talker repo filename, tokenizer repo filename) for one selection."""
    m = re.fullmatch(r"(0\.6b|1\.7b)-(base|customvoice|voicedesign)", (combo or "").strip())
    if not m or quant not in QUANTS:
        return None
    size, variant = m.group(1), m.group(2)
    if size == "0.6b" and variant == "voicedesign":
        return None
    return (f"qwen-talker-{size}-{variant}-{quant.upper()}.gguf",
            f"qwen-tokenizer-12hz-{quant.upper()}.gguf")


def catalog() -> List[Dict[str, Any]]:
    """The two dropdowns flattened; the frontend filters by `size`."""
    out: List[Dict[str, Any]] = []
    for (size, variant), per in TALKER_MB.items():
        combo = f"{size}-{variant}"
        for q in QUANTS:
            total = per[q] + TOK_MB[q]
            out.append({
                "id": combo, "combo": combo, "size": size, "variant": variant,
                "quant": q,
                "model_label": _L(*MODEL_LABELS[(size, variant)]),
                "model_hint": f"{_L(*MODEL_HINTS[size])} | {_L(*VARIANT_HINTS[variant])}",
                "quant_label": QUANT_LABELS[q],
                "label": QUANT_LABELS[q],
                "mb": total,
                "total_gb": round(total / 1024.0, 2),
                # rule of thumb for GGML inference: model bytes x 1.15 plus a
                # ~0.3 GB runtime floor (matches the measured 1.7B-Q4 working
                # set of ~2 GB on this machine)
                "vram_gb": round(total / 1024.0 * 1.15 + 0.3, 1),
                "files": list(combo_files(combo, q) or ()),
            })
    out.sort(key=lambda e: (e["size"], e["variant"], QUANTS.index(e["quant"])))
    return out


# ---- shared HTTP helpers --------------------------------------------------------

def _get(url: str, timeout: float = 15.0):
    return urlopen(Request(url, headers={"User-Agent": _UA, "Accept": "*/*"}), timeout=timeout)


def _resolve(url: str, timeout: float = 12.0) -> Tuple[Optional[str], Optional[int], Optional[str], Optional[str]]:
    """Follow redirects manually -> (final_url, content_length, etag, last_modified)."""
    cur = url
    for _ in range(6):
        try:
            resp = urlopen(Request(cur, headers={"User-Agent": _UA, "Accept-Encoding": "identity"}),
                           timeout=timeout)
        except HTTPError as e:
            if e.code in (301, 302, 303, 307, 308) and e.headers.get("Location"):
                cur = urljoin(cur, e.headers["Location"])
                continue
            return None, None, None, None
        except Exception:
            return None, None, None, None
        try:
            cl = resp.headers.get("Content-Length")
            return (resp.geturl(), int(cl) if cl and cl.isdigit() else None,
                    resp.headers.get("ETag"), resp.headers.get("Last-Modified"))
        finally:
            try:
                resp.close()
            except Exception:
                pass
    return None, None, None, None


def _open_range(url: str, start: int):
    headers = {"User-Agent": _UA, "Accept-Encoding": "identity"}
    if start > 0:
        headers["Range"] = f"bytes={start}-"
    return urlopen(Request(url, headers=headers), timeout=40.0)


def _speed_test(url: str, timeout: float = 6.0) -> Optional[float]:
    """Seconds until the first response bytes arrive; None on failure (DNS,
    TLS, timeout, HTTP error). TLS-tolerant (cert problems are the norm
    behind corporate proxies; connectivity is what we actually measure).
    Small files are fine: any 2xx body counts, the size threshold was a
    rookie mistake on 4.9 KB READMEs."""
    import ssl
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    t0 = time.monotonic()
    resp = None
    try:
        resp = urlopen(Request(url, headers={"User-Agent": _UA}), timeout=timeout, context=ctx)
        got = b""
        while len(got) < 131072:
            b = resp.read(min(32768, 131072 - len(got)))
            if not b:
                break
            got += b
        return time.monotonic() - t0 if got else None
    except Exception:
        return None
    finally:
        try:
            if resp is not None:
                resp.close()
        except Exception:
            pass


def pick_hf_endpoint() -> Dict[str, Any]:
    """Fastest reachable Hugging Face endpoint for this session (never cached,
    so a VPN toggle is picked up on the next click). Winner = fastest
    first-128-KB time."""
    scores = {c: _speed_test(f"{c}/{HF_REPO}/resolve/main/README.md") for c in HF_ENDPOINTS}
    reach = {k: v for k, v in scores.items() if v is not None}
    if not reach:
        return {"endpoint": None, "scores": scores}
    return {"endpoint": min(reach, key=lambda k: reach[k]), "scores": scores}


def probe_endpoint(explicit: Optional[str] = None) -> Dict[str, Any]:
    """Source status for the UI: per-endpoint timings + the winner. An explicit
    user override passes straight through (auto=False)."""
    if explicit in HF_ENDPOINTS:
        return {"endpoint": explicit, "auto": False, "scores": {}}
    r = pick_hf_endpoint()
    r["auto"] = True
    return r


# ---- state machine: one JSON on disk, atomic replace, workers poll cancel --------

_LOCK = threading.Lock()


def _state_file() -> Path:
    return _p.hermes_home() / "yache-download.json"


def _cancel_marker() -> Path:
    return _state_file().with_suffix(".json.cancel")


def _save(st: Dict[str, Any]) -> None:
    st["updated_at"] = time.time()
    p = _state_file()
    tmp = p.with_name(f"{p.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(st, fh, ensure_ascii=False, indent=1)
        os.replace(tmp, p)
    except OSError:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


def load_state() -> Optional[Dict[str, Any]]:
    try:
        with open(_state_file(), encoding="utf-8") as fh:
            st = json.load(fh)
        return st if isinstance(st, dict) and st.get("kind") else None
    except (OSError, ValueError):
        return None


def _busy(st: Optional[Dict[str, Any]]) -> bool:
    return bool(st) and st.get("state") in ("running", "cancelling", "extracting", "probing")


def cancel_download() -> Dict[str, Any]:
    st = load_state()
    if not _busy(st):
        return {"ok": False, "detail": _i18n.ui("当前没有正在下载的任务", "nothing is downloading")}
    st["state"] = "cancelling"  # type: ignore[index]
    _save(st)
    try:
        _cancel_marker().write_text(str(time.time()), encoding="utf-8")
    except OSError:
        pass
    return {"ok": True}


def _cancel_requested(st: Dict[str, Any]) -> bool:
    if _cancel_marker().exists():
        return True
    fresh = load_state()
    return bool(fresh and fresh.get("state") == "cancelling")


def _clear_cancel_marker() -> None:
    try:
        if _cancel_marker().exists():
            _cancel_marker().unlink()
    except OSError:
        pass


def _new_state(kind: str, target: str) -> Dict[str, Any]:
    _clear_cancel_marker()
    return {
        "kind": kind, "state": "running", "note": None, "target": str(target),
        "steps_total": 1, "step_label": None, "total_bytes": 0, "done_bytes": 0,
        "base_bytes": 0, "mbps": None, "eta_s": None, "endpoint": None,
        "f_etag": None, "f_lmod": None, "f_sha": None,
        "started_at": time.time(), "updated_at": time.time(),
    }


# ---- the one download loop (model files and the GPU zip share it) ----------------

def _download_into(url: str, dest: Path, st: Dict[str, Any],
                   want_sha: Optional[str] = None,
                   total_hint: Optional[int] = None) -> str:
    """Return "ok" | "cancelled" | "interrupted" | "error". Resumable through
    .part + Range; a stale .part (server identity moved) is dropped and the
    file restarted. Progress is persisted every 0.8 s so the panel can poll."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    final, total, etag, lmod = _resolve(url)
    if total is None:
        total = total_hint
    if have:
        stale = ((total is not None and have > total)
                 or (etag and st.get("f_etag") and etag != st["f_etag"])
                 or (lmod and st.get("f_lmod") and lmod != st["f_lmod"]))
        if stale:
            try:
                part.unlink()
            except OSError:
                pass
            have = 0
    st["f_etag"], st["f_lmod"], st["f_sha"] = etag, lmod, want_sha
    resp = None
    try:
        resp = _open_range(final or url, have)
    except HTTPError as e:
        if have and e.code in (404, 416, 500, 501, 502, 503):
            try:
                part.unlink()
            except OSError:
                pass
            have = 0
            try:
                resp = _open_range(final or url, 0)
            except Exception as e2:
                st["note"] = _i18n.ui(f"连接失败: {str(e2)[:140]}", f"connection failed: {str(e2)[:140]}")
                return "interrupted"
        else:
            st["note"] = f"HTTP {e.code}"
            return "interrupted" if e.code >= 500 else "error"
    except (URLError, OSError) as e:
        st["note"] = _i18n.ui(f"连接失败: {str(e)[:140]}", f"connection failed: {str(e)[:140]}")
        return "interrupted"
    except Exception as e:
        st["note"] = str(e)[:160]
        return "error"
    try:
        status = getattr(resp, "status", 200)
        if have and status != 206:  # server ignored the Range header: restart
            have = 0
            try:
                part.unlink()
            except OSError:
                pass
        cl = resp.headers.get("Content-Length")
        file_total = have + int(cl) if (cl and cl.isdigit() and status == 206) else \
            (int(cl) if cl and cl.isdigit() else (total or 0))
        base = int(st.get("base_bytes") or 0)
        t0 = time.monotonic()
        start_have = have
        last_beat = 0.0
        with open(part, "ab" if have else "wb") as fh:
            while True:
                if _cancel_requested(st):
                    st["state"] = "cancelled"
                    st["note"] = None
                    _clear_cancel_marker()
                    return "cancelled"
                try:
                    chunk = resp.read(_CHUNK)
                except Exception:
                    chunk = b""
                if not chunk:
                    break
                fh.write(chunk)
                have += len(chunk)
                st["done_bytes"] = base + have
                now = time.monotonic()
                if now - last_beat >= 0.8:
                    el = max(0.5, now - t0)
                    moved = have - start_have
                    st["mbps"] = round(moved / el / 1048576, 2) if moved > 0 else None
                    if file_total and moved > 0:
                        rem = (file_total - have) * el / moved
                        st["eta_s"] = round(rem) if 0 < rem < 86400 else None
                    _save(st)
                    last_beat = now
        if file_total and have < file_total:
            st["note"] = _i18n.ui("连接断了——按「续传」重试", "connection dropped — press resume")
            _save(st)
            return "interrupted"
        if want_sha and part.exists():
            h = hashlib.sha256()
            with open(part, "rb") as fh:
                for blk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(blk)
            if h.hexdigest() != want_sha:
                try:
                    part.unlink()
                except OSError:
                    pass
                st["f_etag"] = st["f_lmod"] = st["f_sha"] = None
                st["note"] = _i18n.ui("校验和不匹配——文件已丢弃，按「续传」重下", "checksum mismatch — file discarded, press resume to retry")
                _save(st)
                return "interrupted"
        if dest.exists():
            dest.unlink()
        os.replace(part, dest)
        st["f_etag"] = st["f_lmod"] = st["f_sha"] = None
        st["done_bytes"] = base + (total_hint or file_total or have)
        _save(st)
        return "ok"
    finally:
        try:
            resp.close()
        except Exception:
            pass


# ---- model pipeline ----------------------------------------------------------------

def _target_layout(target: str) -> Tuple[Path, Path]:
    r"""(models_dir, tts_base_dir) for a user-picked directory. A directory
    literally named "models" is treated as the models dir itself; anything
    else gets a models\ subdir, mirroring the layout the scanner expects."""
    t = Path(target).expanduser().resolve()
    if t.name.lower() == "models":
        return t, t.parent
    return t / "models", t


def _load_into_settings(base: Path, quant: str) -> Tuple[bool, Optional[str]]:
    """The same verification gate the manual directory-load path uses: the
    GGUF metadata scanner must find a talker+tokenizer pair. On success the
    base dir is persisted into the settings (plus the downloaded quant as
    the preferred talker tier)."""
    scan = _vy.scan_models_dir(base)
    if not scan.get("variants"):
        return False, str(scan.get("detail") or "downloaded files failed the GGUF metadata check")
    if not (scan.get("tokenizer") or {}).get("preferred"):
        return False, "tokenizer missing next to the talker"
    data = _p.load_data()
    data["tts_base_dir"] = str(base)
    v = data.setdefault("voice", {})
    if isinstance(v, dict):
        v.setdefault("variant", "base")
    t = data.setdefault("tts", {})
    if isinstance(t, dict):
        t["quant"] = quant
    _p.save_data(data)
    return True, None


def _run_model(st: Dict[str, Any]) -> None:
    try:
        combo, quant, target = st["combo"], st["quant"], st["target"]
        files = combo_files(combo, quant)
        if not files or not target:
            st.update(state="error", note="invalid combo/quant")
            _save(st)
            return
        pick = pick_hf_endpoint()
        base = pick.get("endpoint")
        if not base:
            st.update(state="error",
                      note="huggingface.co and hf-mirror.com are both unreachable — "
                           "turn on your proxy/VPN and press resume")
            _save(st)
            return
        st["endpoint"] = base
        models_dir, tts_base = _target_layout(target)
        names = list(files)
        plan = [n for n in names if not (models_dir / n).exists()]
        st["steps_total"] = max(1, len(plan))
        st["base_bytes"] = int(st.get("base_bytes") or 0)
        # optimistic denominator from the catalog (right even when the remote
        # HEAD fails); refined per-file from Content-Length inside the loop
        total = 0
        for n in plan:
            for e in catalog():
                if n in e["files"]:
                    total += e["mb"] * 1048576
                    break
        st["total_bytes"] = int(st["base_bytes"]) + total
        _save(st)
        for label in plan:
            st["step_label"] = label
            _save(st)
            res = _download_into(f"{base}/{HF_REPO}/resolve/main/{quote(label)}",
                                 models_dir / label, st)
            if res != "ok":
                if res != "cancelled":
                    st["state"] = "interrupted"
                _save(st)
                return
        st["step_label"] = None
        st.update(state="probing", mbps=None, eta_s=None)
        st["target"] = str(tts_base)
        _save(st)
        ok, detail = _load_into_settings(tts_base, quant)
        if ok:
            _clear_cancel_marker()
            st.update(state="done", note=None, done_bytes=st.get("total_bytes"), pct=100.0)
        else:
            st.update(state="installed_unverified", note=detail)
        _save(st)
    except Exception as e:
        st.update(state="error", note=f"{type(e).__name__}: {str(e)[:160]}")
        _save(st)


def start_model_download(combo: str, quant: str, target: str) -> Dict[str, Any]:
    with _LOCK:
        st = load_state()
        if _busy(st):
            return {"ok": False, "detail": _i18n.ui("已有一个下载任务在跑——先取消它或等它完成",
                        "a download is already running — cancel it or wait")}
        files = combo_files(combo, quant)
        if not files:
            return {"ok": False, "detail": _i18n.ui("模型/量化组合名无效", "invalid model/quant combination")}
        t = Path(target).expanduser()
        try:
            t.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            return {"ok": False, "detail": _i18n.ui(f"无法创建目录: {str(e)[:120]}", f"cannot create directory: {str(e)[:120]}")}
        models_dir, _ = _target_layout(str(t))
        whole = [p for p in (models_dir / n for n in files) if p.exists()]
        if len(whole) == len(files):
            return {"ok": False, "detail": _i18n.ui(f"{_L(*MODEL_LABELS.get(tuple((combo or '').split('-', 1)), (combo, combo)))}这一对文件在这个目录里已经齐了——想下别的组合请换个下拉档位；想直接用这些文件请改走「从目录装载」",
                             f"{combo} is already complete in this directory — pick another tier to download something new, or use directory-load")}
        # partial leftovers from a cancelled run are adopted: whole existing
        # files are kept, only the missing ones get downloaded
        st = _new_state("model", str(t))
        st.update(combo=combo, quant=quant, done_bytes=0, base_bytes=0)
        _save(st)
        threading.Thread(target=_run_model, args=(st,), daemon=True,
                         name="yache-dl-model").start()
        return {"ok": True}


def resume_download() -> Dict[str, Any]:
    """Continue the last interrupted/cancelled job (model: from the .part
    files; gpu: re-stream the zip — its GitHub release asset is the durable
    artifact and the state file keeps nothing but progress)."""
    with _LOCK:
        st = load_state()
        if not st:
            return {"ok": False, "detail": _i18n.ui("没有可续传的下载任务", "no download to resume")}
        if _busy(st):
            return {"ok": False, "detail": _i18n.ui("已有一个下载任务在跑", "a download is already running")}
        if st.get("state") == "done":
            return {"ok": False, "detail": _i18n.ui("下载已经完成了", "already finished")}
        st["state"] = "running"
        st["note"] = None
        st["base_bytes"] = st.get("done_bytes", 0)
        st.pop("mbps", None)
        st.pop("eta_s", None)
        _clear_cancel_marker()
        target = _run_model if st.get("kind") == "model" else _run_gpu
        _save(st)
        threading.Thread(target=target, args=(st,), daemon=True,
                         name=f"yache-dl-resume-{st.get('kind')}").start()
        return {"ok": True}


# ---- GPU accelerator pipeline --------------------------------------------------------

def _gpu_urls(base: str) -> Tuple[str, str]:
    root = f"{base}/{GPU_REPO}/releases/download/{GPU_TAG}"
    return f"{root}/{GPU_ASSET}", f"{root}/{GPU_META}"


def pick_gpu_endpoint() -> str:
    """Direct GitHub first; the public gh-proxy mirror as the CN fallback."""
    for cand in GPU_ENDPOINTS:
        _, meta = _gpu_urls(cand)
        if _speed_test(meta, timeout=6.0) is not None:
            return cand
    return GPU_ENDPOINTS[0]


def _extract_pack(zip_path: Path, dest: Path) -> Optional[str]:
    """Unpack atomically: fresh staging dir, completeness check, swap at the
    end. Returns an error string or None. A single stray top-level folder
    (the usual zip courtesy) is flattened away."""
    staging = dest.with_name(dest.name + ".new")
    try:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        with zipfile.ZipFile(zip_path) as zf:
            for nm in zf.namelist():
                if nm.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:", nm) or ".." in nm.split("/"):
                    return f"unsafe path inside archive: {nm}"
            zf.extractall(staging)
        entries = list(staging.iterdir())
        while len(entries) == 1 and entries[0].is_dir():
            staging = entries[0]
            entries = list(staging.iterdir())
        missing = [f for f in ("qwen.dll", "ggml-cuda.dll", "cublas64_12.dll", "cudart64_12.dll")
                   if not (staging / f).is_file()]
        if missing:
            return "archive incomplete, missing: " + ", ".join(missing)
        old = dest.with_name(dest.name + ".old")
        if dest.exists():
            if old.exists():
                shutil.rmtree(old, ignore_errors=True)
            os.rename(dest, old)
        os.rename(staging, dest)
        shutil.rmtree(old, ignore_errors=True)
        return None
    except Exception as e:
        return f"{type(e).__name__}: {str(e)[:140]}"
    finally:
        try:
            top = dest.with_name(dest.name + ".new")
            if top.exists():
                shutil.rmtree(top, ignore_errors=True)
        except OSError:
            pass


def _run_gpu(st: Dict[str, Any]) -> None:
    try:
        base = pick_gpu_endpoint()
        st["endpoint"] = base
        dl, meta = _gpu_urls(base)
        _save(st)
        want_sha, zip_bytes = None, None
        try:
            with _get(meta, 20.0) as r:
                md = json.loads(r.read().decode("utf-8"))
            want_sha = str(md.get("sha256") or "").lower()
            zip_bytes = int(md.get("bytes") or 0) or None
        except Exception as e:
            st.update(state="error",
                      note=f"cannot read metadata.json from GitHub ({type(e).__name__}) — "
                           f"proxy/VPN needed? {str(e)[:80]}")
            _save(st)
            return
        if not re.fullmatch(r"[0-9a-f]{64}", want_sha or ""):
            st.update(state="error", note="metadata.json has no sha256 digest")
            _save(st)
            return
        cdir = _p.cuda_pack_dir()
        st["target"] = str(cdir)
        st["steps_total"] = 2
        st["base_bytes"] = int(st.get("done_bytes") or 0)
        st["total_bytes"] = st["base_bytes"] + (zip_bytes or 0)
        st["step_label"] = GPU_ASSET
        _save(st)
        zip_path = _p.hermes_home() / GPU_ASSET
        res = _download_into(dl, zip_path, st, want_sha=want_sha, total_hint=zip_bytes)
        if res != "ok":
            if res != "cancelled":
                st["state"] = "interrupted"
            _save(st)
            return
        st.update(state="extracting", step_label=None, mbps=None, eta_s=None)
        _save(st)
        err = _extract_pack(zip_path, cdir)
        try:
            if zip_path.exists():
                zip_path.unlink()
        except OSError:
            pass
        if err:
            st.update(state="error", note=err)
        elif not _p.cuda_pack_ready(cdir):
            st.update(state="error", note="extracted pack failed the completeness check")
        else:
            st.update(state="done", note=None, done_bytes=st.get("total_bytes"), pct=100.0)
        _save(st)
    except Exception as e:
        st.update(state="error", note=f"{type(e).__name__}: {str(e)[:160]}")
        _save(st)


def start_gpu_download(gate_ok: bool, gate_reason: str = "") -> Dict[str, Any]:
    with _LOCK:
        st = load_state()
        if _busy(st):
            return {"ok": False, "detail": _i18n.ui("已有一个下载任务在跑——先取消它或等它完成",
                        "a download is already running — cancel it or wait")}
        if not gate_ok:
            return {"ok": False, "detail": gate_reason or _i18n.ui("先下载并装载一个模型", "download and load a model first")}
        st = _new_state("gpu", str(_p.cuda_pack_dir()))
        _save(st)
        threading.Thread(target=_run_gpu, args=(st,), daemon=True, name="yache-dl-gpu").start()
        return {"ok": True}


# ---- read side for the panel ----------------------------------------------------------

def gpu_status() -> Dict[str, Any]:
    """Installed-ness + the gate that unlocks the download button. The gate is
    computed by the backend, never trusted from the frontend: a model pair
    must scan clean from the currently loaded tts_base_dir."""
    cdir = _p.cuda_pack_dir()
    scan = _vy.scan_models_dir(_p.tts_base_dir())
    # same gate the manual directory-load save path uses: a scan-clean pair
    gate_ok = bool(scan.get("variants")) and bool((scan.get("tokenizer") or {}).get("preferred"))
    return {
        "installed": _p.cuda_pack_ready(cdir),
        "dir": str(cdir),
        "gate_ok": gate_ok,
        "gate_reason": None if gate_ok else
            _i18n.ui("还没有装载任何可用的模型组合——先下载模型或从目录装载",
                  "no usable model pair is loaded yet — download or load a model first"),
        "model_variants": scan.get("variants") or [],
        "tokenizer": (scan.get("tokenizer") or {}).get("preferred"),
    }


def download_status() -> Dict[str, Any]:
    st = load_state()
    if st is None:
        return {"state": "idle"}
    st = dict(st)
    tb = st.get("total_bytes") or 0
    db = st.get("done_bytes") or 0
    if st.get("state") in ("running", "cancelling") and tb > 0:
        st["pct"] = round(min(99.5, db * 100.0 / tb), 1)
    elif st.get("state") in ("probing", "extracting", "installed_unverified"):
        st["pct"] = 99.5
    elif st.get("state") == "done":
        st["pct"] = 100.0
    # a hard-killed worker leaves state=running with a dead timestamp: the
    # live writer beats every 0.8 s, so 120 s of silence means it is gone
    if st.get("state") in ("running", "cancelling") and \
            time.time() - float(st.get("updated_at") or 0) > 120:
        st["state"] = "interrupted"
        st["note"] = st.get("note") or _i18n.ui("下载停滞了——按「续传」重试", "download stalled — press resume")
    return st
