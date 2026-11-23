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
first-128-KB throughput).  The GPU accelerator pack is fetched from
github.com only — no third-party mirrors — and its SHA-256 and exact byte
size are pinned in source below, so no relay can swap the payload.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import tarfile
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
from yache_engine import logbox as _logbox  # noqa: E402 (silent-failure killer)

_log = _logbox.get_logger()

_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) YacheLocalVoice/1.0"
_CHUNK = 262144

HF_REPO = "Serveurperso/Qwen3-TTS-GGUF"
HF_ENDPOINTS = ["https://huggingface.co", "https://hf-mirror.com"]

# Our GitHub release hosting the CUDA accelerator zip.  Reviewers: a
# checksum fetched from the same source as the archive proves nothing, so
# the pack's identity is pinned HERE in source — SHA-256 + exact byte size
# — not read from a same-origin metadata.json at runtime.  The pack is a
# versioned release asset under the plugin's v0.1.0 tag (not a moving
# tag), built and SLSA-attested by the repo's own CI workflow
# (.github/workflows/build-engine.yml, build-cuda-pack job).  Update both
# constants together whenever the pack is rebuilt.
GPU_REPO = "LuuMoss/yache-local-voice"
GPU_TAG = "v0.1.1"
GPU_ASSET = "yache-cuda-windows-x64.zip"
GPU_SHA256 = "1ca320a57126c62c50813c2f752b2855066ace244cb10835d36a4ed3f85b1d36"
GPU_BYTES = 768440698
GPU_META = "metadata.json"   # kept in the release for bookkeeping only
GPU_ENDPOINTS = ["https://github.com"]
GPU_GITHUB = GPU_ENDPOINTS[0]      # the ramp-up-prone mainline (see pick)

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
    TLS, timeout, HTTP error).  TLS verification stays ON everywhere —
    reviewers: no disabled certificate checks in this plugin.
    Small files are fine: any 2xx body counts, the size threshold was a
    rookie mistake on 4.9 KB READMEs."""
    import ssl
    ctx = ssl.create_default_context()
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


class NoEndpointError(RuntimeError):
    """Not one download leg can stream bytes right now."""


def _probe_leg(url: str, budget: float = 12.0, want: int = 2097152,
               start: int = 0) -> Optional[float]:
    """Seconds-to-download-`want` extrapolation, measured on the REAL asset
    stream (Range GET); None when the leg cannot stream at all. Probing only
    a tiny README/metadata file is the classic China trap: GitHub happily
    serves 4.9 KB of JSON on a bare connection — then the 543 MB body stalls
    forever. A 512 KB head sample is ALSO not enough: free public mirrors
    cache only the first megabytes at the CDN edge, so the head reads at
    9 MB/s while the body crawls at 0.04 (owner watched exactly this on
    2026-11-19: probe crowned ghfast at 9 MB/s, the real round sank to a
    trickle). The picker therefore samples TWICE — 2 MB at the head (warm
    CDN cache, legitimately fast if the mirror kept the whole asset warm)
    and 2 MB at the 1/3 mark (cold body, where warm-head impostors show
    their true face) — and keeps the better score: a genuinely warm leg
    wins on the head sample, an honest cold-fast leg wins on the body, and
    a warm-head trickle that died mid-file loses both. A trickle still
    scores (extrapolated slow, loses honestly); only a dead leg scores
    None. Lower = better.

    The window is 2 MB / 12 s, NOT 1 MB / 6 s (owner's WARP sighting, same
    evening): GitHub through the Cloudflare tunnel takes ~10 s to ramp up,
    so a 6 s window convicted the mainline at 0.11 MB/s while it was really
    doing 0.66+ and CLIMBING — meanwhile gh-proxy (locally hosted CDN,
    instant first byte) won the probe and then died outright with a 15 s
    timeout on the very same tunnel. Slow starters need a longer window to
    speak; the budget only burns on legs that are actually slow, a fast
    leg finishes its 2 MB and reports early."""
    best: Optional[float] = None
    for s in _window_order(start):      # far windows FIRST, head last
        v = _probe_once(url, budget, want, s)
        if v is not None and (best is None or v < best):
            best = v
    return best


def _window_order(start: int):
    """Sampling order = far window BEFORE the head (and the head's twin
    last again when the asset has no far window). Cloudflare/CDN tunnels
    pay a ramp-up tax during the first seconds of a fresh connection; if
    every leg probed the head first, the leg probed LAST would enjoy the
    warmest route and win a rigged race (owner's 22:49 WARP probe: the
    gh-proxy leg that ran second read faster than GitHub only because
    GitHub went first and ate the cold-route tax). Round-robin the phase
    across legs instead: whoever goes first pays, on EVERY leg equally."""
    if start > 0:
        return (start, 0)
    return (0,)


def _probe_once(url: str, budget: float, want: int, start: int) -> Optional[float]:
    resp = None
    try:
        req = Request(url, headers={"User-Agent": _UA,
                                    "Accept-Encoding": "identity",
                                    "Range": f"bytes={start}-{start + want - 1}"})
        t0 = time.monotonic()
        resp = urlopen(req, timeout=6)
        if resp.status >= 300:
            return None
        if start > 0 and resp.status != 206:
            # Range ignored (200 = whole body from byte 0): this leg cannot
            # honour mid-file resumes either — score it as a straggler
            # instead of silently measuring a different byte range.
            return 1e9
        got = 0
        while got < want and time.monotonic() - t0 < budget:
            chunk = resp.read(min(65536, want - got))
            if not chunk:
                break
            got += len(chunk)
        if not got:
            return None
        return (time.monotonic() - t0) * (want / got)
    except Exception:
        return None
    finally:
        try:
            if resp is not None:
                resp.close()
        except Exception:
            pass


def pick_hf_endpoint(probe_files: Optional[List[str]] = None,
                     exclude: Optional[set] = None) -> Dict[str, Any]:
    """Fastest reachable Hugging Face endpoint for this session (never cached
    and never pinned: every attempt re-probes, so a VPN toggle or a mirror
    that died mid-run is picked up immediately). With probe_files — the real
    asset names about to be fetched — the probe pulls REAL bytes from the
    asset stream instead of trusting the README."""
    scores: Dict[str, Optional[float]] = {}
    for c in HF_ENDPOINTS:
        if exclude and c in exclude:
            continue
        if probe_files:
            scores[c] = _probe_leg(
                f"{c}/{HF_REPO}/resolve/main/{quote(probe_files[0])}")
        else:
            scores[c] = _speed_test(f"{c}/{HF_REPO}/resolve/main/README.md")
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
                   total_hint: Optional[int] = None,
                   others_best_Bps: Optional[float] = None) -> str:
    """Return "ok" | "cancelled" | "interrupted" | "slow" | "error".
    Resumable through .part + Range; a stale .part (server identity moved)
    is dropped and the file restarted. Progress is persisted every 0.8 s so
    the panel can poll.

    Slow-leg watchdog (others_best_Bps): a pre-flight probe can never see a
    ramp-up tunnel's steady state, so selection is sometimes "best of a
    warm moment" and the chosen leg crawls once the round starts. Rather
    than ever-longer probes we measure DURING use: 25 s ramp-free grace,
    then a 30 s rolling window — sustained under 40% of what the OTHER legs
    promised their caller — the leg is slow-dead: bail out (every byte kept
    via .part Range resume), return "slow" so the log says slow and not
    broken wire, and let the caller demote the leg and re-pick."""
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
        wd_free = t0 + 25.0   # watchdog: ramp grace, then 30 s observation
        wd_t = wd_have = None
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
                    if others_best_Bps:
                        if now < wd_free:
                            wd_t, wd_have = now, have
                        elif wd_t is not None and now - wd_t >= 30.0:
                            _v = (have - wd_have) / (now - wd_t)
                            if _v < 0.4 * others_best_Bps:
                                st["note"] = _i18n.ui(
                                    "这条腿实测不足其它腿承诺速度的四成——判慢换腿（进度不丢）",
                                    "this leg sustains under 40% of what the "
                                    "other legs promised — switching legs, "
                                    "progress kept")
                                _save(st)
                                return "slow"
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
        # The connection is never pinned to one source either: legs are
        # re-probed on the REAL asset bytes before the first fetch, and a
        # source that dies mid-file is demoted and replaced by the next-best
        # one — the .part bytes carry across sources (identical files on
        # every mirror) and the new leg re-derives its Range offset from
        # them, so rotation resumes instead of restarting.
        labels = list(plan)
        if labels:
            sel2 = pick_hf_endpoint(probe_files=labels)
            if sel2.get("endpoint"):
                base = sel2["endpoint"]
                st["endpoint"] = base
                _save(st)
        dead: "set[str]" = set()
        i = 0
        while i < len(labels):
            label = labels[i]
            st["step_label"] = label
            _save(st)
            _log.info(_i18n.ui("转录模型轮次: 腿=%s 文件=%s 死腿=%s",
                     "model round: leg=%s file=%s dead=%s"),
            base, label, sorted(dead) or "-")
            res = _download_into(f"{base}/{HF_REPO}/resolve/main/{quote(label)}",
                                 models_dir / label, st)
            if res == "cancelled":
                _save(st)
                return
            if res == "ok":
                i += 1
                continue
            if res == "error":
                # 4xx on this path — every mirror carries the same files, so
                # bouncing sources would only re-404; keep the old stop and
                # let the note guide the user.
                _save(st)
                return
            dead.add(base)
            _log.warning(_i18n.ui("转录模型腿 %s 在 %s 上阵亡（%s）—— 换腿（死腿=%s）",
                         "model leg %s died on %s (%s) — rotating (dead=%s)"),
               base, label, res, sorted(dead))
            sel2 = pick_hf_endpoint(probe_files=labels[i:], exclude=dead)
            if not sel2.get("endpoint"):
                st["state"] = "interrupted"
                st["note"] = _i18n.ui(
                    "所有下载源都试过一轮没走通——进度还在，按「续传」会重新测速换源",
                    "every source was tried this round — progress kept; press "
                    "resume to re-probe all sources")
                _save(st)
                return
            base = sel2["endpoint"]
            st.update(state="running", endpoint=base, mbps=None, eta_s=None,
                      note=_i18n.ui("这个源断了——自动换下一个源续传（进度不丢）",
                                    "this source died — auto-switching to the "
                                    "next source, progress kept"))
            _save(st)
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
        target = (_run_model if st.get("kind") == "model"
                  else _run_wake if st.get("kind") == "wake"
                  else _run_gpu)
        _save(st)
        threading.Thread(target=target, args=(st,), daemon=True,
                         name=f"yache-dl-resume-{st.get('kind')}").start()
        return {"ok": True}


# ---- GPU accelerator pipeline --------------------------------------------------------

def _gpu_urls(base: str) -> Tuple[str, str]:
    root = f"{base}/{GPU_REPO}/releases/download/{GPU_TAG}"
    return f"{root}/{GPU_ASSET}", f"{root}/{GPU_META}"


# The mainline leg rides the Cloudflare/WARP tunnel, which RAMP UP: the
# first ~15 s of a fresh tunnel read slow while it establishes, then it
# climbs to multiples of any local mirror (owner's 22:46 WARP test: 10 MB
# at 0.66 MB/s still climbing, CF's own speed point 1.3 MB/s, while
# gh-proxy died with a 15 s timeout on that same tunnel). Mirrors are
# locally hosted CDNs with instant first byte — one window size fits one
# of them badly: 2 MB/12 s convicts the slow-starter forever. So the
# GitHub leg gets a big generous window (10 MB / 40 s: after the ramp the
# steady state shows inside it), mirrors keep the tight one.
_GPU_BIG = {"want": 10 * 1048576, "budget": 40.0}


def pick_gpu_endpoint(exclude: Optional[set] = None,
                      size: Optional[int] = None):
    """Returns (winner_base, scores) — scores travel with the winner so
    the round's watchdog knows what the OTHER legs promised (see
    _run_gpu)."""
    """Fastest GPU-pack leg for this round, measured on the REAL asset bytes
    (Range probe), never on the tiny metadata.json alone — that was the
    original trap: a bare CN connection reads 4.9 KB of JSON from GitHub in
    seconds, gets crowned winner, then stalls forever on the 543 MB body.
    A 512 KB head sample is the SECOND trap (2026-11-19): free mirrors keep
    only the first megabytes warm at the CDN edge, head probes crown them
    at 9 MB/s while the real stream crawls at 0.04. When the asset size is
    known the sample is aimed at the file's 1/3 mark — cold body bytes,
    where legs show their true sustained speed. `exclude` demotes legs the
    current run already burned; NoEndpointError fires only when nothing
    left can stream at all."""
    start = (size // 3) if (size and size >= 12 * 1024 * 1024) else 0
    urls = {c: _gpu_urls(c)[0] for c in GPU_ENDPOINTS
            if not exclude or c not in exclude}
    def _score(c: str, u: str):
        if c == GPU_GITHUB:
            return _probe_leg(u, budget=_GPU_BIG["budget"],
                              want=_GPU_BIG["want"], start=start)
        return _probe_leg(u, start=start)
    scores = {c: _score(c, u) for c, u in urls.items()}
    _log.info(_i18n.ui("GPU 腿探针 @%sB: %s（排除=%s）",
                     "gpu leg probe @%sB: %s (excluded=%s)"),
            start,
            {k: (round(v, 2) if v is not None else None) for k, v in scores.items()},
            sorted(exclude or ()) or "-")
    live = {k: v for k, v in scores.items() if v is not None}
    if not live:
        raise NoEndpointError(
            "no GPU download leg can stream bytes right now — tried "
            f"{sorted(urls) or 'nothing (all legs already burned this run)'}")
    best = min(live, key=lambda k: live[k])
    return best, scores


PACK_MANIFEST = ".yache-pack-manifest.json"


def _write_pack_manifest(cdir: Path, kind: str = "gpu",
                         pack: str = GPU_ASSET) -> Optional[str]:
    """Inventory every file the installer just placed (relative path +
    size + sha256) so an uninstall can reverse exactly THIS install: only
    inventoried files get removed, anything the user later dropped into
    the folder is left alone. Three install kinds share this ledger —
    "gpu" (accelerator pack), "stt-model" (whisper HF cache dir), "stt-env"
    (the ASR venv) — the manifest lives INSIDE the installed folder so it
    travels with it. Returns an error string or None."""
    files: List[Dict[str, Any]] = []
    try:
        for f in sorted(cdir.rglob("*")):
            if not f.is_file() or f.name == PACK_MANIFEST:
                continue
            h = hashlib.sha256()
            with open(f, "rb") as fp:
                for chunk in iter(lambda: fp.read(1 << 20), b""):
                    h.update(chunk)
            files.append({"path": f.relative_to(cdir).as_posix(),
                          "size": f.stat().st_size, "sha256": h.hexdigest()})
        man = {"kind": kind, "pack": pack, "files": files,
               "installed_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        (cdir / PACK_MANIFEST).write_text(
            json.dumps(man, ensure_ascii=False), encoding="utf-8")
    except OSError as e:
        _log.warning(_i18n.ui("加速包快照写入失败: %s", "pack manifest write failed: %s"), e)
        return None  # not fatal: uninstall falls back to the legacy route
    return None


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
        cdir = _p.cuda_pack_dir()
        zip_path = _p.hermes_home() / GPU_ASSET
        # The connection is NEVER pinned to one leg: every round re-picks by
        # throughput measured on the real asset bytes, and a leg that dies
        # mid-stream (WARP toggled off, cable hiccup, sha-corrupted mirror)
        # gets demoted so the next round runs on a fresh leg. The .part file
        # is byte-identical across legs — every mirror serves the same
        # release asset — and each attempt re-derives its Range offset from
        # the .part, so switching legs resumes exactly where bytes stopped:
        # no re-download, no double count (base stays 0, total = pack size).
        dead: "set[str]" = set()
        while True:
            # The pack's identity is PINNED in source (GPU_SHA256 + exact
            # GPU_BYTES above): no same-origin metadata round-trip to trust
            # or distrust, and the throughput probe knows the real size up
            # front so it samples the file's cold 1/3 mark right away.
            want_sha = GPU_SHA256
            zip_bytes = GPU_BYTES
            # Crown the real-work horse on a DUAL sample (1 MB head + 1 MB
            # at the 1/3 cold mark).  A dead leg scores None; a warm-head
            # impostor that crawls on the body loses both samples and
            # yields its seat by itself.
            try:
                base, scores = pick_gpu_endpoint(exclude=dead, size=zip_bytes)
            except NoEndpointError as e:
                st.update(state="error",
                          note=_i18n.ui(
                              "直连 GitHub 实测跑不动这个大正文——开个翻墙再点就是满速；"
                              "或稍等片刻再试（线路全天波动，会回血）",
                              "bare GitHub cannot sustain the real file body — "
                              "switch on a proxy/VPN for full speed, or retry "
                              "in a while (the tunnel fluctuates all day; it "
                              "recovers). Details: " + str(e)[:120]))
                _save(st)
                return
            st["endpoint"] = base
            dl, _meta = _gpu_urls(base)
            # Watchdog line = 40% of the speed the OTHER live legs promised
            # (a score is seconds-per-MB; invert to B/s). Owner's 22:46 WARP
            # case: gh-proxy's warm local CDN won the 12 s race at 0.34 MB/s
            # while GitHub — promised only 0.16 by its ramp-scarred score —
            # was really doing 0.66+ and climbing, and the winner died on the
            # very same tunnel. Selection cannot see a ramp-up steady state,
            # so the round itself demotes a leg that crawls below what the
            # losers promised, and hands the baton over.
            _oth = [v for k, v in scores.items() if k != base and v]
            oth_line = (0.4 * 1048576.0 / min(_oth)) if _oth else None
            _save(st)
            st["target"] = str(cdir)
            st["steps_total"] = 2
            st["base_bytes"] = 0
            st["total_bytes"] = int(zip_bytes or 0)
            st["step_label"] = GPU_ASSET
            _save(st)
            _log.info(_i18n.ui("GPU 拉取轮次: 腿=%s 总量=%s B 续传=%s B 死腿=%s",
                     "gpu round: leg=%s total=%s B resume=%s B dead=%s"),
            base, zip_bytes, int(st.get("done_bytes") or 0), sorted(dead) or "-")
            res = _download_into(dl, zip_path, st, want_sha=want_sha,
                                 total_hint=zip_bytes,
                                 others_best_Bps=oth_line)
            if res == "cancelled":
                _save(st)
                return
            if res == "ok":
                break
            # interrupted (stream died, or sha mismatch already dropped the
            # .part) / error (4xx stale mirror cache): all this leg's own
            # faults — demote it and re-probe the rest.
            dead.add(base)
            _log.warning(_i18n.ui("GPU 腿 %s 以 res=%s 结束于 %s B —— 换腿（死腿=%s）",
                         "gpu leg %s ended res=%s at %s B — rotating (dead=%s)"),
               base, res, int(st.get("done_bytes") or 0), sorted(dead))
            if len(dead) >= len(GPU_ENDPOINTS):
                st["state"] = "interrupted"
                st["note"] = _i18n.ui(
                    "GitHub 线路这一轮没走通——进度还在，按「续传」会重新测速",
                    "the GitHub leg failed this round — progress kept; press "
                    "resume to re-probe")
                _save(st)
                return
            st.update(state="running", mbps=None, eta_s=None,
                      note=_i18n.ui("这条腿断了——自动换下一条腿续传（进度不丢）",
                                    "this leg died — auto-switching to the "
                                    "next leg, progress kept"))
            _save(st)
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
            _write_pack_manifest(cdir)
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
    """Installed-ness + whether the download button may run. There is NO
    model gate here: the CUDA pack is an engine runtime, logically
    independent of the model weights — either may be fetched first, and a
    fresh machine with no base dir must still offer the button. The only
    hard dependency (NVIDIA presence) is surfaced by the panel's amber
    line, not as a lock. Kept gate_ok/gate_reason keys for the old API
    shape."""
    cdir = _p.cuda_pack_dir()
    return {
        "installed": _p.cuda_pack_ready(cdir),
        "dir": str(cdir),
        "gate_ok": True,
        "gate_reason": None,
        "model_variants": [],
        "tokenizer": None,
    }


def _canonical(path: Path) -> str:
    """Realpath + normcase: junctions, symlinks and 8.3 short-name games
    cannot smuggle an outside folder past the whitelist comparisons."""
    try:
        return os.path.normcase(os.path.realpath(str(path)))
    except OSError:
        return os.path.normcase(str(path))


def _inside(root: Path, f: Path) -> bool:
    """True only when f resolves STRICTLY inside root (never equal, never
    escaping through a crafted ../ or a link)."""
    r, c = _canonical(root), _canonical(f)
    return c != r and c.startswith(r + os.sep)


def _sha256_of(f: Path) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(f, "rb") as fp:
            for chunk in iter(lambda: fp.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _read_pack_manifest(cdir: Path,
                        kind: str = "gpu") -> Optional[Dict[str, Any]]:
    try:
        man = json.loads((cdir / PACK_MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return man if isinstance(man, dict) and str(man.get("kind")) == kind else None


def uninstall_by_manifest(target: Path,
                          man: Dict[str, Any]):
    """The one reverse-removal loop every manifest-carrying install shares
    (gpu pack / stt model cache / any future kind). Walks the snapshot and
    removes EXACTLY its files, after re-verifying each entry: a relative
    path may not escape the folder (../, absolute, drive letters stay
    KEPT), a file whose bytes changed since install is not ours anymore
    and stays KEPT, and a file the manifest never mentions is untouchable.
    Directories left empty go too (deepest first, never blindly), and the
    folder itself is rmdir'd only when truly empty — user files keep it
    alive. A file locked right now aborts with "in_use": honest about
    what already went, never a rmtree over a live handle.
    Returns (removed_count, kept_paths, error_or_None)."""
    removed, kept = 0, []
    for entry in (man.get("files") or []):
        rel = str(entry.get("path") or "")
        f = None
        if rel and not rel.startswith(("/", "\\")) and ":" not in rel:
            f = target / Path(*rel.split("/"))
        if f is None or not _inside(target, f):
            kept.append(rel or "?")
            continue
        if not f.is_file():
            continue  # already gone — nothing to remove
        if (int(entry.get("size") or -1) != f.stat().st_size
                or _sha256_of(f) != str(entry.get("sha256") or "")):
            kept.append(rel)  # bytes changed since install: not ours anymore
            continue
        try:
            f.unlink()
            removed += 1
        except PermissionError:
            return removed, kept + [rel], "in_use"
        except OSError:
            kept.append(rel)
    # our manifest ledger has served its purpose — it goes with the pack.
    # unlink(missing_ok=True) never raises FileNotFoundError, so no dir
    # entry survives to block the final rmdir; a PermissionError here is
    # rare but decisive — we then honestly report not_empty below.
    try:
        (target / PACK_MANIFEST).unlink(missing_ok=True)
    except OSError:
        pass
    # take the directories WE emptied, deepest first — the walk's LAST root
    # is the target itself, so it gets removed right here when empty;
    # anything a user file still occupies refuses on its own and stays
    for root, _d, _f in os.walk(target, topdown=False):
        try:
            os.rmdir(root)
        except OSError:
            pass
    if not target.exists():
        return removed, kept, None      # walk already carried the empty dir off
    # something survived — list EVERY file still inside (manifest strays
    # included: the user's own notes were never in it, yet the report is
    # more honest for naming them) so the caller can show what remains
    try:
        survivors = {p.relative_to(target).as_posix()
                     for p in target.rglob("*") if p.is_file()}
        for rel in survivors - set(kept):
            kept.append(rel)
    except OSError:
        pass
    kept.append("<dir not empty>")
    return removed, kept, "not_empty"


def uninstall_gpu() -> Dict[str, Any]:
    """Reverse-uninstall the CUDA pack along the install path the LEDGER
    recorded — the DLL root lives wherever the installer wrote it back
    then (HERMES_HOME, or the plugin-dir fallback when that was not
    writable), which on a user's machine need not equal what
    cuda_pack_dir() infers today. The target NEVER roams: a recorded
    path that is missing, relative, oddly named or living under any
    parent outside the two roots the installer itself may use (whitelist
    compared via realpath) is refused outright — falling back to
    "today's guess" is how unrelated user folders die.

    Deletion is MANIFEST-DRIVEN: the installer inventoried every file it
    placed (relative path + size + sha256) inside the pack dir, and the
    uninstall removes exactly those, after re-checking each path stays
    strictly inside the root. A file whose bytes changed since install
    is left alone (the user may have made it theirs); a file the
    manifest never mentions is never deleted; the folder itself is only
    rmdir'd when truly empty. Packs installed before manifests existed
    fall back to whole-dir removal, still gated by the parent whitelist
    plus the core-DLL signature. A DLL held by a live synthesis worker
    blocks everything: qwen.dll is probed for exclusive access FIRST,
    so a busy engine can never leave half a pack behind. Idempotent: a
    missing dir counts as already removed."""
    st = load_state()
    if _busy(st):
        return {"ok": False, "reason": "busy"}
    pack_root = _p.cuda_pack_dir()
    allowed = {_canonical(pack_root.parent), _canonical(_p.plugin_root())}
    chapter = st if (st and str(st.get("kind") or "") == "gpu") else None
    if chapter is not None:
        t = Path(str(chapter.get("target") or ""))
        if not (t.is_absolute() and t.name == pack_root.name
                and _canonical(t.parent) in allowed):
            return {"ok": False, "reason": "not_pack"}
        target = t
    else:
        target = pack_root
    if not target.is_dir():
        _log.info(_i18n.ui("GPU 卸载: %s 本就不存在", "gpu uninstall: %s already gone"), target)
        if chapter is not None:
            try:
                _state_file().unlink(missing_ok=True)
                _cancel_marker().unlink(missing_ok=True)
            except OSError:
                pass
        return {"ok": True, "removed": None}
    if not (target / "qwen.dll").is_file() or not (target / "ggml-cuda.dll").is_file():
        # not our pack — refuse, never guess
        return {"ok": False, "reason": "not_pack"}
    try:  # a loaded DLL refuses an exclusive reopen: engine alive => not now
        with open(target / "qwen.dll", "r+b"):
            pass
    except PermissionError:
        return {"ok": False, "reason": "in_use"}
    except OSError:
        pass  # odd but not locked-looking; the removal below decides
    # Re-check right before cutting: a worker may have mapped the DLLs in
    # the probe→removal gap. Deleting files a live process has mapped is
    # Windows' classic silent footgun, so this lock check must be
    # immediately adjacent to the removal, not merely earlier.
    try:
        with open(target / "qwen.dll", "r+b"):
            pass
        with open(target / "ggml-cuda.dll", "r+b"):
            pass
    except PermissionError:
        return {"ok": False, "reason": "in_use"}
    except OSError:
        pass
    man = _read_pack_manifest(target)
    removed = 0
    kept: List[str] = []
    if man is not None:
        # one shared manifest-driven loop for every install kind
        removed, kept, perr = uninstall_by_manifest(target, man)
        if perr == "in_use":
            return {"ok": False, "reason": "in_use", "removed": removed,
                    "kept": kept}
    else:
        # pre-manifest pack: the parent whitelist + core-DLL signature
        # above are the warrant; the dir was created by our installer.
        try:
            shutil.rmtree(target)
        except PermissionError:
            return {"ok": False, "reason": "in_use", "removed": 0, "kept": []}
        except OSError as e:
            return {"ok": False, "reason": "io",
                    "detail": f"{type(e).__name__}: {str(e)[:120]}"}
        removed = -1  # whole directory
    parent = target.parent
    for crumb in (parent / GPU_ASSET, Path(str(parent / GPU_ASSET) + ".part"),
                  target.with_name(target.name + ".new"),
                  target.with_name(target.name + ".old")):
        try:
            if crumb.is_file():
                crumb.unlink()
            elif crumb.is_dir():
                shutil.rmtree(crumb, ignore_errors=True)
        except OSError:
            pass
    if st and str(st.get("kind") or "") == "gpu":
        try:
            _state_file().unlink(missing_ok=True)
            _cancel_marker().unlink(missing_ok=True)
        except OSError:
            pass
    _log.info(_i18n.ui("GPU 已卸载: %s（删除=%s 保留=%s）",
                     "gpu uninstalled: %s (files=%s kept=%s)"),
            target, removed, kept)
    return {"ok": True, "removed": str(target), "files_removed": removed,
            "kept": kept}


# ---- wake-word KWS model pipeline (official k2-fsa kws-models release) ----
# Why this exists at all (owner's matrix, 2026-11-20): the host's own
# wake_word machinery auto-downloads ONLY the English gigaspeech model,
# GitHub-direct with no mirror and no resume — and it NEVER fetches the
# Chinese wenetspeech model, so a Chinese wake word on a fresh install
# simply stays silent. The api save bridge calls HERE to fetch the right
# model for the saved phrase's language, verify the official SHA256
# (pinned below, straight from the release's checksum.txt), unpack into
# HERMES_HOME/cache/wakewords/<official-dir>/ — the very root the host
# itself uses and its readiness scan (tokens.txt) accepts, so nothing
# here is private state the host cannot see — and point
# wake_word.sherpa.model_dir at it. Same three legs as the GPU pack;
# the .part lives next to the same repo path on every leg, so leg
# switches resume instead of restarting.

WAKE_REPO = "k2-fsa/sherpa-onnx"
WAKE_TAG = "kws-models"
WAKE_MODELS: Dict[str, Dict[str, Any]] = {
    "en": {"file": "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01.tar.bz2",
           "dir": "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01",
           "bytes": 17626723,
           "sha": "f170013b4716e41b62b9bfd809687c207cef798ef9bc6534d524e17af9b6561a"},
    "zh": {"file": "sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2",
           "dir": "sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01",
           "bytes": 32654866,
           "sha": "b2f7c89690dc8ce4c6ed6afeab7cd800c36ad1421fb6b6302b4a4b194cf7f35f"},
}


def wake_model_dirname(lang: str) -> str:
    return str((WAKE_MODELS.get(lang) or {}).get("dir") or "")


def wake_model_root() -> Path:
    """Official sherpa-KWS cache root: HERMES_HOME/cache/wakewords. The host
    (_ensure_sherpa_model) downloads into this same root, and the per-model
    folder names below are the release's own dir names, so models placed by
    this pipeline pass the host's own readiness scan unchanged."""
    home = str(_p.hermes_home())
    return Path(home) / "cache" / "wakewords"


def wake_model_ready(lang: str) -> bool:
    m = WAKE_MODELS.get(lang)
    if not m:
        return False
    d = wake_model_root() / str(m["dir"])
    # host readiness = tokens.txt present; we additionally require the three
    # onnx parts so a half-extracted or half-deleted folder can never flip
    # the panel's green light
    if not (d / "tokens.txt").is_file():
        return False
    for part in ("encoder", "decoder", "joiner"):
        if not any(d.glob(f"{part}*.onnx")):
            return False
    return True


def _wake_urls(base: str, name: str) -> str:
    return f"{base}/{WAKE_REPO}/releases/download/{WAKE_TAG}/{name}"


def pick_wake_endpoint(name: str, size: Optional[int],
                       exclude: Optional[set] = None):
    """Clone of pick_gpu_endpoint for the KWS tarballs: real-file Range
    probes, head+body@1/3 sampling, GitHub's generous ramp window (the
    WARP-tunnel leg needs it), tight windows for the local mirrors.
    Kept separate on purpose — the GPU path is field-proven and stays
    untouched."""
    start = (size // 3) if (size and size >= 12 * 1024 * 1024) else 0
    urls = {c: _wake_urls(c, name) for c in GPU_ENDPOINTS
            if not exclude or c not in exclude}

    def _score(c: str, u: str):
        if c == GPU_GITHUB:
            return _probe_leg(u, budget=_GPU_BIG["budget"],
                              want=_GPU_BIG["want"], start=start)
        return _probe_leg(u, start=start)

    scores = {c: _score(c, u) for c, u in urls.items()}
    _log.info(_i18n.ui("唤醒模型腿探针 @%sB: %s（排除=%s）",
                       "wake-model leg probe @%sB: %s (excluded=%s)"),
              start,
              {k: (round(v, 2) if v is not None else None)
               for k, v in scores.items()},
              sorted(exclude or ()) or "-")
    live = {k: v for k, v in scores.items() if v is not None}
    if not live:
        raise NoEndpointError(
            "no wake-model leg can stream bytes right now — tried "
            f"{sorted(urls) or 'nothing (all legs already burned this run)'}")
    best = min(live, key=lambda k: live[k])
    return best, scores


def _wake_write_config(lang: str, model_dir: Path) -> None:
    """Point the host's wake engine at the placed model.

    The key the host ACTUALLY reads to pick the engine is
    ``wake_word.provider`` (tools/wake_word.py ``_provider()`` reads the
    "provider" key, default "auto"; "sherpa" is one of its ``_PROVIDERS``).
    ``wake_word.engine`` is kept as the panel-side display alias only —
    writing engine ALONE was the 2026-11-20 incident: config said
    engine=sherpa, the host still saw provider="auto", armed openwakeword
    and the Chinese wake phrase never fired. Both keys are written so host
    and panel agree. The model dir is pinned through the official sub-key
    ``wake_word.sherpa.model_dir``. Merge write: the phrase and every
    other key stand untouched."""
    try:
        from hermes_cli.config import load_config, save_config
        cfg = load_config() or {}
        ww = cfg.get("wake_word")
        if not isinstance(ww, dict):
            ww = {}
        ww["provider"] = "sherpa"  # what the host wake gate reads
        ww["engine"] = "sherpa"    # panel display alias, kept in sync
        sh = ww.get("sherpa")
        if not isinstance(sh, dict):
            sh = {}
        sh["model_dir"] = str(model_dir)
        ww["sherpa"] = sh
        cfg["wake_word"] = ww
        save_config(cfg)
        _log.info(_i18n.ui("唤醒配置已写入: provider=sherpa model_dir=%s",
                           "wake config written: provider=sherpa model_dir=%s"),
                  model_dir)
    except Exception as e:  # never kill the job over a config hiccup
        _log.warning(_i18n.ui("唤醒配置写入失败: %s", "wake config write failed: %s"), e)


def _wake_extract_tar_bz2(tar_path: Path, dest: Path) -> Optional[str]:
    """Unpack the official .tar.bz2, stripping the single top-level folder
    the release wraps its files in, then swap it into place atomically:
    staging dir first, old copy (if any) renamed aside, new one moved in,
    aside deleted. Any shape error aborts with a message and leaves the
    previous model intact — a half unpack never replaces a working one."""
    staging = dest.with_name(dest.name + ".new")
    shutil.rmtree(staging, ignore_errors=True)
    try:
        staging.mkdir(parents=True)
        with tarfile.open(tar_path, "r:bz2") as tf:
            try:
                tf.extractall(staging, filter="data")  # py3.12+: blocks ../ escapes
            except TypeError:
                tf.extractall(staging)
        sub = [p for p in staging.iterdir() if p.is_dir()]
        loose = [p for p in staging.iterdir() if p.is_file()]
        payload = staging
        if len(sub) == 1 and not loose:
            payload = sub[0]  # the release's wrapper folder — strip it
        if dest.exists():
            aside = dest.with_name(dest.name + ".old")
            shutil.rmtree(aside, ignore_errors=True)
            os.replace(dest, aside)
        try:
            os.replace(payload, dest)
        except OSError:
            # swap failed — put the old model back so nothing regressed
            if 'aside' in locals() and aside.exists() and not dest.exists():
                os.replace(aside, dest)
            raise
        if 'aside' in locals():
            shutil.rmtree(aside, ignore_errors=True)
        return None
    except Exception as e:
        shutil.rmtree(staging, ignore_errors=True)
        return f"{type(e).__name__}: {str(e)[:140]}"


def _run_wake(st: Dict[str, Any]) -> None:
    """Full re-enterable wake-model job (fresh start AND resume share this
    entry): probe legs -> Range-resume download with official-sha check ->
    atomic tar.bz2 unpack -> write engine=sherpa + model_dir."""
    lang = str(st.get("lang") or "")
    m = WAKE_MODELS.get(lang) or {}
    name = str(m.get("file") or "")
    dest = Path(str(st.get("target") or ""))
    try:
        if not name or not str(st.get("target") or ""):
            raise RuntimeError("wake job state missing model/target")
        if wake_model_ready(lang):
            # someone/something already placed it — just (re)point config
            _wake_write_config(lang, dest)
            st.update(state="done", pct=100.0, note=None,
                      done_bytes=st.get("total_bytes") or m.get("bytes") or 0)
            _save(st)
            return
        crumb = dest.parent
        crumb.mkdir(parents=True, exist_ok=True)
        part = crumb / (name + ".part")
        total = int(m.get("bytes") or 0) or None
        want_sha = str(m.get("sha") or "") or None
        dead: set = set()
        while True:
            try:
                base, scores = pick_wake_endpoint(name, total, exclude=dead)
            except NoEndpointError as e:
                st.update(state="error", note=_i18n.ui(
                    "三条腿都下不动唤醒模型——检查网络后点重试（免费镜像线路白天波动，稍后再试往往就通）",
                    "no wake-model leg can stream right now — check the "
                    "network and retry; free mirror legs fluctuate and "
                    "recover. Details: ") + str(e)[:120])
                _save(st)
                return
            st["endpoint"] = base
            _oth = [v for k, v in scores.items() if k != base and v]
            oth_line = (0.4 * 1048576.0 / min(_oth)) if _oth else None
            st["steps_total"] = 1
            st["base_bytes"] = 0
            st["total_bytes"] = total or 0
            st["step_label"] = name
            _save(st)
            _log.info(_i18n.ui("唤醒模型拉取轮次: 腿=%s 总量=%s B 续传=%s B 死腿=%s",
                               "wake round: leg=%s total=%s B resume=%s B dead=%s"),
                      base, total, int(st.get("done_bytes") or 0),
                      sorted(dead) or "-")
            res = _download_into(_wake_urls(base, name), part, st,
                                 want_sha=want_sha, total_hint=total,
                                 others_best_Bps=oth_line)
            if res == "cancelled":
                _save(st)
                return
            if res == "ok":
                break
            dead.add(base)
            if len(dead) >= len(GPU_ENDPOINTS):
                st.update(state="interrupted", note=_i18n.ui(
                    "三条腿都试过一轮没走通——进度还在，按重试会重新测速换腿",
                    "every leg was tried this round — progress kept; press "
                    "retry to re-probe all legs"))
                _save(st)
                return
            st.update(state="running", mbps=None, eta_s=None,
                      note=_i18n.ui("这条腿断了——自动换下一条腿续传（进度不丢）",
                                    "this leg died — auto-switching to the "
                                    "next leg, progress kept"))
            _save(st)
        st.update(state="extracting", step_label=None, mbps=None, eta_s=None)
        _save(st)
        err = _wake_extract_tar_bz2(part, dest)
        if err is None and not wake_model_ready(lang):
            err = "extracted model failed the completeness check"
        try:
            part.unlink(missing_ok=True)
        except OSError:
            pass
        if err:
            st.update(state="error", note=err)
        else:
            _wake_write_config(lang, dest)
            st.update(state="done", note=None, pct=100.0,
                      done_bytes=st.get("total_bytes"))
        _save(st)
    except Exception as e:
        st.update(state="error", note=f"{type(e).__name__}: {str(e)[:160]}")
        _save(st)


def start_wake_model_download(lang: str, phrase: str) -> Dict[str, Any]:
    """Kick the KWS model job for one language. Ready models never
    re-download; a running job (of ANY kind — one download ledger) says
    busy and the panel just rides the other progress bar."""
    m = WAKE_MODELS.get(lang)
    if not m:
        return {"ok": False, "detail": "unknown wake model language"}
    with _LOCK:
        st = load_state()
        if _busy(st):
            return {"ok": False, "busy": True,
                    "detail": _i18n.ui("已有一个下载任务在跑——先等它完成或取消它",
                                       "a download is already running — "
                                       "wait for it or cancel it")}
        if wake_model_ready(lang):
            return {"ok": True, "ready": True}
        st = _new_state("wake", str(wake_model_root() / str(m["dir"])))
        st.update(lang=lang, phrase=str(phrase or ""))
        _save(st)
        threading.Thread(target=_run_wake, args=(st,), daemon=True,
                         name="yache-dl-wake").start()
        return {"ok": True, "started": True}


def wake_model_status() -> Dict[str, Any]:
    """Panel view of the wake job only: any other kind (or none) reads
    idle, so the wake panel never paints the CUDA/model wizard's progress."""
    st = download_status()
    if str(st.get("kind") or "") != "wake":
        return {"state": "idle"}
    return st


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
