"""Yache Voice — single source of truth for every filesystem path.

Resolution order everywhere (no personal absolute path is ever hardcoded):

  1. explicit environment variable           (power users / tests / launcher)
  2. plugin data file ``HERMES_HOME/yache-voice.json``  (settings page / download center write it)
  3. platform default under ``HERMES_HOME``

``worker.py`` runs under the *engine* venv interpreter as a standalone
process; it imports this module directly (``sys.path[0]`` is the engine
directory), so both halves of the chain always agree on the same paths.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

DATA_FILE_NAME = "yache-voice.json"

# Default gguf filenames inside ``<base>/models/`` (Q8_0 sweet spot, see tech
# doc §8). Overridable via env / data file for other quantization tiers.
DEFAULT_TALKER = "qwen-talker-1.7b-base-Q8_0.gguf"
DEFAULT_TOKENIZER = "qwen-tokenizer-12hz-Q8_0.gguf"


def hermes_home() -> Path:
    env = os.environ.get("HERMES_HOME", "").strip()
    home = Path(env).expanduser() if env else None
    if home is None:
        if os.name == "nt":
            local = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
            home = Path(local) / "hermes"
        else:
            home = Path.home() / ".hermes"
    # Self-verification: this plugin always lives at <HERMES_HOME>/plugins/<slug>.
    # If a stale env var or a hijacked LOCALAPPDATA points the resolved home
    # somewhere the plugin is NOT installed under, trust the plugin's own
    # physical location instead (it is the ground truth of where Hermes runs).
    try:
        own = Path(__file__).resolve().parents[3]  # .../hermes/plugins/<slug>/yache_engine/paths.py
        if not (home / "plugins" / "yache-local-voice").is_dir() and (own / "plugins" / "yache-local-voice").is_dir():
            return own
    except OSError:
        pass
    return home


def data_file() -> Path:
    return hermes_home() / DATA_FILE_NAME


def load_data() -> dict:
    """Plugin data file (best-effort; missing/corrupt -> {})."""
    try:
        obj = json.loads(data_file().read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def save_data(data: dict) -> None:
    data_file().parent.mkdir(parents=True, exist_ok=True)
    tmp = data_file().with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(data_file())


# ---------------------------------------------------------------------------
# TTS engine layout: <base>/{models/*.gguf, .venv/}
# ---------------------------------------------------------------------------

def tts_base_dir() -> Path:
    env = os.environ.get("QWEN3_TTS_BASE_DIR", "").strip()
    if env:
        return Path(env).expanduser()
    cfg = str(load_data().get("tts_base_dir", "")).strip()
    if cfg:
        return Path(cfg).expanduser()
    return hermes_home() / "tts" / "qwen3-tts-1.7b"


def plugin_root() -> Path:
    """This plugin's install dir: <HERMES_HOME>/plugins/yache-local-voice."""
    return Path(__file__).resolve().parents[2]


def cuda_pack_ready(d: Path) -> bool:
    """True when d holds a complete CUDA accelerator pack (see the GitHub
    release yache-cuda-windows-x64.zip): the engine dll plus its CUDA companions."""
    d = Path(d)
    return all((d / f).is_file() for f in ("qwen.dll", "ggml-cuda.dll", "cublas64_12.dll"))


def cuda_pack_dir() -> Path:
    """Install root of the GPU accelerator pack. HERMES_HOME first (upgrade-
    safe, shared across every model dir); plugin dir only when it is not
    writable."""
    home = hermes_home()
    if home.is_dir() and os.access(home, os.W_OK):
        return home / "yache-cuda"
    return plugin_root() / "yache-cuda"


def engine_python(base: Path | None = None) -> Path:
    env = os.environ.get("QWEN3_TTS_PYTHON", "").strip()
    if env:
        return Path(env).expanduser()
    base = Path(base or tts_base_dir())
    win = os.name == "nt"
    cand = base / ".venv" / ("Scripts" if win else "bin") / ("python.exe" if win else "python")
    if cand.is_file():
        return cand
    # Release layout (no engine venv): the vendor packages are pure Python +
    # ctypes, so the process that imports this module IS the engine python.
    return Path(sys.executable)


def engine_site_packages(base: Path | None = None) -> Path:
    env = os.environ.get("QWEN3_TTS_SITE_PACKAGES", "").strip()
    if env:
        return Path(env).expanduser()
    base = base or tts_base_dir()
    if os.name == "nt":
        sp = base / ".venv" / "Lib" / "site-packages"
    else:
        # POSIX: resolve the versioned lib dir best-effort.
        sp = base / ".venv" / "lib" / "site-packages"
        lib = base / ".venv" / "lib"
        if lib.is_dir():
            for d in sorted(lib.glob("python*")):
                cand = d / "site-packages"
                if cand.is_dir():
                    sp = cand
                    break
    if sp.is_dir():
        return sp
    # Release layout: vendored faster_qwen3_tts + qwentts_cpp ship inside the
    # plugin and are torch-free on the GGML path.
    return plugin_root() / "vendor"


def dll_search_dir(base: Path | None = None) -> Path:
    """Directory holding qwen.dll (+ CUDA runtime companions)."""
    env = os.environ.get("QWEN3_TTS_DLL_DIR", "").strip()
    if env:
        return Path(env).expanduser()
    sp = engine_site_packages(base)
    if str(sp) != str(plugin_root() / "vendor"):
        # Dev venv wins verbatim (this machine keeps its CUDA lib untouched).
        return sp / "qwentts_cpp" / "lib"
    if cuda_pack_ready(cuda_pack_dir()):
        return cuda_pack_dir()
    return plugin_root() / "vendor" / "qwentts_cpp" / "lib"


def model_paths() -> tuple[Path, Path]:
    """(talker_gguf, tokenizer_gguf) — env / data-file overridable."""
    base = tts_base_dir()
    data = load_data()
    talker = (os.environ.get("QWEN3_TTS_TALKER", "").strip()
              or str(data.get("talker_gguf", "")).strip()
              or DEFAULT_TALKER)
    tokenizer = (os.environ.get("QWEN3_TTS_TOKENIZER", "").strip()
                 or str(data.get("tokenizer_gguf", "")).strip()
                 or DEFAULT_TOKENIZER)
    if not os.path.isabs(talker):
        talker = str(base / "models" / talker)
    if not os.path.isabs(tokenizer):
        tokenizer = str(base / "models" / tokenizer)
    return Path(talker), Path(tokenizer)


# ---------------------------------------------------------------------------
# Voices / worker
# ---------------------------------------------------------------------------

def voices_dir() -> Path:
    env = os.environ.get("HERMES_VOICES_DIR", "").strip()
    if env:
        return Path(env).expanduser()
    return hermes_home() / "voices"


def voices_registry_path() -> Path:
    env = os.environ.get("QWEN3_TTS_REGISTRY", "").strip()
    if env:
        return Path(env).expanduser()
    return voices_dir() / "registry.json"


def worker_script() -> Path:
    """The standalone engine child script, shipped inside this package."""
    env = os.environ.get("QWEN3_TTS_WORKER", "").strip()
    if env:
        return Path(env).expanduser()
    return Path(__file__).resolve().parent / "worker.py"


# ---------------------------------------------------------------------------
# Quantization tiers — hardware sizing guide (tech doc §7.4)
# ---------------------------------------------------------------------------
# Min FREE VRAM per tier, MiB. Ground truth from the 2026-11-11 probe on a
# 24 GB box: with only 1976 MiB free the 1.7B Q8_0 model *loaded* (delta
# 1528 MiB — WDDM page-faults a tight fit in slowly) and synthesis then
# never finished in 150 s. A tier is only usable when free VRAM clears the
# measured footprint plus activation headroom, not merely when it "loads".
#
#   tier        gguf footprint (file)   measured VRAM load   min free
#   1.7b-q8     2.66 GB                 1528 MiB             3800 MiB
#   1.7b-q4     ~1.65 GB                ~1000 MiB (est.)     2500 MiB
#   0.6b-q8     ~1.28 GB                ~800 MiB  (est.)     1800 MiB
#   cpu         —                       0                    —
# (Q4/0.6B rows scale the measured Q8 ratio 0.57× footprint->VRAM; refine
#  with real probes in M2. Workers never hard-crash on a tight fit: the
#  engine falls back to CPU on its own, this table just picks the tier that
#  runs *well*.)
# GPU load floor (MiB free VRAM) — lowered after the 2026-11-11 A/B test:
# with the chat engine saturated and only ~480-500 MiB free, Q8_0 CUDA
# synthesis still succeeded (RTF ~0.25, first chunk ~2s).  The old 3800 was
# an over-cautious value from that morning's WDDM page-fault misdiagnosis
# and needlessly pushed busy machines to the CPU backend.  ~200MiB headroom
# is kept: Q8=700 / Q4=400 / 0.6B=300.
TIER_MIN_FREE_MIB = {
    "1.7b-q8": 700,
    "1.7b-q4": 400,
    "0.6b-q8": 300,
}
TIER_ORDER = ("1.7b-q8", "1.7b-q4", "0.6b-q8")

# (talker_gguf, tokenizer_gguf) filenames per tier under <base>/models/.
# Names follow the upstream HF GGUF repo layout; M2 download center is the
# authority that pins exact filenames.
TIER_FILES = {
    "1.7b-q8": (DEFAULT_TALKER, DEFAULT_TOKENIZER),
    "1.7b-q4": ("qwen-talker-1.7b-base-Q4_K_M.gguf",
                "qwen-tokenizer-12hz-Q4_K_M.gguf"),
    "0.6b-q8": ("qwen-talker-0.6b-base-Q8_0.gguf",
                "qwen-tokenizer-12hz-Q8_0.gguf"),
}


def tier_paths(tier: str) -> tuple[Path, Path]:
    """(talker, tokenizer) gguf paths for one quantization tier."""
    talker, tokenizer = TIER_FILES[tier]
    base = tts_base_dir()
    return base / "models" / talker, base / "models" / tokenizer


def installed_tiers() -> list[str]:
    """Tiers whose gguf pair physically exists in models/ (best-effort)."""
    out = []
    for tier in TIER_ORDER:
        t, k = tier_paths(tier)
        if t.is_file() and k.is_file():
            out.append(tier)
    return out


def recommend_quant(free_mib: int | None, has_cuda: bool) -> dict:
    """Pick the best tier for the machine's *free* VRAM, falling back to
    lower tiers that are actually installed; CPU when nothing fits.

    free_mib None = unknown (no nvidia-smi): with a CUDA device reported we
    assume a healthy idle GPU and suggest the top tier, without cuda -> cpu.
    Returns {"tier", "reason"} for the panel sizing hint (8G/16G/24G cards:
    what matters is free VRAM after the chat model, not the card label).
    """
    if not has_cuda:
        return {"tier": "cpu",
                "reason": "no NVIDIA GPU — runs on CPU (expect ~RTF 1)"}
    installed = installed_tiers()
    if free_mib is None:
        tier = installed[0] if installed else "1.7b-q8"
        return {"tier": tier, "reason": "GPU present, free VRAM unknown"}
    for tier in TIER_ORDER:
        if free_mib >= TIER_MIN_FREE_MIB[tier] and (
                not installed or tier in installed):
            return {"tier": tier,
                    "reason": f"free VRAM {free_mib} MiB fits {tier}"}
    return {"tier": "cpu",
            "reason": f"free VRAM {free_mib} MiB below smallest GPU tier — CPU"}


def selected_quant() -> str | None:
    """User-pinned quantization tier from the data file (None = auto).

    The data file (yache-voice.json) may carry "quant": "1.7b-q8" etc to
    pin a tier; anything else (missing/unknown value) means auto-select,
    which recommend_quant() then decides from live free VRAM.
    """
    q = (load_data() or {}).get("quant")
    return q if q in TIER_MIN_FREE_MIB else None


# ---------------------------------------------------------------------------
# Playback wake-gate (experimental; default OFF = stock behavior)
# ---------------------------------------------------------------------------
def wake_gate_enabled() -> bool:
    """Whether replies pause wake detection while playing (default False).

    Stock design: one wake word opens one full-duplex conversation, the
    host pauses wake detection while that conversation is live, and stop
    phrases end it. Auto read-aloud in text-chat mode has no stock pause
    (self-hearing: the reply contains the wake word), and the gate is our
    extra layer for that gap. Shipped OFF by default so published behavior
    stays pure stock semantics; opt in locally with ``wake_gate: true`` in
    the data file while the approach is proven by real usage.
    """
    v = (load_data() or {}).get("wake_gate")
    return isinstance(v, bool) and v


# ---------------------------------------------------------------------------
# STT worker interpreter (the ASR model lives OUTSIDE the gateway process)
# ---------------------------------------------------------------------------
def stt_env_dir() -> Path:
    """<HERMES_HOME>/stt/.venv — the ASR environment the installer owns.

    One venv holding faster-whisper + PyAV, built by the panel's one-click
    installer (dashboard POST /stt/env/install). It lives under HERMES_HOME
    (never the gateway checkout, never the gateway venv, never a personal
    directory), so official upgrades — new source, regenerated gateway
    environment, refreshed dependency lock — all leave it untouched, and
    every user gets the same documented location. ``<HERMES_HOME>/stt/`` is
    only its container; the wake-word environments the official provider
    owns stay in their own ``wake-envs/`` tree next to it.
    """
    return hermes_home() / "stt" / ".venv"


def stt_env_python() -> Path:
    """Interpreter inside stt_env_dir() (may not exist yet).

    Single source of truth shared by the installer, the state panel and
    the candidate ladder below, so all three agree on one location.
    """
    script = "Scripts" if os.name == "nt" else "bin"
    exe = "python.exe" if os.name == "nt" else "python"
    return stt_env_dir() / script / exe


def stt_python_candidates() -> list[Path]:
    """Interpreter(s) used to run the ASR worker (faster-whisper venv).

    Kept out of the gateway process on purpose: the official dependency
    manager regenerates gateway environments from the stock lock, and
    faster-whisper is not in that lock, so an in-gateway install silently
    dies on every rebuild (bit us on 2026-11-14). An external venv is
    immune.

    Order:
      1. env YACHE_STT_PYTHON        -> exact single path
      2. data file "stt_python"      -> exact single path
      3. <HERMES_HOME>/stt/.venv     -> plugin-owned env (panel installer)
      4. gateway interpreter itself  -> works if faster-whisper is importable
    """
    import sys

    env = os.environ.get("YACHE_STT_PYTHON", "").strip()
    if env:
        return [Path(env).expanduser()]
    configured = str(load_data().get("stt_python", "")).strip()
    if configured:
        return [Path(configured).expanduser()]
    # NOTE: os.name, not platform.system() — this module never imported
    # `platform`, and a missing interpreter here used to raise NameError on
    # a fresh install (no env var, no data file), killing STT at first use.
    return [stt_env_python(), Path(sys.executable)]


def asr_worker_script() -> Path:
    """Path of yache_engine/asr_worker.py inside this plugin."""
    return Path(__file__).resolve().parent / "asr_worker.py"
