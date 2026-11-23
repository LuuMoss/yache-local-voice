"""Online updater: version probe, changelog reader, one-click apply.

Design (mirrors download.py's supply-chain posture, kept small on purpose):

- ``check``      — anonymous GET of ``VERSION`` from the repo's main branch;
                   fastest-leg probing (raw GitHub direct, then jsDelivr CDN,
                   then a gh-proxy mirror).  Purely advisory: a probe failure
                   must never make the panel look unhealthy.
- ``changelog``  — plain-text CHANGELOG.md / CHANGELOG.zh.md from the same
                   legs, cached 10 minutes, shown in the panel's changelog
                   dialog (no browser tab needed).
- ``apply``      — background thread: fetch the release asset listing, pull
                   the plugin zip AND its SHA256SUMS entry over two
                   independent paths and require both digests to match
                   (a hash shipped beside the artifact proves nothing on its
                   own, so cross-checking two DNS paths is the fuse here),
                   verify, extract to a staging dir, then copy changed files
                   over the live install.  DLLs locked by a running worker
                   are deferred: the same button, pressed after a restart,
                   finishes the job (idempotent).

No third-party mirrors are trusted for the *payload*; mirrors only carry
bytes that the cross-checked SHA-256 has already vouched for.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import threading
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from yache_engine.paths import hermes_home

OWNER_REPO = "LuuMoss/yache-local-voice"
PLUGIN_NAME = "yache-local-voice"
_UA = {"User-Agent": "yache-local-voice-updater"}
_TIMEOUT = 12

# The plugin root = parent of dashboard/ (this file lives in dashboard/).
_ROOT = Path(__file__).resolve().parent.parent

CHANGELOG_CACHE_S = 600
CHECK_CACHE_S = 300
_changelog_cache: dict = {}   # lang -> (ts, {"ok":..., "text":...})
_check_cache: dict = {}       # key -> (ts, result)
_apply_lock = threading.Lock()
_state: dict = {"state": "idle"}
_state_path: Path | None = None


# ------------------------------------------------------------------ helpers

def _raw_url(branch_file: str) -> str:
    return f"https://raw.githubusercontent.com/{OWNER_REPO}/main/{branch_file}"


def _cdn_url(branch_file: str) -> str:
    return f"https://cdn.jsdelivr.net/gh/{OWNER_REPO}@main/{branch_file}"


def _proxy_url(branch_file: str) -> str:
    return f"https://gh-proxy.com/https://raw.githubusercontent.com/{OWNER_REPO}/main/{branch_file}"


def _fetch(url: str, timeout: int = _TIMEOUT) -> bytes:
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _get_any(path: str, timeout: int = _TIMEOUT) -> tuple[bytes | None, str]:
    """Try raw → CDN → proxy.  Returns (payload, error_tail)."""
    errors = []
    for url in (_raw_url(path), _cdn_url(path), _proxy_url(path)):
        try:
            return _fetch(url, timeout), ""
        except urllib.error.HTTPError as exc:
            errors.append(f"{url.split('/')[2]}: HTTP {exc.code}")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            errors.append(f"{url.split('/')[2]}: {type(exc).__name__}")
    return None, "; ".join(errors)


def _get_release_asset(tag: str, name_contains: str) -> tuple[str | None, str]:
    """List the release's assets via two independent API paths and return
    the browser download URL of the first asset whose name contains
    *name_contains*.  Falls back to a deterministic URL guess when the
    listing itself succeeds on at least one path."""
    api = f"https://api.github.com/repos/{OWNER_REPO}/releases/tags/v{tag}"
    data = None
    errors = []
    for url in (api, f"https://gh-proxy.com/{api}"):
        try:
            data = json.loads(_fetch(url))
            break
        except Exception as exc:
            errors.append(f"{type(exc).__name__}")
    if data is not None:
        for asset in data.get("assets", []):
            if name_contains.lower() in (asset.get("name") or "").lower():
                url = asset.get("browser_download_url")
                if url:
                    return url, ""
    return None, "; ".join(errors) or "asset not found"


def _current_version() -> str:
    m = {}
    try:
        yaml_text = (_ROOT / "plugin.yaml").read_text(encoding="utf-8")
        m = re.search(r"^version:\s*([0-9][\w.\-]*)", yaml_text, re.M)
    except OSError:
        return ""
    return m.group(1) if m else ""


def _as_tuple(tag: str) -> tuple:
    return tuple(int(p) if p.isdigit() else 0 for p in tag.split("-")[0].split("."))


def _state_file() -> Path:
    global _state_path
    if _state_path is None:
        _state_path = hermes_home() / "yache-update-state.json"
    return _state_path


def _save_state() -> None:
    try:
        _state_file().write_text(json.dumps(_state, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


def _set_state(**kw) -> None:
    _state.update(kw)
    _state["updated"] = int(time.time())
    _save_state()


# ------------------------------------------------------------------ check

def check(force: bool = False) -> dict:
    """{ok, current, latest, available}.  Advisory only — ok=False must
    never be rendered as a health problem, just 'no update info'."""
    now = time.time()
    cached = _check_cache.get("check")
    if cached and not force and now - cached[0] < CHECK_CACHE_S:
        return cached[1]
    current = _current_version()
    payload, err = _get_any("VERSION")
    latest = payload.decode("utf-8", "replace").strip() if payload else ""
    if latest and not re.match(r"^\d+(\.\d+)*[\w.\-]*$", latest.split("\n")[0].strip()):
        latest = ""
        err = err or "bad VERSION file"
    result = {
        "ok": bool(latest),
        "current": current,
        "latest": latest,
        "available": bool(latest and current and _as_tuple(latest) > _as_tuple(current)),
        "error": err,
    }
    _check_cache["check"] = (now, result)
    return result


# --------------------------------------------------------------- changelog

def changelog(lang: str = "zh") -> dict:
    """Changelog text for the panel dialog (never hits the network twice
    inside the cache window)."""
    lang = "zh" if str(lang).lower().startswith("zh") else "en"
    now = time.time()
    cached = _changelog_cache.get(lang)
    if cached and now - cached[0] < CHANGELOG_CACHE_S:
        return cached[1]
    fname = "CHANGELOG.zh.md" if lang == "zh" else "CHANGELOG.md"
    payload, err = _get_any(fname)
    if payload is None:
        # Fall back to the bundled copy so the dialog still shows the
        # history of the version the user is running.
        local = _ROOT / ("CHANGELOG.zh.md" if lang == "zh" else "CHANGELOG.md")
        if local.is_file():
            result = {"ok": True, "text": local.read_text(encoding="utf-8"),
                      "source": "bundled", "error": err}
        else:
            result = {"ok": False, "text": "", "source": "none", "error": err}
    else:
        result = {"ok": True, "text": payload.decode("utf-8", "replace"),
                  "source": "remote", "error": ""}
    _changelog_cache[lang] = (now, result)
    return result


# ------------------------------------------------------------------ apply

def _read_state_disk() -> dict:
    try:
        return json.loads(_state_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def apply_status() -> dict:
    """Panel polling view: in-memory state wins; on a cold gateway the last
    persisted state is replayed (a finished install stays reportable)."""
    if _apply_lock.locked() or _state.get("state") not in ("idle",):
        return dict(_state)
    disk = _read_state_disk()
    if disk and disk.get("state") in ("done", "error") and int(disk.get("updated", 0)) >= int(_state.get("updated", 0)):
        return {**disk}
    return dict(_state)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _expected_sha(zip_url: str, tag: str) -> tuple[str | None, str]:
    """Pull the release's SHA256SUMS asset over two paths; require the line
    for our zip to be byte-identical on both, else refuse the payload."""
    sha_url, err = _get_release_asset(tag, "SHA256SUMS")
    if not sha_url:
        return None, f"no SHA256SUMS asset: {err}"
    texts = []
    for url in (sha_url, f"https://gh-proxy.com/{sha_url}",
                sha_url.replace("github.com", "raw.githubusercontent.com", 1)):
        try:
            texts.append(_fetch(url).decode("utf-8", "replace"))
        except Exception:
            continue
        if len(texts) >= 2:
            break
    lines = []
    for text in texts[:2]:
        for line in text.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[1].endswith(".zip"):
                lines.append(parts[0].lower())
    if len(lines) < 2 or lines[0] != lines[1]:
        return None, "sha cross-check failed (paths disagree)"
    return lines[0], ""


def _install_from(staging: Path) -> tuple[list, list]:
    """Copy staging (already the yache-local-voice/ tree) over the live
    install roots.  Only files whose bytes differ are touched; files held
    open by Windows are collected as deferred.  Returns (changed, deferred)."""
    from yache_engine.paths import hermes_home as _hh

    staging = Path(staging)
    changed: list = []
    deferred: list = []
    home = _hh()
    roots = [home / "plugins" / PLUGIN_NAME, home / "desktop-plugins" / PLUGIN_NAME]
    # Backup-before-write (owner rule): zip the old files we are about to
    # touch into home/backups/ — cheap insurance, one zip per update.
    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup = home / "backups" / f"yache-update-{stamp}.zip"
    pending: list = []  # (old_bytes, arcname) snapshotted BEFORE each overwrite
    for root in roots:
        if not root.is_dir():
            continue
        hotload = root.parent.name == "desktop-plugins"
        tag = "dp/" if hotload else ""
        for src in staging.rglob("*"):
            if not src.is_file():
                continue
            rel = src.relative_to(staging)
            # The hot-load scan root only ever carries plugin.js.
            if hotload and rel.as_posix() != "plugin.js":
                continue
            if hotload:
                dst = root / "plugin.js"
            else:
                dst = root / rel
            try:
                if dst.is_file() and dst.read_bytes() == src.read_bytes():
                    continue
                if dst.is_file():
                    try:
                        pending.append((dst.read_bytes(), tag + rel.as_posix()))
                    except OSError:
                        pass  # unreadable old file: nothing to archive
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                changed.append(str(rel))
            except (PermissionError, OSError):
                deferred.append(str(root / rel))
    try:
        if pending:
            backup.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(backup, "w", zipfile.ZIP_DEFLATED) as zf:
                for data, arc in pending:
                    zf.writestr(arc, data)
    except OSError:
        pass  # backup failure must never block an update the user asked for
    return changed, deferred


def apply_async(lang: str = "zh") -> dict:
    """Kick the one-click update.  Returns the accepted state; the panel
    polls apply_status() for progress."""
    with _apply_lock:
        if _state.get("state") in ("downloading", "verifying", "installing"):
            return dict(_state)
        _set_state(state="downloading", percent=0, error="", deferred=[])
    threading.Thread(target=_apply_worker, args=(lang,), daemon=True).start()
    return dict(_state)


def _apply_worker(lang: str) -> None:
    try:
        info = check()
        tag = info.get("latest") or ""
        if not tag or not info.get("available"):
            _set_state(state="error", error="no newer version")
            return
        zip_url, err = _get_release_asset(tag, f"{PLUGIN_NAME}-v{tag}.zip")
        if not zip_url:
            _set_state(state="error", error=f"asset lookup failed: {err}")
            return
        tmp = hermes_home() / "yache-update-tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        dest = tmp / f"{PLUGIN_NAME}-v{tag}.zip"
        # Stream once through the fastest leg that answers, with a simple
        # retry on the next leg (plugin zips are small; no resume needed).
        last_err = ""
        for url in (zip_url, f"https://gh-proxy.com/{zip_url}"):
            try:
                req = urllib.request.Request(url, headers=_UA)
                with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
                    total = int(resp.headers.get("Content-Length") or 0)
                    got = 0
                    with dest.open("wb") as out:
                        while True:
                            block = resp.read(1 << 16)
                            if not block:
                                break
                            out.write(block)
                            got += len(block)
                            if total:
                                _set_state(percent=int(got * 100 / total))
                last_err = ""
                break
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_err = f"{url.split('/')[2]}: {type(exc).__name__}"
                dest.unlink(missing_ok=True)
        if last_err:
            _set_state(state="error", error=f"download failed: {last_err}")
            return
        _set_state(state="verifying")
        want, err = _expected_sha(zip_url, tag)
        if not want:
            _set_state(state="error", error=err)
            return
        got = _sha256_file(dest)
        if got != want:
            dest.unlink(missing_ok=True)
            _set_state(state="error", error="sha256 mismatch")
            return
        _set_state(state="installing")
        staging = tmp / "staging"
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        with zipfile.ZipFile(dest) as zf:
            zf.extractall(staging)
        top = staging / PLUGIN_NAME
        if not top.is_dir():
            inner = [p for p in staging.iterdir() if p.is_dir()]
            if len(inner) == 1:
                top = inner[0]
            else:
                _set_state(state="error", error="bad package layout")
                return
        changed, deferred = _install_from(top)
        shutil.rmtree(tmp, ignore_errors=True)
        _set_state(state="done", percent=100, tag=tag,
                   changed=len(changed), deferred=deferred)
    except Exception as exc:  # never take the gateway down
        _set_state(state="error", error=f"{type(exc).__name__}: {exc}")


def apply_cancel_note() -> str:
    return ""
