# Changelog

## 0.1.3 — online updater (2026-11-23)

**Update yourself from the panel — no browser, no reinstall.**

- The panel footer now shows the installed version and a **Check update**
  button. When a newer release exists it turns into an amber **Update now**
  button beside a "New version" hint — one click downloads the release,
  verifies its SHA-256 and installs every file. Fully quit and reopen Hermes
  to load the new code; updates are never forced.
- **Changelog** opens the full release history inside the panel, fetched from
  the official repo; offline it falls back to the bundled copy.
- Before replacing anything, every old file about to be overwritten is
  archived into `HERMES_HOME/backups/yache-update-<timestamp>.zip`.
- Version probes travel raw.githubusercontent.com → jsDelivr → gh-proxy with
  automatic fallback; a failed probe stays silent and never disturbs the
  doctor lamps.

## 0.1.2 — desktop-half packaging fix (2026-11-23)

- The plugin zip now ships `desktop/plugin.js`, the desktop-half entry that
   Hermes Desktop's plugin loader reconciles into its hot-load scan root after
   an install. 0.1.1 shipped without it, so a fresh install showed no status-bar
   icon (a manual hot-load copy on the dev machine masked this).
- `desktop/plugin.js` is now a byte-identical twin of the root `plugin.js`
   (0.1.0 had shipped two divergent builds of this file).

## 0.1.1 — CI-built engine, supply chain closed (2026-11-23)

- Both engine binary sets are now compiled from source **in this
  repository's GitHub Actions**, with SLSA build provenance:
  - CPU: the four Qwen3-TTS/ggml DLLs (`third_party/qwentts.cpp` pinned at
    6fae9291) are rebuilt by `.github/workflows/build-engine.yml` on every
    `engine-v*` tag and attested via `actions/attest-build-provenance`.
  - CUDA: the full 45-DLL accelerator pack (CUDA 12.4.1, cubins for
    sm_86/89/90 + PTX) is rebuilt by the `build-cuda-pack` job with the
    same attestation, and ships as the versioned Release asset
    `yache-cuda-windows-x64.zip` of this release.
- The CUDA pack download URL moved from the moving `cuda-pack-v1` tag to
  this release's versioned asset (`releases/download/v0.1.1/…`), and the
  pinned SHA-256 now matches the CI-built pack byte for byte; the old
  `cuda-pack-v1` release stays available for installs made before 0.1.1.
- Worker/child processes no longer inherit provider secrets: the engine
  launcher strips every `*KEY*/*TOKEN*/*SECRET*/*PASSWORD*`-style variable
  from the environment it hands to compute children.
- The plugin-catalog entry now registers `attested_binaries`, and the
  unneeded `catalog_self_updater_allowlist.json` row was dropped.
- Documentation: README gained a Build provenance section documenting the
  CI chain end to end; `docs/engine-provenance.md` now records the v0.1.1
  pack pin (run 45101468558, multi-arch kernels) and the versioned-asset
  policy.

## 0.1.0 — first public release (2026-11-19; rebuilt 2026-11-22 after catalog review #158052)

### Review rebuild (2026-11-22)
- Engine DLLs are now built from source in this repository: the exact
  upstream `qwentts.cpp` sources (pinned at 6fae92914045cd83364d2845ceaa
  0f7969727319, including its ggml submodule) are vendored under
  `third_party/qwentts.cpp/`, and `.github/workflows/build-engine.yml`
  rebuilds the four DLLs on every `engine-v*` tag and publishes SLSA build
  provenance (`attested_binaries` in the plugin-catalog entry). The four
  DLLs under `vendor/qwentts_cpp/lib/` are fresh builds from that tree.
- Dropped the vendored `libsndfile` payload (`vendor/soundfile.py`,
  `vendor/_soundfile.py`, `vendor/_soundfile_data/`,
  `vendor/soundfile-0.14.0.dist-info`): the host already ships
  `soundfile`, so the plugin now uses the host's copy instead of bundling
  an unverified DLL.
- The CUDA accelerator pack is now a **versioned release asset** with its
  SHA-256 and exact byte size pinned in source (no runtime trust in a
  same-origin metadata file); third-party GitHub mirrors
  (gh-proxy.com / ghfast.top) were removed in favor of github.com only,
  and the pack itself is now CI-built from the same in-tree sources with
  SLSA attestation (`build-cuda-pack` job).
- The download-throughput probe no longer disables TLS certificate
  verification — every HTTPS path in the plugin verifies certificates.
- `HERMES_HOME` resolution prefers the host's canonical
  `hermes_constants.get_hermes_home()` (profile-aware), with the
  env-var / platform-default chain kept only as a fallback for the
  standalone engine venv.

### Original release notes

- Voice cloning TTS on Qwen3-TTS 1.7B: fully local, zero cloud, zero API
  keys; any-reference voice cloning with your own recorded timbres.
- Time and number read-aloud follows each sentence's own language: "Call me at
  9:05" is spoken as natural English ("nine-oh-five"), while "明天 8:03 开会" is
  spoken as natural Chinese (八点零三分). Dates, durations and units normalize the
  same way, and English sentences stay fully English on time words.
- Speech recognition moved onto its own legs: the engine lives in
  `<HERMES_HOME>/stt/.venv`, built on demand by the panel with a live
  progress bar, so official Hermes upgrades can no longer take
  faster-whisper down with them.
- STT model weights download wizard: device-matched default tier
  (large-v3 on NVIDIA CUDA, medium on CPU), live size estimate,
  mirror-first source choice, resumable downloads.
- GPU accelerator pack: one-click CUDA 13 runtime download (~720 MB)
  for NVIDIA-only machines, gated server-side and delivered as a
  GitHub Release asset, pinned by SHA-256 in source.
- Read-aloud wake word on the official built-in sherpa-onnx KWS
  engine only; stop-phrase and wake-session editors persist to the
  official config keys.
- Read-aloud self-echo guard: fail-soft read of host state keeps the
  detector from ignoring its own TTS audio mid-answer.
