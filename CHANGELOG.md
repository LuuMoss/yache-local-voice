# Changelog

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
