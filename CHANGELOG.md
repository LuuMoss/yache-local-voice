# Changelog

## 0.1.0 — first public release (2026-11-19)
- Voice cloning TTS on Qwen3-TTS 1.7B: fully local, zero cloud, zero API
  keys; any-reference voice cloning with your own recorded timbres.
- Speech recognition moved onto its own legs: the engine lives in
  `<HERMES_HOME>/stt/.venv`, built on demand by the panel with a live
  progress bar, so official Hermes upgrades can no longer take
  faster-whisper down with them.
- STT model weights download wizard: device-matched default tier
  (large-v3 on NVIDIA CUDA, medium on CPU), live size estimate,
  mirror-first source choice, resumable downloads.
- GPU accelerator pack: one-click CUDA 13 runtime download (~720 MB)
  for NVIDIA-only machines, gated server-side and delivered as a
  GitHub Release asset (`cuda-pack-v1`).
- Read-aloud wake word on the official built-in sherpa-onnx KWS
  engine only; stop-phrase and wake-session editors persist to the
  official config keys.
- Read-aloud self-echo guard: fail-soft read of host state keeps the
  detector from ignoring its own TTS audio mid-answer.
