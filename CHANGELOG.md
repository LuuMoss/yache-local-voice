# Changelog

## 0.3.0
- Speech recognition moved onto its own legs: the engine now lives in
  `<HERMES_HOME>/stt/.venv`, built on demand by the panel with a live
  progress bar, so official Hermes upgrades can no longer take
  faster-whisper down with them.
- STT model weights download wizard: device-matched default tier
  (large-v3 on NVIDIA CUDA, medium on CPU), live size estimate,
  mirror-first source choice, resumable downloads.
- GPU accelerator pack: one-click CUDA 13 runtime download (~720 MB)
  for NVIDIA-only machines, gated server-side and delivered as a
  GitHub Release asset (`cuda-pack-v1`).
- Read-aloud wake word now runs on the official built-in sherpa-onnx
  KWS engine only.

## 0.2.0
- Wake engine registration retired: wake detection is the official
  built-in sherpa-onnx KWS engine only (no core registry writes).
  Stop-phrase and wake-session editors in the panel persist to the
  official config keys.
- The superseded 0.1.0 wake engine moved outside the release package.

## 0.1.0
- First bundle: voice cloning TTS + speech recognition + plugin-side
  wake engine.
