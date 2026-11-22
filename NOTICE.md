# Notice

Yache Voice — Hermes Desktop voice plugin
Copyright 2026 LuuMoss (Xiaoya)

This product bundles third-party code, credited below.

## vendor/faster_qwen3_tts/

Vendored from [faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts)
by Andres Marafioti, MIT License. Real-time Qwen3-TTS inference using
CUDA graphs. Locally adapted: streaming CUDA-stream abort and model-dir
plumbing adjusted for this plugin's zero-disk playback pipeline.

## vendor/qwentts_cpp/lib/*.dll + third_party/qwentts.cpp/

The four bundled engine DLLs (`qwen.dll`, `ggml.dll`, `ggml-base.dll`,
`ggml-cpu.dll`) are compiled from the `qwentts.cpp` engine
(MIT License, "The omnivoice.cpp authors", upstream
[ServeurpersoCom/qwentts.cpp](https://github.com/ServeurpersoCom/qwentts.cpp)).
The exact upstream sources — pinned at commit
`6fae92914045cd83364d2845ceaa0f7969727319` (2026-09-28), including its
`ggml/` submodule at `40e16e4a814f7fe851a0c486fb9e8c722e957830` — are
vendored under `third_party/qwentts.cpp/`, with one documented fork patch
(`tools/version.cmake` learns the revision from cache variables so the
engine builds from a tarball without a `.git` directory, keeping the
`qt_version()` handshake byte-identical to an upstream build).

A GitHub Actions workflow (`.github/workflows/build-engine.yml`) rebuilds
the DLLs from that source tree on every `engine-v*` tag and publishes SLSA
build provenance for the vendored files; the Hermes plugin catalog gates
them via `attested_binaries` + `attestations: github`, verifiable with:

    gh attestation verify <file> --repo LuuMoss/yache-local-voice \
        --source-digest <pinned-commit-sha> \
        --predicate-type https://slsa.dev/provenance/v1

The CUDA runtime is not among these DLLs; GPU users install it separately
as the accelerator pack (`yache-cuda-windows-x64.zip`) from this
repository's Releases.

## Qwen3-TTS model weights

The Qwen3-TTS GGUF model files are made by the Qwen team (Alibaba) under
the Apache-2.0 license and are downloaded by the user through the panel's
download wizard. They are **not** part of this repository.

## wake word

Wake-word detection uses Hermes' own built-in sherpa-onnx KWS engine;
nothing about it is bundled here.
