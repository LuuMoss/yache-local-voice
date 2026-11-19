# Notice

Yache Voice — Hermes Desktop voice plugin
Copyright 2026 LuuMoss (Xiaoya)

This product bundles third-party code, credited below.

## vendor/faster_qwen3_tts/

Vendored from [faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts)
by Andres Marafioti, MIT License. Real-time Qwen3-TTS inference using
CUDA graphs. Locally adapted: streaming CUDA-stream abort and model-dir
plumbing adjusted for this plugin's zero-disk playback pipeline.

## vendor/qwentts_cpp/ + lib/*.dll

Python bindings and prebuilt DLLs derived from `qwentts.cpp`
(MIT License, "The omnivoice.cpp authors"), maintained as a local fork
with a patched CUDA stream abort. The bundled `qwen.dll` and
`ggml-*.dll` are built from that fork (CUDA-flavored build; the CUDA
runtime itself is installed separately as the GPU accelerator pack from
this repository's `cuda-pack-v1` release).

## Qwen3-TTS model weights

The Qwen3-TTS GGUF model files are made by the Qwen team (Alibaba) under
the Apache-2.0 license and are downloaded by the user through the panel's
download wizard. They are **not** part of this repository.

## wake word

Wake-word detection uses Hermes' own built-in sherpa-onnx KWS engine;
nothing about it is bundled here.
