# Engine binary provenance

The plugin ships four native DLLs at `vendor/qwentts_cpp/lib/`:

| File        | Built from                                   |
| ----------- | -------------------------------------------- |
| `qwen.dll`  | `third_party/qwentts.cpp/` @ `6fae92914045cd83364d2845ceaa0f7969727319` |
| `ggml.dll`  | (same tree, including its `ggml/` submodule at `40e16e4a814f7fe851a0c486fb9e8c722e957830`) |
| `ggml-base.dll` |                                            |
| `ggml-cpu.dll`  |                                            |

## How they are built

`.github/workflows/build-engine.yml` runs on `windows-latest`:

1. Installs pinned CMake 4.4.5 + Ninja from PyPI (the MSVC-bundled
   `cmake 3.31.6-msvc6` crashes with `STATUS_STACK_BUFFER_OVERRUN` during
   ggml's ASM compiler ABI detection).
2. Configures `third_party/qwentts.cpp/` with `QWEN_SHARED=ON`,
   `GGML_NATIVE=OFF`, and injects the pinned revision/date through the
   `QT_REVISION` / `QT_DATE` cache variables (the single fork patch in
   `third_party/qwentts.cpp/tools/version.cmake` teaches the build system
   to take them from the cache, and falls back to the pinned constants for
   tarball builds — the `qt_version()` handshake stays byte-identical to
   an upstream git build).
3. Builds the `qwen` target and runs a sanity check: it loads the fresh
   `qwen.dll` and asserts `qt_version()` reports the pinned revision.
4. Uploads the fresh build as a workflow artifact (transparency) and emits
   **SLSA build provenance** for the four vendored DLLs via
   `actions/attest-build-provenance@v2`.

The plugin-catalog entry lists these files under `attested_binaries` with
`attestations: github`, so Hermes Desktop and the catalog tooling can
verify each DLL's origin before installing.

## Verify it yourself

```bash
gh attestation verify vendor/qwentts_cpp/lib/qwen.dll \
    --repo LuuMoss/yache-local-voice \
    --source-digest <release-commit-sha> \
    --predicate-type https://slsa.dev/provenance/v1
```

## Rebuilding by hand (Windows)

```bat
cmake -S third_party\qwentts.cpp -B build -A x64 ^
      -DQWEN_SHARED=ON -DGGML_NATIVE=OFF ^
      -DQT_REVISION=6fae92914045cd83364d2845ceaa0f7969727319 -DQT_DATE=2026-09-28
cmake --build build --config Release --target qwen
```

Use CMake >= 4.0 (e.g. `pip install cmake`). Outputs land in `build\Release\`.

## What is *not* attested here

- `ggml-cuda.dll` and the CUDA runtime DLLs — GPU users download the
  accelerator pack (`yache-cuda-windows-x64.zip`) from this repository's
  Releases; its checksum is pinned in `dashboard/download.py`.
- `vendor/_soundfile_data/libsndfile_x64.dll` was removed in 0.1.1: the
  host environment already provides Python `soundfile`, so the plugin uses
  the host's copy instead of bundling its own.
