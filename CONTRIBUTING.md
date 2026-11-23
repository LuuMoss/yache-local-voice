# Contributing to yache-local-voice

Thanks for your interest! This is a desktop plugin for
[Hermes Agent](https://hermes-agent.nousresearch.com) providing local
text-to-speech (voice read-aloud, voice cloning) plus speech-to-text, with an
optional CUDA acceleration pack.

## Where things live

| Path | What it is |
| ---- | ---------- |
| `plugin.yaml` | Plugin manifest (version, capabilities, settings schema) |
| `plugin.js` | Frontend half (status-bar entry, panels) — see the sync rule below |
| `desktop/plugin.js` | Same file, copied into `desktop/` for the desktop hot-load root |
| `yache_engine/` | Backend: synthesis worker, download/update logic |
| `dashboard/` | Backend services (settings, download fuse, wake word, STT) |
| `third_party/` | Vendored engine sources (see `LICENSE` / `NOTICE.md`) |
| `.github/workflows/build-engine.yml` | CI that builds the Windows CUDA pack |

> **Important:** `plugin.js` and `desktop/plugin.js` must stay byte-identical.
> Release zips that omit the `desktop/` half break the status-bar icon on
> fresh installs (this actually happened in v0.1.1). When editing, update both
> copies — the release build verifies they match.

## Dev environment

1. Install Hermes Desktop (Windows today).
2. Copy this repository into `<HERMES_HOME>/plugins/yache-local-voice/`
   (Windows: `%LOCALAPPDATA%\hermes\plugins\yache-local-voice\`).
3. Restart the Hermes Desktop app fully (packaged builds do not hot-reload
   plugin frontends).
4. Settings -> Plugin management -> **search the full name**
   `yache-local-voice` and toggle it on.

Backend changes (anything under `yache_engine/` or `dashboard/`) need a
gateway restart; frontend changes need a full app restart.

## Making changes

* Fork, create a branch, keep each PR focused.
* Bump `version:` in `plugin.yaml` and add a `CHANGELOG.md` entry.
* Audio changes must be verified by listening to a generated sample in both
   Chinese and English (English number/time normalization is easy to break).
* Do not weaken the SHA-256 fuse in the CUDA/model download logic, and keep
   the GitHub Actions-built CUDA pack pinned in `dashboard/download.py`.
* Logs land in `<HERMES_HOME>/plugin-data/yache-local-voice/`.

Questions about usage belong in
[Discussions -> Q&A](https://github.com/LuuMoss/yache-local-voice/discussions/categories/q-a).
