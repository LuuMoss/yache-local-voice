# Security Policy

## Supported versions

| Version | Supported |
| ------- | --------- |
| Latest release on GitHub Releases | yes |
| Older releases | best effort |

## What this plugin touches

* It runs **locally**: speech synthesis happens on your own machine (local
  Qwen3-TTS GGML engine) or against an API endpoint **you** configure.
* It spawns a local worker process for the synthesis engine and passes a
  minimal, sanitized environment to it (no token passthrough).
* Model and CUDA-runtime downloads go to GitHub Releases and Hugging Face
  mirrors over HTTPS, and every downloaded package is verified against a
  pinned SHA-256 before it is unpacked.
* The optional Windows CUDA pack is built by this repository's own GitHub
  Actions workflow (`build-engine.yml`) and published as a Release asset.

## Reporting a vulnerability

Please report security issues **privately**, not in a public issue:

* Use [private vulnerability reporting](https://github.com/LuuMoss/yache-local-voice/security/advisories/new)
  on GitHub (recommended), or
* open a Discussion titled "security (private contact)" and a maintainer will
  move the conversation to a private channel.

Please do not post secrets (API keys, tokens, model paths containing personal
data) in public issues — the bug-report form asks for log excerpts, and logs
may contain local file paths.

We aim to acknowledge private reports within a week and will publish a GitHub
Security Advisory for confirmed issues affecting released versions.
