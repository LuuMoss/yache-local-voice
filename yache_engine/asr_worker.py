"""local-stt ASR worker — runs under an *external* venv interpreter.

Same lesson as the TTS worker and video-xray's R2T2 worker: the model
lives in a dedicated venv, the gateway only spawns short-lived children.
That makes STT immune to the gateway environment being rebuilt by the
official dependency manager (faster-whisper is not in the stock lock,
so every regeneration silently dropped it).

Contract (stdout last line = single JSON object; stderr = free-form logs):

    python -I asr_worker.py --probe
        -> {"ok": true, "version": "..."}        exit 0/1
    python -I asr_worker.py --wav PATH [--model small] [--language zh]
            [--prompt TEXT] [--device auto] [--compute int8]
        -> {"transcript": "...", "language": "zh", "duration": 12.3}

Only stdlib + faster_whisper are required in the worker venv.
"""

from __future__ import annotations

import argparse
import json
import sys
import time


def _patch_av_open() -> None:
    """faster-whisper 1.2.x calls av.open(metadata_errors=...) which older
    PyAV builds reject; strip the kwarg whatever the venv happens to carry."""
    try:
        import av

        if getattr(av.open, "_yache_patched", False):
            return
        orig = av.open

        def patched(*args, **kwargs):
            kwargs.pop("metadata_errors", None)
            return orig(*args, **kwargs)

        patched._yache_patched = True
        av.open = patched
    except Exception:
        pass


def _load(model_name: str, device: str, compute_type: str):
    from faster_whisper import WhisperModel

    _patch_av_open()
    try:
        return WhisperModel(model_name, device=device, compute_type=compute_type), device
    except Exception:
        if device != "cpu":
            print(f"[asr-worker] device={device} failed, falling back to cpu",
                  file=sys.stderr)
            return WhisperModel(model_name, device="cpu",
                                compute_type=compute_type), "cpu"
        raise


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--wav", default="")
    ap.add_argument("--model", default="small")
    ap.add_argument("--language", default="")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--compute", default="int8")
    a = ap.parse_args()

    try:
        if a.probe:
            import faster_whisper

            print(json.dumps({"ok": True,
                              "version": getattr(faster_whisper, "__version__", "?")}))
            return 0

        if a.verify:
            # Real-load proof for the settings panel: genuinely load the
            # chosen model (an official-source download happens here on
            # first use) and run a short transcribe over 1 s of silence.
            # Whisper has no VAD short-circuit, so a fake or unreadable
            # model directory fails right here, with the engine's own
            # error coming back verbatim for the panel to show.
            import faster_whisper
            import os
            import tempfile
            import wave

            t0 = time.time()
            model, used = _load(a.model, a.device, a.compute)
            fd, quiet = tempfile.mkstemp(suffix=".wav")
            try:
                os.close(fd)
                with wave.open(quiet, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16000)
                    wf.writeframes(b"\x00" * 32000)
                model.transcribe(quiet, beam_size=1)
            finally:
                try:
                    os.unlink(quiet)
                except OSError:
                    pass
            print(json.dumps({"ok": True,
                              "version": getattr(faster_whisper,
                                                 "__version__", "?"),
                              "model": a.model, "device": used,
                              "load_s": round(time.time() - t0, 1)}))
            return 0

        if not a.wav:
            raise ValueError("--wav is required unless --probe")

        t0 = time.time()
        model, used_device = _load(a.model, a.device, a.compute)
        print(f"[asr-worker] model {a.model} loaded on {used_device} "
              f"in {time.time() - t0:.1f}s", file=sys.stderr)

        kwargs = {}
        if a.language:
            kwargs["language"] = a.language
        if a.prompt:
            kwargs["initial_prompt"] = a.prompt

        segments, info = model.transcribe(a.wav, **kwargs)
        text = " ".join(s.text.strip() for s in segments).strip()
        print(json.dumps({
            "transcript": text,
            "language": getattr(info, "language", a.language or ""),
            "language_probability": round(float(getattr(info, "language_probability", 0.0) or 0.0), 3),
            "duration": round(float(getattr(info, "duration", 0.0) or 0.0), 2),
        }, ensure_ascii=True))  # ASCII-only: immune to any console codepage
        return 0
    except BaseException as exc:  # noqa: BLE001 — one clean failure line
        import traceback

        traceback.print_exc(file=sys.stderr)
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    sys.exit(main())
