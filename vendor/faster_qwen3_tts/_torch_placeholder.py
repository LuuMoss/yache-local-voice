"""Import-time placeholder for torch on torch-free (GGML-only) installs.

model.py is imported on every engine path, but only the CUDA-graph backend
ever uses torch — the GGML path (this plugin's engine, backend='ggml')
returns the ctypes GGMLQwen3TTS before touching a single tensor.  The
module body still *evaluates* torch attributes at import time: six
@torch.inference_mode() decorators and two ``dtype: torch.dtype =
torch.bfloat16`` signatures.  So the import cannot simply be skipped —
this placeholder satisfies exactly those evaluations and turns any real
tensor work into an actionable error.

Why it exists: a fresh Hermes desktop gateway runs on a standalone python
without torch, and forcing a ~200 MB torch download for an import the GGML
path never exercises is not an acceptable first-run experience (2026-11-19
fresh-machine report: the whole package died on "No module named torch").
"""
import types
from typing import Any


def _decorator(*args: Any, **kwargs: Any) -> Any:
    # @torch.inference_mode() / @torch.no_grad() decorate CUDA-graph-only
    # methods; those run exclusively when the real torch is installed, so
    # pass the function through untouched here.
    if len(args) == 1 and not kwargs and callable(args[0]):
        return args[0]

    def wrap(fn):
        return fn

    return wrap


class _PlaceholderModule(types.ModuleType):
    # torch.dtype appears inside Union[str, torch.dtype] — typing validates
    # Union members at def time, so this must be a real type.
    dtype = object
    # dtype defaults (torch.bfloat16) only ever matter to the CUDA-graph
    # backend; a sentinel is enough to survive the def-time evaluation.
    bfloat16 = float16 = float32 = None
    inference_mode = staticmethod(_decorator)
    no_grad = staticmethod(_decorator)

    def __getattr__(self, item: str) -> Any:
        raise ModuleNotFoundError(
            f"torch.{item} reached on a torch-free engine install: the GGML "
            "backend needs no torch; the CUDA-graph backend needs "
            "'pip install torch' in the engine environment")


def make() -> Any:
    mod = _PlaceholderModule("torch")
    mod.__path__ = []  # type: ignore[attr-defined]
    return mod
