"""Model loading with explicit, honest precision control.

The trap this module exists to avoid: `clip.load(..., device="cuda")` returns a
model whose weights are ALREADY fp16 -- `build_model` calls `convert_weights`,
and the `.float()` restore only runs when the device is CPU. Anyone who treats
the default load as an fp32 baseline is silently comparing fp16 against fp16.

Here every precision is reached from a canonical fp32 model via an explicit
cast, and fp16/bf16 use the SAME layer selection (openai's `convert_weights`
policy, generalised over dtype) so the two low-precision configs differ only in
the number format -- not in which layers were converted.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from . import config

DTYPE_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def _convert_weights_to(model: nn.Module, dtype: torch.dtype) -> None:
    """openai/CLIP `convert_weights`, generalised to an arbitrary dtype.

    LayerNorm and the logit scale deliberately stay fp32, exactly as upstream
    leaves them.
    """

    def _convert(l: nn.Module) -> None:
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.to(dtype)
            if l.bias is not None:
                l.bias.data = l.bias.data.to(dtype)

        if isinstance(l, nn.MultiheadAttention):
            attrs = [f"{s}_proj_weight" for s in ("in", "q", "k", "v")]
            attrs += ["in_proj_bias", "bias_k", "bias_v"]
            for attr in attrs:
                tensor = getattr(l, attr, None)
                if tensor is not None:
                    tensor.data = tensor.data.to(dtype)

        for name in ("text_projection", "proj"):
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None and isinstance(attr, torch.Tensor):
                    attr.data = attr.data.to(dtype)

    model.apply(_convert)


def load_model(model_name: str, dtype: str, device: str = "cuda"):
    """Load a CLIP model with weights genuinely cast to `dtype`."""
    import clip

    if dtype not in DTYPE_MAP:
        raise ValueError(f"unknown dtype {dtype!r}; expected one of {list(DTYPE_MAP)}")

    model, preprocess = clip.load(
        model_name,
        device=device,
        jit=False,
        download_root=str(config.CLIP_DOWNLOAD_ROOT),
    )
    # Canonical fp32 starting point regardless of what `clip.load` handed back.
    model = model.float()

    if dtype != "fp32":
        _convert_weights_to(model, DTYPE_MAP[dtype])

    model.eval()
    return model, preprocess


def input_dtype(dtype: str) -> torch.dtype:
    """Dtype the image tensor must be fed in as, to match conv1's weights."""
    return DTYPE_MAP[dtype]


def parameter_bytes(model: nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())


def parameter_dtype_histogram(model: nn.Module) -> dict[str, int]:
    """Element counts per dtype -- proof that the cast actually landed."""
    hist: dict[str, int] = {}
    for p in model.parameters():
        key = str(p.dtype).replace("torch.", "")
        hist[key] = hist.get(key, 0) + p.numel()
    return hist
