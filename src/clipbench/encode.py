"""Encoding with cached embeddings plus speed and memory instrumentation.

Timing covers the GPU forward pass only (CUDA-synchronised, warmed up, median
over batches). Image decoding and CPU preprocessing are identical across
precisions, so folding them into the number would only dilute the effect being
measured. Wall-clock for the whole pass is recorded separately.
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from . import config, models
from .data import TestSplit


class ImageDataset(Dataset):
    def __init__(self, paths, preprocess):
        self.paths = paths
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        image = Image.open(self.paths[idx]).convert("RGB")
        return self.preprocess(image)


def _timed_forward(fn, batch, warmup: int = 0) -> tuple[torch.Tensor, float]:
    for _ in range(warmup):
        with torch.no_grad():
            fn(batch)
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        out = fn(batch)
    torch.cuda.synchronize()
    return out, time.perf_counter() - start


def encode_split(
    split: TestSplit,
    model_name: str,
    dtype: str,
    device: str = "cuda",
    image_batch_size: int = config.IMAGE_BATCH_SIZE,
    text_batch_size: int = config.TEXT_BATCH_SIZE,
    num_workers: int = 8,
    progress: bool = True,
) -> dict:
    import clip

    model, preprocess = models.load_model(model_name, dtype, device=device)
    tensor_dtype = models.input_dtype(dtype)

    stats: dict = {
        "model": model_name,
        "dtype": dtype,
        "parameter_bytes": models.parameter_bytes(model),
        "parameter_dtypes": models.parameter_dtype_histogram(model),
        "image_batch_size": image_batch_size,
        "text_batch_size": text_batch_size,
    }

    # ---- images -------------------------------------------------------------
    loader = DataLoader(
        ImageDataset(split.image_paths, preprocess),
        batch_size=image_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    torch.cuda.reset_peak_memory_stats()
    image_feats: list[np.ndarray] = []
    batch_times: list[float] = []
    wall_start = time.perf_counter()
    first = True
    for batch in tqdm(loader, desc=f"{model_name} {dtype} images", disable=not progress):
        batch = batch.to(device, dtype=tensor_dtype, non_blocking=True)
        out, elapsed = _timed_forward(
            model.encode_image, batch, warmup=config.WARMUP_BATCHES if first else 0
        )
        first = False
        # Full-batch timings only; a short trailing batch is not comparable.
        batch_times.append((batch.shape[0], elapsed))
        image_feats.append(out.float().cpu().numpy())
    stats["image_wall_seconds"] = time.perf_counter() - wall_start
    stats["image_peak_vram_bytes"] = int(torch.cuda.max_memory_allocated())
    per_item, full_only = _median_seconds_per_item(batch_times, image_batch_size)
    stats["image_seconds_per_item"] = per_item
    stats["image_throughput_per_second"] = 1.0 / per_item if per_item else None
    stats["image_timing_full_batches_only"] = full_only

    # ---- text ---------------------------------------------------------------
    # CLIP's context is 77 BPE tokens; longer captions are truncated. Every
    # precision sees the identical truncation, so the comparison axis is clean,
    # but the count is recorded so the absolute numbers can be read honestly.
    tokens = clip.tokenize(split.captions, truncate=True)
    stats["captions_truncated"] = count_truncated(split.captions)
    stats["captions_total"] = len(split.captions)

    torch.cuda.reset_peak_memory_stats()
    text_feats: list[np.ndarray] = []
    batch_times = []
    wall_start = time.perf_counter()
    first = True
    n_text_batches = (len(tokens) + text_batch_size - 1) // text_batch_size
    for i in tqdm(
        range(0, len(tokens), text_batch_size),
        total=n_text_batches,
        desc=f"{model_name} {dtype} text",
        disable=not progress,
    ):
        batch = tokens[i : i + text_batch_size].to(device)
        out, elapsed = _timed_forward(
            model.encode_text, batch, warmup=config.WARMUP_BATCHES if first else 0
        )
        first = False
        batch_times.append((batch.shape[0], elapsed))
        text_feats.append(out.float().cpu().numpy())
    stats["text_wall_seconds"] = time.perf_counter() - wall_start
    stats["text_peak_vram_bytes"] = int(torch.cuda.max_memory_allocated())
    per_item, full_only = _median_seconds_per_item(batch_times, text_batch_size)
    stats["text_seconds_per_item"] = per_item
    stats["text_throughput_per_second"] = 1.0 / per_item if per_item else None
    stats["text_timing_full_batches_only"] = full_only

    del model
    torch.cuda.empty_cache()

    # Embeddings are always stored fp32: re-quantising at the storage layer
    # would corrupt the very drift this benchmark measures.
    return {
        "image_features": np.concatenate(image_feats).astype(np.float32),
        "text_features": np.concatenate(text_feats).astype(np.float32),
        "stats": stats,
    }


def count_truncated(captions, context_length: int = 77) -> int:
    """How many captions overflow CLIP's 77-token context and get clipped."""
    from clip.simple_tokenizer import SimpleTokenizer

    tokenizer = SimpleTokenizer()
    # +2 accounts for the <|startoftext|> and <|endoftext|> sentinels.
    return sum(1 for c in captions if len(tokenizer.encode(c)) + 2 > context_length)


def _median_seconds_per_item(
    batch_times: list[tuple[int, float]], batch_size: int
) -> tuple[float | None, bool]:
    """Median per-item forward time, preferring full batches.

    A short trailing batch under-utilises the GPU and is not comparable, so
    full batches are used when any exist. Smoke runs smaller than one batch
    have none; those fall back to whatever ran, flagged so the number is not
    mistaken for a comparable measurement.
    """
    full = [t / n for n, t in batch_times if n == batch_size]
    if full:
        return statistics.median(full), True
    partial = [t / n for n, t in batch_times]
    if partial:
        return statistics.median(partial), False
    return None, False


def cache_dir(model_name: str, dtype: str, tag: str = "full") -> Path:
    return config.EMBEDDING_DIR / tag / config.model_slug(model_name) / dtype


def save_embeddings(result: dict, split: TestSplit, model_name: str, dtype: str, tag: str) -> Path:
    out = cache_dir(model_name, dtype, tag)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "image.npy", result["image_features"])
    np.save(out / "text.npy", result["text_features"])
    meta = dict(result["stats"])
    meta.update(
        {
            "order_hash": split.order_hash(),
            "n_images": split.n_images,
            "n_captions": split.n_captions,
            "torch_version": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "tag": tag,
        }
    )
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return out


def load_embeddings(model_name: str, dtype: str, tag: str, split: TestSplit) -> dict | None:
    """Return cached embeddings, or None if absent or stale."""
    out = cache_dir(model_name, dtype, tag)
    meta_path = out / "meta.json"
    if not meta_path.exists():
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    if meta.get("order_hash") != split.order_hash():
        return None
    return {
        "image_features": np.load(out / "image.npy"),
        "text_features": np.load(out / "text.npy"),
        "stats": meta,
    }
