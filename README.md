# CLIP precision & scale benchmark — CUHK-PEDES text→image retrieval

Measures how CLIP model scale and **weight precision** affect zero-shot
text-to-image person retrieval on the CUHK-PEDES test split.

## What it measures

| Axis | Metrics |
|---|---|
| Retrieval accuracy | R@1, R@5, R@10, mAP |
| Speed & memory | image/text encoding throughput, peak VRAM, weight bytes |
| Numerical drift vs fp32 | per-vector cosine similarity, Top-1 agreement, Top-10 overlap, Kendall τ |

Models: `ViT-B/32`, `ViT-B/16`, `ViT-L/14`, `RN50` (ResNet as a contrast group).
Precisions: `fp32`, `fp16`, `bf16`.

## Two things that make the numbers trustworthy

1. **`clip.load()` returns fp16 weights on CUDA.** `build_model` calls
   `convert_weights`, and the `.float()` restore only runs when the device is
   CPU. A naive "fp32 baseline" is therefore fp16 compared against fp16. Here
   every precision is reached from an explicit `model.float()` and then cast.
2. **fp16 and bf16 convert the *same* layers.** `models._convert_weights_to`
   generalises openai's `convert_weights` policy over dtype, so the two
   low-precision configs differ only in the number format — LayerNorm stays
   fp32 in both, exactly as upstream leaves it. Casting with a plain
   `model.to(bfloat16)` would have converted LayerNorm too and made the
   comparison invalid.

Precision is **true weight casting**, not `torch.autocast` — autocast keeps
fp32 master weights and fp32 accumulation, which hides the drift this benchmark
exists to measure and yields no memory saving.

## Protocol

CUHK-PEDES `test` split: 3,074 gallery images, 6,156 caption queries, 1,000
identities. Each caption queries the full gallery; a hit is a **person `id`
match**, so other images of the same identity count — the protocol used by the
CUHK-PEDES paper and subsequent ReID work.

Image preprocessing is CLIP's stock transform (square center-crop at the
model's native resolution). Pedestrian crops are tall, so this does clip head
and feet, but changing it would confound the precision axis and break
comparability with published zero-shot numbers.

Captions over CLIP's 77-token context are truncated rather than dropped, to
keep the 6,156-query protocol intact; the truncated count is reported.

Timing covers the CUDA-synchronised forward pass only, after warmup, taking the
median over full batches. **Batch size is fixed across precisions** so speedups
are attributable to precision rather than to batching.

## Usage

```bash
uv sync
uv run clipbench run --limit 200          # smoke run, separate cache namespace
uv run clipbench run                      # full benchmark
uv run clipbench report                   # rebuild the report from cached metrics
```

Subcommands: `encode` (embed + cache), `eval` (metrics), `report` (HTML), `run` (all three).
Useful flags: `--models`, `--dtypes`, `--limit N`, `--tag NAME`, `--force`, `--device`.

Everything under `results/` is generated and gitignored — rerun the CLI to
reproduce it.

Embeddings are cached at `results/embeddings/<tag>/<model>/<dtype>/` and are
**always stored fp32** — re-quantising at the storage layer would corrupt the
drift measurement. Each cache carries an order hash over item identity *and*
order; a mismatch invalidates the cache rather than silently producing wrong
metrics.

## Requirements

Torch from the `cu128` index — one of the machines this runs on is an RTX
5070 Ti (Blackwell, sm_120) and older wheels will not run on it. `openai/CLIP`
is installed from git; its `requirements.txt` leaves `torch` unpinned, so there
is no conflict.

Dataset path is auto-detected from a short candidate list and overridable with
`CUHK_PEDES_ROOT`. Checkpoint cache defaults to `~/.cache/clip`; override with
`CLIP_DOWNLOAD_ROOT`.

## Hardware

Committed results in `results/` were measured on an **RTX A6000 (48 GiB)**;
`cuda_device` and `torch_version` are recorded per row in the metrics JSON.
Retrieval accuracy is hardware-independent (fp32 rows reproduce to the second
decimal across two GPUs), but throughput and the *speedup ratio* from precision
are not — ViT-L/14 fp16 is 2.80x fp32 on an RTX 5070 Ti and 3.81x on the A6000.
Never mix speed columns measured on different devices.

## Scope

Zero-shot only — no CUHK-PEDES fine-tuning. Absolute recall sits far below
fine-tuned SOTA (IRRA-class methods reach R@1 > 70%); the precision-vs-precision
comparison within a model is unaffected. int8/FP8 and TensorRT are out of scope
for this pass.
