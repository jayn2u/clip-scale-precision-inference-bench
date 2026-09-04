# CLIP Scale & Precision Inference Benchmark

This benchmark measures how OpenAI CLIP **model scale** and **weight precision** affect zero-shot text-to-image person retrieval. It uses the CUHK-PEDES test split and measures retrieval quality, throughput, GPU memory, numerical drift from fp32, and ranking changes.

## Benchmark scope

| Category | What is measured |
| --- | --- |
| Models | `ViT-B/32`, `ViT-B/16`, `ViT-L/14`, `RN50` |
| Precisions | `fp32`, `fp16`, `bf16` |
| Retrieval quality | R@1, R@5, R@10, mAP |
| Performance and memory | Image/text throughput, wall time, peak VRAM, weight bytes |
| Change from fp32 | Embedding cosine similarity, Top-1 agreement, Top-10 overlap, Kendall tau |

The current scope is zero-shot inference only. CUHK-PEDES fine-tuning, INT8/FP8 quantization, and TensorRT optimization are not included.

## Requirements

- Python 3.12 or later
- [`uv`](https://docs.astral.sh/uv/)
- An NVIDIA GPU with CUDA support
- A CUDA 12.8-compatible driver
- The CUHK-PEDES dataset

PyTorch and torchvision are installed from the CUDA 12.8 index configured in `pyproject.toml`. The implementation directly uses `torch.cuda` for timing and memory measurements, so a CUDA environment is effectively required. `--device cpu` is not a complete CPU execution mode.

The CUDA 12.8 wheels are also required by the RTX 5070 Ti test machine, whose Blackwell GPU uses sm_120. OpenAI CLIP is installed from Git; its dependency metadata does not pin PyTorch, so it does not conflict with the configured wheel.

## Quick start

### 1. Install dependencies

```bash
uv sync
```

CLIP checkpoints are downloaded automatically on first use. The default cache is `~/.cache/clip`; set `CLIP_DOWNLOAD_ROOT` to use another location.

### 2. Prepare the dataset

The dataset root is automatically detected at either of these locations:

- `/mnt/data/lab_datasets/CUHK-PEDES`
- `/data/jayn2u/lab_datasets/CUHK-PEDES`

For any other location, set `CUHK_PEDES_ROOT`.

```text
CUHK-PEDES/
├── reid_raw.json
└── imgs/
    └── ...
```

```bash
export CUHK_PEDES_ROOT=/path/to/CUHK-PEDES
```

Values in the annotation file's `file_path` field are resolved relative to `imgs/`. The standard test split contains 3,074 images, 6,156 captions, and 1,000 person identities.

### 3. Run a smoke test

Start with a small subset to verify the installation and dataset path.

```bash
uv run clipbench run --limit 200
```

Results from `--limit 200` are automatically stored in the `smoke200` namespace, so they cannot contaminate the full-run cache. Small runs may measure partial batches and should not be used for formal throughput comparisons.

### 4. Run the full benchmark

```bash
uv run clipbench run
```

This executes `encode` → `eval` → `report`. The generated report is written to `results/report/report_full.html`.

## Hardware and result comparability

The results committed to this repository were measured on an **RTX A6000 (48 GiB)**. Every metrics row records `cuda_device` and `torch_version`.

Retrieval accuracy is largely hardware-independent—the fp32 results reproduced to two decimal places across two GPUs—but throughput and precision speedup ratios are not. For example, ViT-L/14 fp16 was 2.80× faster than fp32 on an RTX 5070 Ti and 3.81× faster on the A6000. Do not combine speed columns measured on different devices as if they came from one experiment.

## Commands

```bash
uv run clipbench encode [options]  # Generate and cache embeddings
uv run clipbench eval [options]    # Calculate retrieval and drift metrics
uv run clipbench report [options]  # Build an HTML report from existing metrics
uv run clipbench run [options]     # Run all three stages in order
```

All subcommands accept the following options:

| Option | Description |
| --- | --- |
| `--models MODEL ...` | CLIP models to run |
| `--dtypes {fp32,fp16,bf16} ...` | Precisions to run |
| `--limit N` | Use only the first N test images for a smoke run |
| `--tag NAME` | Override the cache and result namespace |
| `--force` | Re-encode even when a valid cache exists |
| `--device DEVICE` | Device on which to load the model; defaults to `cuda` |
| `--quiet` | Disable progress bars |

Examples:

```bash
# Compare fp32 and fp16 for one model
uv run clipbench run --models ViT-B/32 --dtypes fp32 fp16 --tag vit-b32

# Reuse embeddings and rebuild only the metrics and report
uv run clipbench eval --tag full
uv run clipbench report --tag full

# Ignore an existing cache and encode again
uv run clipbench encode --models RN50 --dtypes fp32 --force
```

Drift metrics require an fp32 embedding cache for the same model and tag. Include `fp32` in `--dtypes` when running a comparison experiment.

## Output and cache structure

```text
results/
├── embeddings/<tag>/<model>/<dtype>/
│   ├── image.npy
│   ├── text.npy
│   └── meta.json
├── metrics/results_<tag>.json
└── report/report_<tag>.html
```

- Embeddings are always stored as fp32 NumPy arrays, regardless of the model precision being measured. Quantizing them again at the storage boundary would corrupt the drift measurement.
- Each cache records a hash of item identity and order. A cache that does not match the current split is automatically invalidated.
- Without `--tag`, a full run uses `full` and a `--limit N` run uses `smokeN`.
- Large embeddings, logs, and smoke/sanity artifacts are excluded from Git.
- Metrics JSON and HTML reports from reproducible full runs may be committed as project results.

## Evaluation protocol

Each caption is a query against the complete test-image gallery. A result is correct when its **person ID matches** the query identity, not only when it is the exact image associated with the caption. Other images of the same person therefore count as positives, following the CUHK-PEDES paper and subsequent person ReID work.

- Images use each CLIP model's stock preprocessing transform.
- Captions longer than CLIP's 77-token context are truncated rather than removed, and the truncated count is recorded.
- Throughput measures CUDA-synchronized forward passes after warmup.
- Timing prefers the median per-item time from full batches and excludes the final partial batch when possible.
- Image and text batch sizes are fixed at 64 and 256 across precisions so that batching does not confound precision comparisons.

## Precision implementation invariants

This benchmark compares **real weight casting**, not `torch.autocast`. Autocast retains fp32 master weights and some fp32 operations, which would hide the memory reduction and numerical drift this benchmark is intended to measure.

Because `clip.load()` may return fp16 weights on CUDA, every model is first explicitly restored to fp32 and then converted to its target dtype. fp16 and bf16 use the same layer-selection policy, while LayerNorm and the logit scale remain fp32. Their measured difference therefore comes from the number format, not from converting different layers.

Preserve these invariants when changing the implementation:

1. Every dtype must start from the same fp32 reference model.
2. fp16 and bf16 must convert the same kinds of layers.
3. Batch sizes, preprocessing, data order, and truncation policy must be identical across dtypes.
4. Cached embeddings must be stored as fp32.
5. Results without an fp32 reference must not be interpreted as drift measurements.

## Interpreting results

- R@K is the percentage of queries for which at least one matching person ID appears in the first K results.
- mAP reflects where all positive images for an identity appear throughout the ranking.
- Embedding cosine similarity shows how far a vector moved from the fp32 reference.
- Top-1 agreement and Top-10 overlap show whether that movement changed the highest-ranked results.
- Kendall tau measures full-ranking changes over a fixed-seed sample of up to 500 queries.

Do not compare the absolute retrieval accuracy directly with fine-tuned state-of-the-art models. The primary question is the relative effect of changing precision within the same model.

## Project structure

```text
src/clipbench/
├── cli.py      # CLI and encode/eval/report workflow
├── config.py   # Models, dtypes, batch sizes, paths, and evaluation settings
├── data.py     # CUHK-PEDES test split loading and order hashing
├── models.py   # Model loading and explicit weight casting
├── encode.py   # Embedding generation, performance measurement, and caching
├── metrics.py  # Retrieval quality and fp32-relative drift metrics
└── report.py   # Self-contained HTML report generation
```

Check `src/clipbench/config.py` first when changing benchmark settings. Adding a model or precision also requires reviewing model loading, input dtypes, and the report's colors and presentation.

## Development guidelines

1. Before making a change, establish a reproducible baseline with a small `--limit` run.
2. Run the same model, dtype, limit, and tag after the change.
3. Inspect the parameter dtype histogram, weight bytes, peak VRAM, and fp32 agreement in addition to retrieval metrics.
4. When changing the protocol or defaults, use a new `--tag` instead of mixing the results with an older experiment.
5. Do not commit large embeddings. Review metrics and reports from full runs together with the code and configuration that produced them.

The CLI surface can be checked without running a benchmark:

```bash
uv run clipbench --help
uv run clipbench run --help
```

There is currently no automated test suite. At minimum, validate changes with the CLI help commands and a small smoke benchmark. Add unit tests first when extending calculation logic.
