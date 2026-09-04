"""Central configuration for the CLIP precision benchmark."""
from __future__ import annotations

import os
from pathlib import Path

# --- Dataset -----------------------------------------------------------------
DATASET_ROOT = Path(os.environ.get("CUHK_PEDES_ROOT", "/mnt/data/lab_datasets/CUHK-PEDES"))
ANNOTATION_FILE = DATASET_ROOT / "reid_raw.json"
IMAGE_ROOT = DATASET_ROOT / "imgs"
SPLIT = "test"

# --- Models under test -------------------------------------------------------
# openai/CLIP checkpoint names. ResNet included as a contrast group: its
# precision sensitivity profile differs from the ViT family.
MODELS = ["ViT-B/32", "ViT-B/16", "ViT-L/14", "RN50"]

# --- Precisions under test ---------------------------------------------------
# True weight casting (not autocast): parameters themselves are converted, so
# memory savings are real and numerical drift is not hidden by fp32 accumulation.
DTYPES = ["fp32", "fp16", "bf16"]
REFERENCE_DTYPE = "fp32"

# --- Encoding ----------------------------------------------------------------
# Batch sizes are held FIXED across dtypes so throughput differences are
# attributable to precision rather than to batching.
IMAGE_BATCH_SIZE = 64
TEXT_BATCH_SIZE = 256
WARMUP_BATCHES = 3

# --- Paths -------------------------------------------------------------------
RESULTS_DIR = Path("results")
EMBEDDING_DIR = RESULTS_DIR / "embeddings"
METRICS_DIR = RESULTS_DIR / "metrics"
REPORT_DIR = RESULTS_DIR / "report"
CLIP_DOWNLOAD_ROOT = Path(os.environ.get("CLIP_DOWNLOAD_ROOT", Path.home() / ".cache" / "clip"))

# --- Evaluation --------------------------------------------------------------
RECALL_KS = (1, 5, 10)
# Kendall tau over the full 3074-image ranking for all 6156 queries is
# needlessly expensive; a fixed random subsample is statistically ample.
TAU_QUERY_SAMPLE = 500
TAU_SEED = 0


def model_slug(model_name: str) -> str:
    return model_name.replace("/", "-").replace("@", "-at-")
