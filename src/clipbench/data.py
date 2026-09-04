"""CUHK-PEDES test split loading.

Standard protocol: the test split holds 3,074 images and 6,156 captions over
1,000 person identities. A caption is correct for an image when their person
`id` matches -- other images of the same person count as hits, as in the
CUHK-PEDES paper and the ReID literature that followed it.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from . import config


@dataclass
class TestSplit:
    image_paths: list[Path]
    image_ids: list[int]
    captions: list[str]
    caption_ids: list[int]
    # index into image_paths of the image each caption was written for
    caption_image_index: list[int]

    @property
    def n_images(self) -> int:
        return len(self.image_paths)

    @property
    def n_captions(self) -> int:
        return len(self.captions)

    def order_hash(self) -> str:
        """Fingerprint of item identity AND order.

        Cached embeddings are only valid for the exact ordering they were
        produced from; this hash is what makes a stale cache fail loudly
        instead of silently producing wrong metrics.
        """
        h = hashlib.sha256()
        for p, i in zip(self.image_paths, self.image_ids):
            h.update(f"{p}|{i}\n".encode())
        for c, i in zip(self.captions, self.caption_ids):
            h.update(f"{c}|{i}\n".encode())
        return h.hexdigest()[:16]


def load_test_split(
    annotation_file: Path | None = None,
    image_root: Path | None = None,
    split: str = config.SPLIT,
    limit: int | None = None,
) -> TestSplit:
    annotation_file = annotation_file or config.ANNOTATION_FILE
    image_root = image_root or config.IMAGE_ROOT

    with open(annotation_file) as f:
        records = json.load(f)

    entries = [r for r in records if r["split"] == split]
    entries.sort(key=lambda r: r["file_path"])  # deterministic order
    if limit is not None:
        entries = entries[:limit]

    image_paths: list[Path] = []
    image_ids: list[int] = []
    captions: list[str] = []
    caption_ids: list[int] = []
    caption_image_index: list[int] = []

    for entry in entries:
        idx = len(image_paths)
        image_paths.append(image_root / entry["file_path"])
        image_ids.append(int(entry["id"]))
        for caption in entry["captions"]:
            captions.append(caption.strip())
            caption_ids.append(int(entry["id"]))
            caption_image_index.append(idx)

    return TestSplit(
        image_paths=image_paths,
        image_ids=image_ids,
        captions=captions,
        caption_ids=caption_ids,
        caption_image_index=caption_image_index,
    )
