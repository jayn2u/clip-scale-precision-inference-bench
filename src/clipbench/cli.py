"""Command line interface: encode -> eval -> report."""
from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import config, encode as encode_mod, metrics as metrics_mod
from .data import load_test_split


def _num(value, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _tag(args) -> str:
    if args.tag:
        return args.tag
    # Smoke runs get their own cache namespace so a truncated gallery can never
    # contaminate the real results.
    return "full" if args.limit is None else f"smoke{args.limit}"


def _split(args):
    return load_test_split(limit=args.limit)


def cmd_encode(args) -> int:
    split = _split(args)
    tag = _tag(args)
    print(f"[encode] tag={tag} images={split.n_images} captions={split.n_captions}")
    for model_name in args.models:
        for dtype in args.dtypes:
            if not args.force and encode_mod.load_embeddings(model_name, dtype, tag, split):
                print(f"[encode] cached, skipping: {model_name} {dtype}")
                continue
            result = encode_mod.encode_split(
                split, model_name, dtype, device=args.device, progress=not args.quiet
            )
            out = encode_mod.save_embeddings(result, split, model_name, dtype, tag)
            s = result["stats"]
            print(
                f"[encode] {model_name:>10} {dtype:<5} "
                f"img {_num(s.get('image_throughput_per_second'), 1)}/s  "
                f"txt {_num(s.get('text_throughput_per_second'), 1)}/s  "
                f"vram {_num((s.get('image_peak_vram_bytes') or 0) / 2**20, 0)}MiB  -> {out}"
            )
    return 0


def cmd_eval(args) -> int:
    split = _split(args)
    tag = _tag(args)
    caption_ids = np.asarray(split.caption_ids)
    image_ids = np.asarray(split.image_ids)

    rows: list[dict] = []
    for model_name in args.models:
        reference_sim = None
        reference_feats = None
        # The reference must be evaluated first for the drift columns to exist.
        dtypes = sorted(args.dtypes, key=lambda d: d != config.REFERENCE_DTYPE)
        for dtype in dtypes:
            cached = encode_mod.load_embeddings(model_name, dtype, tag, split)
            if cached is None:
                print(f"[eval] missing embeddings: {model_name} {dtype} (run encode)", file=sys.stderr)
                continue
            sim = metrics_mod.similarity_matrix(
                cached["text_features"], cached["image_features"]
            )
            row = {
                "model": model_name,
                "dtype": dtype,
                **metrics_mod.retrieval_metrics(sim, caption_ids, image_ids, config.RECALL_KS),
            }
            for key in (
                "image_throughput_per_second",
                "text_throughput_per_second",
                "image_peak_vram_bytes",
                "text_peak_vram_bytes",
                "image_wall_seconds",
                "text_wall_seconds",
                "parameter_bytes",
                "captions_truncated",
                "captions_total",
                "parameter_dtypes",
                "torch_version",
                "cuda_device",
            ):
                row[key] = cached["stats"].get(key)

            if dtype == config.REFERENCE_DTYPE:
                reference_sim = sim
                reference_feats = cached
            elif reference_sim is not None:
                row["image_drift"] = metrics_mod.embedding_drift(
                    reference_feats["image_features"], cached["image_features"]
                )
                row["text_drift"] = metrics_mod.embedding_drift(
                    reference_feats["text_features"], cached["text_features"]
                )
                row.update(
                    metrics_mod.ranking_agreement(
                        reference_sim, sim, config.TAU_QUERY_SAMPLE, config.TAU_SEED
                    )
                )
            rows.append(row)
            print(
                f"[eval] {model_name:>10} {dtype:<5} "
                f"R@1 {row['R@1']:.2f}  R@5 {row['R@5']:.2f}  "
                f"R@10 {row['R@10']:.2f}  mAP {row['mAP']:.2f}"
            )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tag": tag,
        "dataset": {
            "name": "CUHK-PEDES",
            "annotation_file": str(config.ANNOTATION_FILE),
            "split": config.SPLIT,
            "n_images": split.n_images,
            "n_captions": split.n_captions,
            "n_identities": len(set(split.image_ids)),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "config": {
            "image_batch_size": config.IMAGE_BATCH_SIZE,
            "text_batch_size": config.TEXT_BATCH_SIZE,
            "warmup_batches": config.WARMUP_BATCHES,
            "reference_dtype": config.REFERENCE_DTYPE,
        },
        "results": rows,
    }
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.METRICS_DIR / f"results_{tag}.json"
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[eval] wrote {out}")
    return 0


def cmd_report(args) -> int:
    from .report import build_report

    tag = _tag(args)
    src = config.METRICS_DIR / f"results_{tag}.json"
    if not src.exists():
        print(f"[report] no results at {src}; run eval first", file=sys.stderr)
        return 1
    with open(src) as f:
        payload = json.load(f)
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = config.REPORT_DIR / f"report_{tag}.html"
    out.write_text(build_report(payload))
    print(f"[report] wrote {out}")
    return 0


def cmd_run(args) -> int:
    for step in (cmd_encode, cmd_eval, cmd_report):
        code = step(args)
        if code:
            return code
    return 0


def main(argv: list[str] | None = None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--models", nargs="+", default=config.MODELS)
    common.add_argument("--dtypes", nargs="+", default=config.DTYPES, choices=config.DTYPES)
    common.add_argument("--device", default="cuda")
    common.add_argument(
        "--limit", type=int, default=None,
        help="use only the first N test images (smoke run; separate cache namespace)",
    )
    common.add_argument("--tag", default=None, help="override the cache/results namespace")
    common.add_argument("--force", action="store_true", help="re-encode even if cached")
    common.add_argument("--quiet", action="store_true")

    parser = argparse.ArgumentParser(
        prog="clipbench",
        description="CLIP model/precision benchmark on CUHK-PEDES text-to-image retrieval",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name, fn in (
        ("encode", cmd_encode),
        ("eval", cmd_eval),
        ("report", cmd_report),
        ("run", cmd_run),
    ):
        sub.add_parser(name, parents=[common]).set_defaults(func=fn)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
