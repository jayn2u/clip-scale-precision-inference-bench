from __future__ import annotations

import html
import json
import math
import os
import re
import signal
import struct
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "results" / "metrics" / "results_full.json"
CAPACITY_PATH = ROOT / "results" / "metrics" / "train_capacity_a6000.md"
OUT_DIR = ROOT / "docs" / "figures"

SCALE = 2
INK = "#1b2433"
MUTED = "#5c6b80"
GRID = "#e6edf5"
AXIS = "#c5d0dc"
PAPER = "#ffffff"
FP32 = "#3a6ea5"
FP16 = "#2e7d63"
BF16 = "#b4552f"
SANS = "Inter, Helvetica, Arial, sans-serif"
MONO = "JetBrains Mono, ui-monospace, SFMono-Regular, Menlo, monospace"

MODELS = ["ViT-B/32", "ViT-B/16", "ViT-L/14", "RN50"]
DTYPES = ["fp32", "fp16", "bf16"]
DTYPE_COLOR = {"fp32": FP32, "fp16": FP16, "bf16": BF16}
MODEL_COLOR = {"ViT-L/14": "#3a6ea5", "ViT-B/16": "#7a4e9a", "ViT-B/32": "#2e7d63"}


def esc(value) -> str:
    return html.escape(str(value), quote=True)


class Svg:
    def __init__(self, w: float, h: float):
        self.w = w
        self.h = h
        self.parts: list[str] = []

    def add(self, item: str) -> None:
        self.parts.append(item)

    def rect(self, x, y, w, h, fill, rx=0, stroke="none", sw=0, opacity=1) -> None:
        self.add(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
            f'fill="{fill}" fill-opacity="{opacity}" rx="{rx}" stroke="{stroke}" '
            f'stroke-width="{sw}"/>'
        )

    def line(self, x1, y1, x2, y2, stroke, sw=1, dash=None, opacity=1) -> None:
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{stroke}" stroke-opacity="{opacity}" stroke-width="{sw}"{extra}/>'
        )

    def circle(self, cx, cy, r, fill, stroke="none", sw=0) -> None:
        self.add(
            f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}"/>'
        )

    def polyline(self, points, stroke, sw=1.8, dash=None, opacity=1) -> None:
        if len(points) < 2:
            return
        pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(
            f'<polyline points="{pts}" fill="none" stroke="{stroke}" '
            f'stroke-opacity="{opacity}" stroke-width="{sw}" '
            f'stroke-linejoin="round" stroke-linecap="round"{extra}/>'
        )

    def text(self, x, y, value, size=13, fill=INK, anchor="start", weight="400", family=SANS, transform=None) -> None:
        tr = f' transform="{transform}"' if transform else ""
        self.add(
            f'<text x="{x:.2f}" y="{y:.2f}" fill="{fill}" font-size="{size}" '
            f'font-weight="{weight}" font-family="{family}" text-anchor="{anchor}"{tr}>'
            f'{esc(value)}</text>'
        )

    def render(self, title: str, desc: str) -> str:
        body = "\n".join(self.parts)
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w * SCALE:.0f}" '
            f'height="{self.h * SCALE:.0f}" viewBox="0 0 {self.w:.0f} {self.h:.0f}" role="img">\n'
            f"<title>{esc(title)}</title>\n<desc>{esc(desc)}</desc>\n"
            f'<rect width="{self.w:.0f}" height="{self.h:.0f}" fill="{PAPER}"/>\n'
            f"{body}\n</svg>\n"
        )


def legend(svg: Svg, right: float, y: float, items: list[tuple[str, str]]) -> None:
    widths = [14 + 6.6 * len(label) + 18 for _, label in items]
    cursor = right - sum(widths)
    for (color, label), width in zip(items, widths):
        svg.rect(cursor, y - 9, 11, 11, color, rx=2)
        svg.text(cursor + 16, y, label, size=12.5, fill=INK)
        cursor += width


def header(svg: Svg, title: str, subtitle: str, items: list[tuple[str, str]] | None = None) -> None:
    svg.text(28, 34, title, size=20, weight="640")
    svg.text(28, 56, subtitle, size=12.5, fill=MUTED)
    if items:
        legend(svg, svg.w - 28, 32, items)


def footer(svg: Svg, note: str) -> None:
    svg.text(28, svg.h - 16, note, size=11, fill=MUTED)


def y_of(v, y, h, vmax) -> float:
    return y + h - (v / vmax) * h


def grouped_panel(svg: Svg, box, models, series, vmax, ticks, ylabel, tick_fmt, value_fmt, labels="horizontal") -> None:
    x, y, w, h = box
    svg.text(x - 46, y + h / 2, ylabel, size=12, fill=MUTED, anchor="middle", transform=f"rotate(-90 {x - 46:.2f} {y + h / 2:.2f})")
    for tick in ticks:
        yy = y_of(tick, y, h, vmax)
        svg.line(x, yy, x + w, yy, GRID, 1)
        svg.text(x - 8, yy + 4, tick_fmt(tick), size=11, fill=MUTED, anchor="end", family=MONO)
    svg.line(x, y + h, x + w, y + h, AXIS, 1.2)
    n_g = len(models)
    n_s = len(series)
    group_w = w / n_g
    inner = group_w * 0.78
    bar_w = inner / n_s
    for i, model in enumerate(models):
        gx = x + i * group_w + (group_w - inner) / 2
        for j, (_name, color, values) in enumerate(series):
            value = values[i]
            if value is None:
                continue
            bh = max((value / vmax) * h, 0)
            bx = gx + j * bar_w
            by = y + h - bh
            svg.rect(bx + 1.4, by, max(bar_w - 2.8, 1), bh, color)
            if labels:
                label = value_fmt(value)
                cx = bx + bar_w / 2
                if labels == "vertical":
                    svg.text(
                        0,
                        0,
                        label,
                        size=10,
                        fill="#3e4c60",
                        anchor="start",
                        family=MONO,
                        transform=f"translate({cx:.2f},{by - 5:.2f}) rotate(-90)",
                    )
                else:
                    svg.text(cx, by - 6, label, size=10, fill="#3e4c60", anchor="middle", family=MONO)
        svg.text(x + i * group_w + group_w / 2, y + h + 20, model, size=12, anchor="middle")


def png_size(path: Path) -> tuple[int, int] | None:
    data = path.read_bytes()
    if len(data) < 24 or not data.startswith(b"\x89PNG\r\n\x1a\n") or not data.endswith(b"IEND\xaeB`\x82"):
        return None
    return struct.unpack(">II", data[16:24])


def rasterize(svg_text: str, png_path: Path, width: int, height: int) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    if png_path.exists():
        png_path.unlink()
    with tempfile.TemporaryDirectory() as tmp:
        svg_path = Path(tmp) / "figure.svg"
        svg_path.write_text(svg_text, encoding="utf-8")
        cmd = [
            "google-chrome",
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--hide-scrollbars",
            "--force-device-scale-factor=1",
            f"--user-data-dir={tmp}/profile",
            "--virtual-time-budget=1500",
            "--default-background-color=FFFFFFFF",
            f"--window-size={width},{height}",
            f"--screenshot={png_path}",
            svg_path.as_uri(),
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        deadline = time.time() + 20
        got = None
        while time.time() < deadline:
            if png_path.exists():
                got = png_size(png_path)
                if got is not None:
                    break
            if proc.poll() is not None and png_path.exists():
                got = png_size(png_path)
                if got is not None:
                    break
            time.sleep(0.1)
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    if got != (width, height):
        raise SystemExit(f"{png_path.name}: expected {width}x{height}, got {got}")


def save(svg: Svg, name: str, title: str, desc: str) -> None:
    text = svg.render(title, desc)
    svg_path = OUT_DIR / f"{name}.svg"
    png_path = OUT_DIR / f"{name}.png"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    svg_path.write_text(text, encoding="utf-8")
    rasterize(text, png_path, int(svg.w * SCALE), int(svg.h * SCALE))


def load_metrics() -> dict:
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def index_rows(payload: dict) -> dict[tuple[str, str], dict]:
    return {(row["model"], row["dtype"]): row for row in payload["results"]}


def mib(n: int) -> float:
    return n / 2**20


def series_from(rows, dtypes, getter):
    out = []
    for dtype in dtypes:
        out.append((dtype, DTYPE_COLOR[dtype], [getter(rows[(model, dtype)]) for model in MODELS]))
    return out


def fmt_num(value, digits=0) -> str:
    if digits == 0:
        return f"{value:,.0f}"
    return f"{value:,.{digits}f}"


def recall_figure(payload, rows) -> None:
    svg = Svg(940, 520)
    header(
        svg,
        "Recall@1 by model and precision",
        "CUHK-PEDES test  ·  6,156 captions  ·  3,074 images  ·  person-ID hit",
        [(FP32, "fp32"), (FP16, "fp16"), (BF16, "bf16")],
    )
    grouped_panel(
        svg,
        (78, 96, 832, 340),
        MODELS,
        series_from(rows, DTYPES, lambda r: r["R@1"]),
        14,
        [0, 2, 4, 6, 8, 10, 12],
        "R@1 (%)",
        lambda v: f"{v:.0f}",
        lambda v: f"{v:.2f}",
    )
    device = payload["results"][0]["cuda_device"]
    footer(svg, f"Source: results/metrics/results_full.json  ·  {device}  ·  torch {payload['results'][0]['torch_version']}")
    save(svg, "recall_at_1", "Recall@1 by model and precision", "Grouped bars of zero-shot Recall@1 for four CLIP models at fp32, fp16, and bf16.")


def throughput_figure(payload, rows) -> None:
    svg = Svg(1040, 540)
    header(
        svg,
        "Encoding throughput",
        "Full batches only  ·  image batch 64  ·  text batch 256  ·  median per-item time",
        [(FP32, "fp32"), (FP16, "fp16"), (BF16, "bf16")],
    )
    grouped_panel(
        svg,
        (78, 108, 450, 340),
        MODELS,
        series_from(rows, DTYPES, lambda r: r["image_throughput_per_second"]),
        5200,
        [0, 1000, 2000, 3000, 4000, 5000],
        "Images / s",
        lambda v: f"{v:,.0f}",
        lambda v: f"{v:,.0f}",
        labels=None,
    )
    grouped_panel(
        svg,
        (590, 108, 420, 340),
        MODELS,
        series_from(rows, DTYPES, lambda r: r["text_throughput_per_second"]),
        7500,
        [0, 1500, 3000, 4500, 6000],
        "Captions / s",
        lambda v: f"{v:,.0f}",
        lambda v: f"{v:,.0f}",
        labels=None,
    )
    svg.text(78, 92, "Image tower", size=13, weight="620")
    svg.text(590, 92, "Text tower", size=13, weight="620")
    footer(
        svg,
        f"Source: results/metrics/results_full.json  ·  {payload['results'][0]['cuda_device']}",
    )
    save(svg, "throughput", "Image and text encoding throughput", "Grouped bars of image and text throughput for each CLIP model and precision.")


def memory_figure(payload, rows) -> None:
    svg = Svg(1160, 540)
    header(
        svg,
        "Peak VRAM and weight size",
        "MiB = bytes / 2^20  ·  peak is torch.cuda max allocated during that tower",
        [(FP32, "fp32"), (FP16, "fp16"), (BF16, "bf16")],
    )
    panels = [
        (80, "Image peak", "image_peak_vram_bytes", 3200, [0, 800, 1600, 2400, 3200]),
        (450, "Text peak", "text_peak_vram_bytes", 1600, [0, 400, 800, 1200, 1600]),
        (820, "Weights", "parameter_bytes", 2000, [0, 500, 1000, 1500, 2000]),
    ]
    for x, title, key, vmax, ticks in panels:
        svg.text(x, 92, title, size=13, weight="620")
        grouped_panel(
            svg,
            (x, 108, 300, 340),
            MODELS,
            series_from(rows, DTYPES, lambda r, key=key: mib(r[key])),
            vmax,
            ticks,
            "MiB",
            lambda v: f"{v:,.0f}",
            lambda v: f"{v:,.0f}",
            labels=None,
        )
    footer(svg, f"Source: results/metrics/results_full.json  ·  {payload['results'][0]['cuda_device']}")
    save(svg, "memory", "Peak VRAM and weight size", "Image peak VRAM, text peak VRAM, and parameter bytes by model and precision.")


def agreement_figure(payload, rows) -> None:
    svg = Svg(940, 520)
    header(
        svg,
        "Top-1 agreement with the fp32 ranking",
        "Share of 6,156 caption queries whose top gallery image is unchanged",
        [(FP16, "fp16"), (BF16, "bf16")],
    )
    grouped_panel(
        svg,
        (78, 96, 832, 340),
        MODELS,
        series_from(rows, ["fp16", "bf16"], lambda r: r["top1_agreement_pct"]),
        118,
        [0, 25, 50, 75, 100],
        "Top-1 agreement (%)",
        lambda v: f"{v:.0f}",
        lambda v: f"{v:.2f}",
    )
    footer(svg, "fp32 is the reference and is not plotted  ·  Source: results/metrics/results_full.json")
    save(svg, "rank_agreement", "Top-1 agreement with fp32", "Grouped bars of top-1 ranking agreement for fp16 and bf16 against each model's fp32 embeddings.")


def _log_x(value, x, w, lo, hi) -> float:
    return x + (math.log10(value) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * w


def drift_figure(payload, rows) -> None:
    svg = Svg(1040, 450)
    header(svg, "Embedding deviation from fp32", "1 − cosine similarity  ·  log scale  ·  box is p25–p75, bar is the median, whiskers reach min and max")
    lo, hi = 1e-7, 1e-1
    labels = [(model, dtype) for model in MODELS for dtype in ("fp16", "bf16")]
    row_h = 28
    top = 136
    plot_h = row_h * len(labels)
    panels = [(188, 390, "Image embeddings", "image_drift"), (620, 390, "Text embeddings", "text_drift")]
    for px, pw, title, field in panels:
        svg.text(px, 100, title, size=13, weight="620")
        for exp in range(-7, 0):
            xx = _log_x(10**exp, px, pw, lo, hi)
            svg.line(xx, top - 6, xx, top + plot_h, GRID, 1)
            svg.text(xx, top - 14, f"10{superscript(exp)}", size=10, fill=MUTED, anchor="middle", family=MONO)
        svg.line(px, top + plot_h, px + pw, top + plot_h, AXIS, 1.2)
        for i, (model, dtype) in enumerate(labels):
            yy = top + i * row_h + row_h / 2
            if px == panels[0][0]:
                svg.text(28, yy + 4, model, size=11.5, anchor="start", fill=INK)
                svg.text(108, yy + 4, dtype, size=11.5, fill=DTYPE_COLOR[dtype], weight="620")
            stats = rows[(model, dtype)][field]["cosine_percentiles"]
            color = DTYPE_COLOR[dtype]
            p0 = 1 - stats["p0"]
            p25 = 1 - stats["p25"]
            p50 = 1 - stats["p50"]
            p75 = 1 - stats["p75"]
            p100 = 1 - stats["p100"]
            x0 = _log_x(p0, px, pw, lo, hi)
            x1 = _log_x(p100, px, pw, lo, hi)
            svg.line(x1, yy, x0, yy, color, 1.4, opacity=0.45)
            box_a = _log_x(p75, px, pw, lo, hi)
            box_b = _log_x(p25, px, pw, lo, hi)
            left, right = min(box_a, box_b), max(box_a, box_b)
            svg.rect(left, yy - 6, max(right - left, 1.5), 12, color, rx=1, opacity=0.28, stroke=color, sw=1)
            med = _log_x(p50, px, pw, lo, hi)
            svg.line(med, yy - 8, med, yy + 8, color, 2.2)
    footer(svg, "Further right is more drift  ·  Source: results/metrics/results_full.json  ·  every image and caption embedding")
    save(
        svg,
        "embedding_drift",
        "Embedding deviation from fp32",
        "Log-scale percentile ranges of 1 minus cosine similarity against fp32 for image and text embeddings.",
    )


def superscript(exp: int) -> str:
    chars = str(exp).replace("-", "⁻")
    table = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
    return chars.translate(table)


def parse_capacity(text: str) -> tuple[list[dict], float | None]:
    match = re.search(r"\((\d+(?:\.\d+)?)\s*GiB\)", text)
    card = float(match.group(1)) if match else None
    parsed = []
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 6 or not cells[1].isdigit():
            continue
        vram = cells[3]
        parsed.append(
            {
                "model": cells[0],
                "batch": int(cells[1]),
                "ckpt": cells[2],
                "oom": vram == "OOM",
                "vram_gib": None if vram in {"", "OOM"} else float(vram.replace("GiB", "").strip()),
                "step_ms": None if not cells[4] else float(cells[4].replace("ms", "").strip()),
                "epoch_min": None if not cells[5] else float(cells[5].replace("min", "").strip()),
            }
        )
    return parsed, card


def capacity_figure(rows: list[dict], card: float | None) -> None:
    svg = Svg(1040, 560)
    header(svg, "Fine-tune step on one RTX A6000", "Synthetic full step  ·  bf16 autocast, fp32 master weights  ·  median of the last 3 of 5 steps")
    legend(svg, svg.w - 28, 32, [(color, model) for model, color in MODEL_COLOR.items()])
    batches = [32, 64, 128, 256, 512]
    panels = [
        (72, 430, "Peak allocated memory", "GiB", "vram_gib", 56, [0, 12, 24, 36, 48]),
        (590, 400, "Estimated epoch time", "min", "epoch_min", 10, [0, 2, 4, 6, 8, 10]),
    ]
    for x, w, title, unit, key, vmax, ticks in panels:
        svg.text(x, 108, title, size=13, weight="620")
        y, h = 132, 320
        svg.text(x - 40, y + h / 2, unit, size=12, fill=MUTED, anchor="middle", transform=f"rotate(-90 {x - 40:.2f} {y + h / 2:.2f})")
        for tick in ticks:
            yy = y_of(tick, y, h, vmax)
            svg.line(x, yy, x + w, yy, GRID, 1)
            svg.text(x - 8, yy + 4, f"{tick:g}", size=11, fill=MUTED, anchor="end", family=MONO)
        svg.line(x, y + h, x + w, y + h, AXIS, 1.2)
        for i, batch in enumerate(batches):
            xx = x + i * (w / (len(batches) - 1))
            svg.line(xx, y + h, xx, y + h + 5, AXIS, 1)
            svg.text(xx, y + h + 20, str(batch), size=11, fill=MUTED, anchor="middle", family=MONO)
        svg.text(x + w / 2, y + h + 40, "Batch size", size=12, fill=MUTED, anchor="middle")
        if key == "vram_gib" and card is not None:
            yy = y_of(card, y, h, vmax)
            svg.line(x, yy, x + w, yy, "#9aa3b2", 1.2, dash="4 4")
            svg.text(x + w - 4, yy - 6, f"{card:g} GiB", size=11, fill=MUTED, anchor="end", family=MONO)

        def xpos(batch, x=x, w=w):
            return x + batches.index(batch) * (w / (len(batches) - 1))

        drawn = []
        for model, color in MODEL_COLOR.items():
            for ckpt, dash in (("off", None), ("on", "5 4")):
                pts = []
                for row in rows:
                    if row["model"] != model or row["ckpt"] != ckpt or row[key] is None:
                        continue
                    pts.append((xpos(row["batch"]), y_of(row[key], y, h, vmax)))
                pts.sort()
                svg.polyline(pts, color, 2.0, dash=dash)
                drawn.append((ckpt, color, pts))
        for ckpt, color, pts in drawn:
            for px, py in pts:
                if ckpt == "off":
                    svg.circle(px, py, 4.2, color)
                else:
                    svg.circle(px, py, 4.2, PAPER, stroke=color, sw=1.8)
    svg.text(72, 78, "Solid, filled markers: checkpointing off", size=12, fill=MUTED)
    svg.text(360, 78, "Dashed, open markers: checkpointing on", size=12, fill=MUTED)
    footer(svg, "OOM cells are omitted rather than drawn  ·  Source: results/metrics/train_capacity_a6000.md")
    save(svg, "finetune_capacity", "Fine-tune capacity on an RTX A6000", "Peak allocated memory and estimated epoch time against batch size, with and without gradient checkpointing.")


def print_tables(payload, rows) -> None:
    device = payload["results"][0]["cuda_device"]
    torch_version = payload["results"][0]["torch_version"]
    print(f"device={device}")
    print(f"torch={torch_version}")
    print(f"generated_at={payload['generated_at']}")
    print(f"n_images={payload['dataset']['n_images']} n_captions={payload['dataset']['n_captions']} n_identities={payload['dataset']['n_identities']}")
    print()
    print("| Model | Precision | R@1 | Δ R@1 | R@5 | R@10 | mAP |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for model in MODELS:
        ref = rows[(model, "fp32")]
        for dtype in DTYPES:
            row = rows[(model, dtype)]
            if dtype == "fp32":
                delta = "—"
            else:
                delta = f"{row['R@1'] - ref['R@1']:+.2f}"
            print(
                f"| {model} | {dtype} | {row['R@1']:.2f} | {delta} | {row['R@5']:.2f} | "
                f"{row['R@10']:.2f} | {row['mAP']:.2f} |"
            )
    print()
    print("| Model | Precision | Images/s | vs fp32 | Captions/s | Image MiB | Text MiB | Weights MiB |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for model in MODELS:
        ref = rows[(model, "fp32")]
        for dtype in DTYPES:
            row = rows[(model, dtype)]
            speed = row["image_throughput_per_second"] / ref["image_throughput_per_second"]
            print(
                f"| {model} | {dtype} | {row['image_throughput_per_second']:,.0f} | {speed:.2f}× | "
                f"{row['text_throughput_per_second']:,.0f} | {mib(row['image_peak_vram_bytes']):,.0f} | "
                f"{mib(row['text_peak_vram_bytes']):,.0f} | {mib(row['parameter_bytes']):,.0f} |"
            )
    print()
    print("| Model | Precision | Top-1 agree | Top-10 overlap | Kendall τ | τ min | 1−cos image | 1−cos text |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for model in MODELS:
        for dtype in ("fp16", "bf16"):
            row = rows[(model, dtype)]
            print(
                f"| {model} | {dtype} | {row['top1_agreement_pct']:.2f} | {row['top10_overlap_pct']:.2f} | "
                f"{row['kendall_tau_mean']:.4f} | {row['kendall_tau_min']:.4f} | "
                f"{1 - row['image_drift']['cosine_mean']:.2e} | {1 - row['text_drift']['cosine_mean']:.2e} |"
            )
    print()
    deltas = []
    for model in MODELS:
        ref = rows[(model, "fp32")]
        for dtype in ("fp16", "bf16"):
            row = rows[(model, dtype)]
            for metric in ("R@1", "R@5", "R@10", "mAP"):
                deltas.append((abs(row[metric] - ref[metric]), model, dtype, metric, row[metric] - ref[metric], ref[metric], row[metric]))
    deltas.sort(reverse=True)
    for item in deltas[:8]:
        print(f"delta {item[1]} {item[2]} {item[3]} {item[4]:+.4f} ({item[5]:.4f} -> {item[6]:.4f})")
    print()
    for model in MODELS:
        ref = rows[(model, "fp32")]
        for dtype in ("fp16", "bf16"):
            row = rows[(model, dtype)]
            img = row["image_drift"]["cosine_percentiles"]
            print(
                f"drift {model} {dtype} image mean {1 - row['image_drift']['cosine_mean']:.3e} "
                f"p50 {1 - img['p50']:.3e} p0 {1 - img['p0']:.3e}"
            )


def main() -> None:
    payload = load_metrics()
    rows = index_rows(payload)
    expected = {(model, dtype) for model in MODELS for dtype in DTYPES}
    if set(rows) != expected:
        raise SystemExit(f"unexpected model/dtype keys: {sorted(set(rows))}")
    recall_figure(payload, rows)
    throughput_figure(payload, rows)
    memory_figure(payload, rows)
    agreement_figure(payload, rows)
    drift_figure(payload, rows)
    capacity_rows, card = parse_capacity(CAPACITY_PATH.read_text(encoding="utf-8"))
    if len(capacity_rows) != 18:
        raise SystemExit(f"expected 18 capacity rows, found {len(capacity_rows)}")
    capacity_figure(capacity_rows, card)
    print_tables(payload, rows)


if __name__ == "__main__":
    main()
