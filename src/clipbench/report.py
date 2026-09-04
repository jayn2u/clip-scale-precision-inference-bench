"""Self-contained HTML report with inline SVG charts (no external assets)."""
from __future__ import annotations

import html
import json
from typing import Sequence

DTYPE_COLORS = {"fp32": "#3b6ea5", "fp16": "#c96a3d", "bf16": "#4f8a5b"}


def _esc(x) -> str:
    return html.escape(str(x))


def _fmt(x, digits: int = 2, dash: str = "—") -> str:
    if x is None:
        return dash
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def _mib(x) -> str:
    return "—" if x is None else f"{x / 2**20:.0f}"


def _axis_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    if hi <= lo:
        return [lo]
    step = (hi - lo) / n
    return [lo + step * i for i in range(n + 1)]


def grouped_bar_chart(rows: Sequence[dict], models: Sequence[str], dtypes: Sequence[str],
                      metric: str = "R@1") -> str:
    W, H = 760, 340
    pad_l, pad_r, pad_t, pad_b = 56, 16, 16, 56
    plot_w, plot_h = W - pad_l - pad_r, H - pad_t - pad_b
    values = [r[metric] for r in rows if metric in r]
    if not values:
        return ""
    vmax = max(values) * 1.18
    group_w = plot_w / max(len(models), 1)
    bar_w = group_w / (len(dtypes) + 1)

    parts = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{_esc(metric)} by model and precision">']
    for t in _axis_ticks(0, vmax):
        y = pad_t + plot_h - (t / vmax) * plot_h
        parts.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{W-pad_r}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{pad_l-8}" y="{y+4:.1f}" class="tick" text-anchor="end">{t:.1f}</text>')

    for mi, model in enumerate(models):
        gx = pad_l + mi * group_w
        for di, dtype in enumerate(dtypes):
            match = [r for r in rows if r["model"] == model and r["dtype"] == dtype]
            if not match or metric not in match[0]:
                continue
            v = match[0][metric]
            h = (v / vmax) * plot_h
            x = gx + bar_w * (di + 0.5)
            y = pad_t + plot_h - h
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w*0.85:.1f}" height="{h:.1f}" '
                f'fill="{DTYPE_COLORS.get(dtype, "#888")}" rx="2"><title>{_esc(model)} {_esc(dtype)}: '
                f'{v:.2f}</title></rect>'
            )
            parts.append(
                f'<text x="{x + bar_w*0.42:.1f}" y="{y-5:.1f}" class="barlabel" '
                f'text-anchor="middle">{v:.1f}</text>'
            )
        parts.append(
            f'<text x="{gx + group_w/2:.1f}" y="{H-pad_b+22}" class="axis" '
            f'text-anchor="middle">{_esc(model)}</text>'
        )
    parts.append(
        f'<line x1="{pad_l}" y1="{pad_t+plot_h}" x2="{W-pad_r}" y2="{pad_t+plot_h}" class="axisline"/>'
    )
    parts.append(f'<text x="14" y="{pad_t+plot_h/2}" class="axis" transform="rotate(-90 14 {pad_t+plot_h/2})" text-anchor="middle">{_esc(metric)} (%)</text>')
    parts.append("</svg>")
    return "".join(parts)


def scatter_chart(rows: Sequence[dict], x_key: str, y_key: str,
                  x_label: str, y_label: str) -> str:
    pts = [r for r in rows if r.get(x_key) is not None and r.get(y_key) is not None]
    if not pts:
        return ""
    W, H = 760, 380
    pad_l, pad_r, pad_t, pad_b = 62, 100, 18, 52
    plot_w, plot_h = W - pad_l - pad_r, H - pad_t - pad_b
    xs = [r[x_key] for r in pts]
    ys = [r[y_key] for r in pts]
    x_lo, x_hi = 0, max(xs) * 1.12
    y_lo, y_hi = min(ys) * 0.94, max(ys) * 1.06

    parts = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{_esc(y_label)} versus {_esc(x_label)}">']
    for t in _axis_ticks(y_lo, y_hi):
        y = pad_t + plot_h - (t - y_lo) / (y_hi - y_lo) * plot_h
        parts.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{W-pad_r}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{pad_l-8}" y="{y+4:.1f}" class="tick" text-anchor="end">{t:.1f}</text>')
    for t in _axis_ticks(x_lo, x_hi):
        x = pad_l + (t - x_lo) / (x_hi - x_lo) * plot_w
        parts.append(f'<text x="{x:.1f}" y="{H-pad_b+20}" class="tick" text-anchor="middle">{t:.0f}</text>')

    for r in pts:
        x = pad_l + (r[x_key] - x_lo) / (x_hi - x_lo) * plot_w
        y = pad_t + plot_h - (r[y_key] - y_lo) / (y_hi - y_lo) * plot_h
        color = DTYPE_COLORS.get(r["dtype"], "#888")
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}" fill-opacity="0.85" '
            f'stroke="var(--bg)" stroke-width="1.5"><title>{_esc(r["model"])} {_esc(r["dtype"])}: '
            f'{r[x_key]:.1f}, {r[y_key]:.2f}</title></circle>'
        )
        parts.append(
            f'<text x="{x+9:.1f}" y="{y+4:.1f}" class="pointlabel">{_esc(r["model"])}</text>'
        )
    parts.append(f'<line x1="{pad_l}" y1="{pad_t+plot_h}" x2="{W-pad_r}" y2="{pad_t+plot_h}" class="axisline"/>')
    parts.append(f'<text x="{pad_l+plot_w/2:.1f}" y="{H-8}" class="axis" text-anchor="middle">{_esc(x_label)}</text>')
    parts.append(f'<text x="14" y="{pad_t+plot_h/2}" class="axis" transform="rotate(-90 14 {pad_t+plot_h/2})" text-anchor="middle">{_esc(y_label)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def drift_box_chart(rows: Sequence[dict], field: str, title: str) -> str:
    """Drift on a log axis of (1 - cosine).

    Precisions differ in drift by orders of magnitude -- bf16 carries 8 mantissa
    bits against fp16's 10 -- so a linear cosine axis would collapse the smaller
    one to a point. Plotting the deviation from 1 on a log scale keeps both
    legible.
    """
    FLOOR = 1e-9  # exact agreement has no log; pin it to the axis floor

    def dev(v: float) -> float:
        return max(1.0 - v, FLOOR)

    entries = [(r, r[field]["cosine_percentiles"]) for r in rows if r.get(field)]
    if not entries:
        return ""

    import math

    W = 760
    row_h = 42
    H = 52 + row_h * len(entries)
    pad_l, pad_r = 168, 96
    plot_w = W - pad_l - pad_r

    all_dev = [dev(p[k]) for _, p in entries for k in ("p0", "p100")]
    lo_e = math.floor(math.log10(min(all_dev)))
    hi_e = math.ceil(math.log10(max(all_dev)))
    if hi_e - lo_e < 2:
        hi_e = lo_e + 2
    span = hi_e - lo_e

    def px(v: float) -> float:
        return pad_l + (math.log10(dev(v)) - lo_e) / span * plot_w

    parts = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{_esc(title)}">']
    for e in range(lo_e, hi_e + 1):
        x = pad_l + (e - lo_e) / span * plot_w
        parts.append(f'<line x1="{x:.1f}" y1="24" x2="{x:.1f}" y2="{H-22}" class="grid"/>')
        parts.append(f'<text x="{x:.1f}" y="18" class="tick" text-anchor="middle">1e{e}</text>')

    for i, (r, p) in enumerate(entries):
        y = 44 + i * row_h
        color = DTYPE_COLORS.get(r["dtype"], "#888")
        parts.append(
            f'<text x="{pad_l-10}" y="{y+4}" class="rowlabel" text-anchor="end">'
            f'{_esc(r["model"])} &#183; {_esc(r["dtype"])}</text>'
        )
        # whiskers run from the least-drifted vector (p100 cosine) to the most (p0)
        parts.append(
            f'<line x1="{px(p["p100"]):.1f}" y1="{y}" x2="{px(p["p0"]):.1f}" y2="{y}" '
            f'stroke="{color}" stroke-opacity="0.45" stroke-width="2"/>'
        )
        x1, x2 = px(p["p75"]), px(p["p25"])
        parts.append(
            f'<rect x="{min(x1,x2):.1f}" y="{y-8}" width="{max(abs(x2-x1),1.5):.1f}" '
            f'height="16" fill="{color}" fill-opacity="0.35" stroke="{color}" rx="2"/>'
        )
        parts.append(
            f'<line x1="{px(p["p50"]):.1f}" y1="{y-9}" x2="{px(p["p50"]):.1f}" y2="{y+9}" '
            f'stroke="{color}" stroke-width="2.5">'
            f'<title>median 1-cos = {dev(p["p50"]):.2e}</title></line>'
        )
        parts.append(
            f'<text x="{W-pad_r+8}" y="{y+4}" class="tick">worst {dev(p["p0"]):.1e}</text>'
        )
    parts.append(
        f'<text x="{pad_l+plot_w/2:.1f}" y="{H-4}" class="axis" text-anchor="middle">'
        f'1 &minus; cosine similarity vs fp32 (log scale)</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


CSS = """
:root{--bg:#fbfaf8;--fg:#1c1b19;--muted:#6b6862;--line:#e2ded7;--card:#ffffff;--accent:#3b6ea5}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#16171a;--fg:#e8e6e3;--muted:#9a978f;--line:#2c2e33;--card:#1d1f23;--accent:#7aa7d9}}
:root[data-theme="dark"]{--bg:#16171a;--fg:#e8e6e3;--muted:#9a978f;--line:#2c2e33;--card:#1d1f23;--accent:#7aa7d9}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--fg);font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;margin:0;padding:0}
.wrap{max-width:900px;margin:0 auto;padding:48px 24px 96px}
h1{font-size:30px;line-height:1.25;margin:0 0 8px;letter-spacing:-.02em}
h2{font-size:20px;margin:48px 0 6px;letter-spacing:-.01em;padding-top:20px;border-top:1px solid var(--line)}
h3{font-size:15px;margin:28px 0 6px;color:var(--muted);font-weight:600}
p{margin:10px 0}
.sub{color:var(--muted);margin:0 0 4px}
.meta{color:var(--muted);font-size:13px;margin:16px 0 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:18px;margin:18px 0}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
table{border-collapse:collapse;width:100%;font-size:13.5px;min-width:560px}
th,td{text-align:right;padding:7px 10px;border-bottom:1px solid var(--line);white-space:nowrap}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}
thead th{color:var(--muted);font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.04em}
tbody tr:last-child td{border-bottom:none}
tr.ref td{background:color-mix(in srgb,var(--accent) 7%,transparent)}
svg{width:100%;height:auto;display:block}
.grid{stroke:var(--line);stroke-width:1}
.axisline{stroke:var(--muted);stroke-width:1}
text{fill:var(--fg);font-family:inherit}
.tick{font-size:10.5px;fill:var(--muted)}
.axis{font-size:12px;fill:var(--muted)}
.barlabel{font-size:10.5px;fill:var(--muted)}
.pointlabel{font-size:10.5px;fill:var(--muted)}
.rowlabel{font-size:12px;fill:var(--fg)}
.legend{display:flex;gap:18px;flex-wrap:wrap;margin:6px 0 0;font-size:13px;color:var(--muted)}
.legend span{display:inline-flex;align-items:center;gap:6px}
.sw{width:11px;height:11px;border-radius:2px;display:inline-block}
.kv{display:grid;grid-template-columns:auto 1fr;gap:4px 18px;font-size:13.5px}
.kv dt{color:var(--muted)}
.kv dd{margin:0}
code{background:color-mix(in srgb,var(--fg) 7%,transparent);padding:1px 5px;border-radius:4px;font-size:12.5px}
"""


def _legend(dtypes: Sequence[str]) -> str:
    items = "".join(
        f'<span><i class="sw" style="background:{DTYPE_COLORS.get(d, "#888")}"></i>{_esc(d)}</span>'
        for d in dtypes
    )
    return f'<div class="legend">{items}</div>'


def build_report(payload: dict) -> str:
    rows = payload["results"]
    models = list(dict.fromkeys(r["model"] for r in rows))
    dtypes = list(dict.fromkeys(r["dtype"] for r in rows))
    ds = payload["dataset"]
    truncated = next((r.get("captions_truncated") for r in rows if r.get("captions_truncated") is not None), None)

    # ---- retrieval table ----
    ret_rows = "".join(
        f'<tr class="{"ref" if r["dtype"] == payload["config"]["reference_dtype"] else ""}">'
        f'<td>{_esc(r["model"])}</td><td>{_esc(r["dtype"])}</td>'
        f'<td>{_fmt(r.get("R@1"))}</td><td>{_fmt(r.get("R@5"))}</td>'
        f'<td>{_fmt(r.get("R@10"))}</td><td>{_fmt(r.get("mAP"))}</td></tr>'
        for r in rows
    )

    # ---- speed table ----
    speed_rows = "".join(
        f'<tr class="{"ref" if r["dtype"] == payload["config"]["reference_dtype"] else ""}">'
        f'<td>{_esc(r["model"])}</td><td>{_esc(r["dtype"])}</td>'
        f'<td>{_fmt(r.get("image_throughput_per_second"), 1)}</td>'
        f'<td>{_fmt(r.get("text_throughput_per_second"), 1)}</td>'
        f'<td>{_mib(r.get("image_peak_vram_bytes"))}</td>'
        f'<td>{_mib(r.get("parameter_bytes"))}</td></tr>'
        for r in rows
    )

    # ---- agreement table ----
    agree = [r for r in rows if r.get("top1_agreement_pct") is not None]
    agree_rows = "".join(
        f'<tr><td>{_esc(r["model"])}</td><td>{_esc(r["dtype"])}</td>'
        f'<td>{_fmt(r["top1_agreement_pct"])}</td>'
        f'<td>{_fmt(r["top10_overlap_pct"])}</td>'
        f'<td>{_fmt(r["kendall_tau_mean"], 4)}</td>'
        f'<td>{_fmt(r["kendall_tau_min"], 4)}</td>'
        f'<td>{_fmt(r["image_drift"]["cosine_mean"], 5)}</td>'
        f'<td>{_fmt(r["text_drift"]["cosine_mean"], 5)}</td></tr>'
        for r in agree
    )

    bar = grouped_bar_chart(rows, models, dtypes, "R@1")
    bar_map = grouped_bar_chart(rows, models, dtypes, "mAP")
    scat = scatter_chart(
        rows, "image_throughput_per_second", "R@1",
        "Image encoding throughput (images/s, fixed batch)", "R@1 (%)",
    )
    box_img = drift_box_chart(agree, "image_drift", "Image embedding cosine vs fp32")
    box_txt = drift_box_chart(agree, "text_drift", "Text embedding cosine vs fp32")

    return f"""<title>CLIP Precision Retrieval Bench</title>
<style>{CSS}</style>
<div class="wrap">
<h1>CLIP precision &amp; scale on CUHK-PEDES text&#8594;image retrieval</h1>
<p class="sub">Zero-shot openai/CLIP across {len(models)} model scales &#215; {len(dtypes)} weight precisions, measured on accuracy, speed, memory, and numerical drift.</p>

<div class="card">
<dl class="kv">
<dt>Dataset</dt><dd>{_esc(ds["name"])} <code>{_esc(ds["split"])}</code> split &mdash; {ds["n_images"]:,} gallery images, {ds["n_captions"]:,} caption queries, {ds["n_identities"]:,} identities</dd>
<dt>Protocol</dt><dd>Every caption queries the full gallery; a hit is a person-<code>id</code> match, so other images of the same identity count.</dd>
<dt>Precision</dt><dd>True weight casting via openai's <code>convert_weights</code> layer policy, generalised over dtype &mdash; not autocast. LayerNorm stays fp32 in every config, as upstream leaves it.</dd>
<dt>Timing</dt><dd>CUDA-synchronised forward pass only, {payload["config"]["warmup_batches"]} warmup batches, median over full batches. Batch size fixed at {payload["config"]["image_batch_size"]} (image) / {payload["config"]["text_batch_size"]} (text) across all precisions.</dd>
<dt>Truncation</dt><dd>{"—" if truncated is None else f"{truncated:,} of {ds['n_captions']:,} captions"} exceed CLIP's 77-token context and are truncated &mdash; identically in every config.</dd>
<dt>Generated</dt><dd>{_esc(payload["generated_at"])} &middot; tag <code>{_esc(payload["tag"])}</code></dd>
</dl>
</div>

<h2>Does precision cost accuracy?</h2>
<p>Recall@1 per model scale and weight precision. The fp32 column is the reference.</p>
{_legend(dtypes)}
<div class="card">{bar}</div>
<p>mAP is the more sensitive instrument &mdash; identities carry several gallery images, so it registers rank shifts that R@1 cannot see.</p>
<div class="card">{bar_map}</div>

<div class="scroll card"><table>
<thead><tr><th>Model</th><th>Precision</th><th>R@1</th><th>R@5</th><th>R@10</th><th>mAP</th></tr></thead>
<tbody>{ret_rows}</tbody></table></div>

<h2>What does precision buy?</h2>
<p>Accuracy against image-encoding throughput. Points toward the upper right dominate; a vertical cluster means precision bought speed for free.</p>
{_legend(dtypes)}
<div class="card">{scat}</div>

<div class="scroll card"><table>
<thead><tr><th>Model</th><th>Precision</th><th>Images/s</th><th>Captions/s</th><th>Peak VRAM (MiB)</th><th>Weights (MiB)</th></tr></thead>
<tbody>{speed_rows}</tbody></table></div>

<h2>Did the embeddings actually move?</h2>
<p>Deviation of each embedding from its fp32 counterpart, as <code>1 &minus; cosine</code> on a log axis &mdash; the precisions differ by orders of magnitude, so a linear cosine axis would collapse the smaller one to a point. Box spans p25&ndash;p75, line is the median, whiskers reach the extremes. Further right is more drift.</p>
<h3>Image embeddings</h3>
<div class="card">{box_img}</div>
<h3>Text embeddings</h3>
<div class="card">{box_txt}</div>

<h2>Did the ranking move?</h2>
<p>Cosine drift says how far the vectors travelled; these columns say whether the user would notice. Top-1 agreement is the share of caption queries that still retrieve the same top image as fp32.</p>
<div class="scroll card"><table>
<thead><tr><th>Model</th><th>Precision</th><th>Top-1 agree (%)</th><th>Top-10 overlap (%)</th><th>Kendall &tau; mean</th><th>&tau; min</th><th>Img cos</th><th>Txt cos</th></tr></thead>
<tbody>{agree_rows}</tbody></table></div>

<p class="meta">Kendall &tau; is computed over the full gallery ranking for a fixed random sample of {_esc(next((r.get("kendall_tau_n_queries") for r in agree), "—"))} queries. Absolute recall is low by design: this is zero-shot CLIP with no CUHK-PEDES fine-tuning, so the numbers sit far below fine-tuned SOTA. The comparison axis &mdash; precision against precision, within a model &mdash; is unaffected.</p>
</div>
"""
