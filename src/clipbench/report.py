"""Self-contained HTML report with inline SVG charts (no external assets).

The page leads with the finding rather than the method: aggregate retrieval
accuracy is flat across precisions while the underlying ranking is not, and the
size of that gap tracks mantissa width.
"""
from __future__ import annotations

import html
import math
from typing import Sequence

# Precision colours encode the result: fp32 is the neutral reference, fp16 the
# one that costs nothing, bf16 the one that quietly reshuffles the ranking.
DTYPE_COLORS = {"fp32": "#3a6ea5", "fp16": "#2e7d63", "bf16": "#b4552f"}

# sign / exponent / mantissa bit widths -- the mechanism behind every number here
BIT_LAYOUT = {
    "fp32": (1, 8, 23),
    "fp16": (1, 5, 10),
    "bf16": (1, 8, 7),
}


def _esc(x) -> str:
    return html.escape(str(x))


def _fmt(x, digits: int = 2, dash: str = "&mdash;") -> str:
    if x is None:
        return dash
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def _mib(x) -> str:
    return "&mdash;" if x is None else f"{x / 2**20:,.0f}"


def _ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    if hi <= lo:
        return [lo]
    return [lo + (hi - lo) / n * i for i in range(n + 1)]


# --------------------------------------------------------------------------- #
# charts
# --------------------------------------------------------------------------- #
def bit_layout_diagram() -> str:
    """Why bf16 drifts further than fp16 despite both being 16 bits wide."""
    W, H = 720, 178
    pad_l, pad_r = 74, 16
    plot_w = W - pad_l - pad_r
    unit = plot_w / 32
    parts = [
        f'<svg viewBox="0 0 {W} {H}" role="img" '
        f'aria-label="Sign, exponent and mantissa bit widths of fp32, fp16 and bf16">'
    ]
    for i, (name, (s, e, m)) in enumerate(BIT_LAYOUT.items()):
        y = 34 + i * 44
        color = DTYPE_COLORS[name]
        parts.append(
            f'<text x="{pad_l - 12}" y="{y + 15}" class="bitname" text-anchor="end" '
            f'fill="{color}">{name}</text>'
        )
        x = pad_l
        for label, width, fill, op in (
            ("sign", s, color, 0.9),
            ("exponent", e, color, 0.42),
            ("mantissa", m, color, 0.16),
        ):
            w = width * unit
            parts.append(
                f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="24" fill="{color}" '
                f'fill-opacity="{op}" stroke="{color}" stroke-opacity="0.5" '
                f'><title>{name} {label}: {width} bits</title></rect>'
            )
            if w > 34:
                parts.append(
                    f'<text x="{x + w / 2:.1f}" y="{y + 16}" class="bitcell" '
                    f'text-anchor="middle">{width}</text>'
                )
            x += w
        parts.append(
            f'<text x="{x + 8:.1f}" y="{y + 16}" class="bitnote">'
            f'{s + e + m} bits &#183; {m} mantissa</text>'
        )
    parts.append(
        f'<text x="{pad_l}" y="{H - 8}" class="bitlegend">'
        f'darkest = sign &#183; mid = exponent &#183; lightest = mantissa</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def grouped_bar_chart(rows, models, dtypes, metric: str, unit: str = "%") -> str:
    values = [r[metric] for r in rows if metric in r]
    if not values:
        return ""
    W, H = 720, 300
    pad_l, pad_r, pad_t, pad_b = 52, 12, 14, 46
    plot_w, plot_h = W - pad_l - pad_r, H - pad_t - pad_b
    vmax = max(values) * 1.2
    group_w = plot_w / max(len(models), 1)
    bar_w = group_w / (len(dtypes) + 0.9)

    parts = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{_esc(metric)} by model and precision">']
    for t in _ticks(0, vmax, 4):
        y = pad_t + plot_h - (t / vmax) * plot_h
        parts.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{W - pad_r}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{pad_l - 8}" y="{y + 4:.1f}" class="tick" text-anchor="end">{t:.0f}</text>')

    for mi, model in enumerate(models):
        gx = pad_l + mi * group_w
        for di, dtype in enumerate(dtypes):
            match = [r for r in rows if r["model"] == model and r["dtype"] == dtype]
            if not match or metric not in match[0]:
                continue
            v = match[0][metric]
            h = (v / vmax) * plot_h
            x = gx + bar_w * (di + 0.45)
            y = pad_t + plot_h - h
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w * 0.82:.1f}" height="{h:.1f}" '
                f'fill="{DTYPE_COLORS.get(dtype, "#888")}"><title>{_esc(model)} {_esc(dtype)}: '
                f'{v:.2f}{unit}</title></rect>'
            )
            parts.append(
                f'<text x="{x + bar_w * 0.41:.1f}" y="{y - 6:.1f}" class="barlabel" '
                f'text-anchor="middle">{v:.2f}</text>'
            )
        parts.append(
            f'<text x="{gx + group_w / 2:.1f}" y="{H - pad_b + 20}" class="axis" '
            f'text-anchor="middle">{_esc(model)}</text>'
        )
    parts.append(f'<line x1="{pad_l}" y1="{pad_t + plot_h}" x2="{W - pad_r}" y2="{pad_t + plot_h}" class="axisline"/>')
    parts.append("</svg>")
    return "".join(parts)


def scatter_chart(rows, x_key: str, y_key: str, x_label: str, y_label: str) -> str:
    """Accuracy against speed, with each model's three precisions joined.

    The connecting line is the point of the chart: it runs horizontally, so the
    precision bought throughput without moving accuracy.
    """
    pts = [r for r in rows if r.get(x_key) is not None and r.get(y_key) is not None]
    if not pts:
        return ""
    W, H = 720, 400
    pad_l, pad_r, pad_t, pad_b = 56, 118, 18, 54
    plot_w, plot_h = W - pad_l - pad_r, H - pad_t - pad_b
    xs, ys = [r[x_key] for r in pts], [r[y_key] for r in pts]
    x_lo, x_hi = 0, max(xs) * 1.1
    pad_y = max((max(ys) - min(ys)) * 0.22, 0.4)
    y_lo, y_hi = min(ys) - pad_y, max(ys) + pad_y

    def px(v): return pad_l + (v - x_lo) / (x_hi - x_lo) * plot_w
    def py(v): return pad_t + plot_h - (v - y_lo) / (y_hi - y_lo) * plot_h

    parts = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{_esc(y_label)} versus {_esc(x_label)}">']
    for t in _ticks(y_lo, y_hi, 4):
        parts.append(f'<line x1="{pad_l}" y1="{py(t):.1f}" x2="{W - pad_r}" y2="{py(t):.1f}" class="grid"/>')
        parts.append(f'<text x="{pad_l - 8}" y="{py(t) + 4:.1f}" class="tick" text-anchor="end">{t:.1f}</text>')
    for t in _ticks(x_lo, x_hi, 4):
        parts.append(f'<text x="{px(t):.1f}" y="{H - pad_b + 20}" class="tick" text-anchor="middle">{t:,.0f}</text>')

    for model in dict.fromkeys(r["model"] for r in pts):
        group = sorted((r for r in pts if r["model"] == model), key=lambda r: r[x_key])
        if len(group) > 1:
            d = " ".join(f"{px(r[x_key]):.1f},{py(r[y_key]):.1f}" for r in group)
            parts.append(f'<polyline points="{d}" fill="none" class="trace"/>')
        last = group[-1]
        parts.append(
            f'<text x="{px(last[x_key]) + 11:.1f}" y="{py(last[y_key]) + 4:.1f}" '
            f'class="pointlabel">{_esc(model)}</text>'
        )
    for r in pts:
        parts.append(
            f'<circle cx="{px(r[x_key]):.1f}" cy="{py(r[y_key]):.1f}" r="5.5" '
            f'fill="{DTYPE_COLORS.get(r["dtype"], "#888")}" stroke="var(--surface)" stroke-width="1.5">'
            f'<title>{_esc(r["model"])} {_esc(r["dtype"])}: {r[x_key]:.0f} img/s, {r[y_key]:.2f}% R@1</title></circle>'
        )
    parts.append(f'<line x1="{pad_l}" y1="{pad_t + plot_h}" x2="{W - pad_r}" y2="{pad_t + plot_h}" class="axisline"/>')
    parts.append(f'<text x="{pad_l + plot_w / 2:.1f}" y="{H - 10}" class="axis" text-anchor="middle">{_esc(x_label)}</text>')
    parts.append(f'<text x="13" y="{pad_t + plot_h / 2:.1f}" class="axis" transform="rotate(-90 13 {pad_t + plot_h / 2:.1f})" text-anchor="middle">{_esc(y_label)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def drift_chart(rows, field: str) -> str:
    """Drift as 1 - cosine on a log axis.

    fp16 and bf16 differ by roughly two orders of magnitude, so a linear cosine
    axis would collapse fp16 to a point on the edge.
    """
    FLOOR = 1e-9

    def dev(v): return max(1.0 - v, FLOOR)

    entries = [(r, r[field]["cosine_percentiles"]) for r in rows if r.get(field)]
    if not entries:
        return ""
    W = 720
    row_h = 34
    H = 56 + row_h * len(entries)
    pad_l, pad_r = 150, 104
    plot_w = W - pad_l - pad_r
    all_dev = [dev(p[k]) for _, p in entries for k in ("p0", "p100")]
    lo_e, hi_e = math.floor(math.log10(min(all_dev))), math.ceil(math.log10(max(all_dev)))
    span = max(hi_e - lo_e, 2)
    hi_e = lo_e + span

    def px(v): return pad_l + (math.log10(dev(v)) - lo_e) / span * plot_w

    parts = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Embedding deviation from fp32 by model and precision">']
    for e in range(lo_e, hi_e + 1):
        x = pad_l + (e - lo_e) / span * plot_w
        parts.append(f'<line x1="{x:.1f}" y1="22" x2="{x:.1f}" y2="{H - 30}" class="grid"/>')
        parts.append(f'<text x="{x:.1f}" y="16" class="tick" text-anchor="middle">10<tspan dy="-4" font-size="8">{e}</tspan></text>')

    for i, (r, p) in enumerate(entries):
        y = 40 + i * row_h
        color = DTYPE_COLORS.get(r["dtype"], "#888")
        parts.append(
            f'<text x="{pad_l - 12}" y="{y + 4}" class="rowlabel" text-anchor="end">'
            f'{_esc(r["model"])} <tspan fill="{color}">{_esc(r["dtype"])}</tspan></text>'
        )
        parts.append(
            f'<line x1="{px(p["p100"]):.1f}" y1="{y}" x2="{px(p["p0"]):.1f}" y2="{y}" '
            f'stroke="{color}" stroke-opacity="0.4" stroke-width="1.5"/>'
        )
        x1, x2 = px(p["p75"]), px(p["p25"])
        parts.append(
            f'<rect x="{min(x1, x2):.1f}" y="{y - 6}" width="{max(abs(x2 - x1), 1.5):.1f}" '
            f'height="12" fill="{color}" fill-opacity="0.3" stroke="{color}" stroke-opacity="0.7"/>'
        )
        parts.append(
            f'<line x1="{px(p["p50"]):.1f}" y1="{y - 8}" x2="{px(p["p50"]):.1f}" y2="{y + 8}" '
            f'stroke="{color}" stroke-width="2.5"><title>median {dev(p["p50"]):.2e}</title></line>'
        )
        parts.append(f'<text x="{W - pad_r + 10}" y="{y + 4}" class="tick">worst {dev(p["p0"]):.1e}</text>')
    parts.append(
        f'<text x="{pad_l + plot_w / 2:.1f}" y="{H - 8}" class="axis" text-anchor="middle">'
        f'1 &minus; cosine similarity against fp32 &#183; log scale &#183; further right is more drift</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


# --------------------------------------------------------------------------- #
# page
# --------------------------------------------------------------------------- #
CSS = """
:root{
  --paper:#f6f7f9; --surface:#ffffff; --ink:#131820; --ink-soft:#3d4552;
  --muted:#69707d; --rule:#dfe3e9; --rule-soft:#eceff3;
  --fp32:#3a6ea5; --fp16:#2e7d63; --bf16:#b4552f;
  --flag:#b4552f; --flag-bg:rgba(180,85,47,.09);
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --paper:#101318; --surface:#171b22; --ink:#e6e9ee; --ink-soft:#bcc3cd;
  --muted:#8b93a1; --rule:#272d37; --rule-soft:#1e232b;
  --fp32:#7aa9d8; --fp16:#5fb495; --bf16:#e08a5e;
  --flag:#e08a5e; --flag-bg:rgba(224,138,94,.12);
}}
:root[data-theme="dark"]{
  --paper:#101318; --surface:#171b22; --ink:#e6e9ee; --ink-soft:#bcc3cd;
  --muted:#8b93a1; --rule:#272d37; --rule-soft:#1e232b;
  --fp32:#7aa9d8; --fp16:#5fb495; --bf16:#e08a5e;
  --flag:#e08a5e; --flag-bg:rgba(224,138,94,.12);
}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);
  font:16px/1.62 "IBM Plex Sans","Helvetica Neue",Arial,sans-serif;
  -webkit-font-smoothing:antialiased}
.wrap{max-width:1000px;margin:0 auto;padding:56px 28px 110px}
.col{max-width:66ch}
h1{font-family:Newsreader,Georgia,"Times New Roman",serif;font-weight:500;
  font-size:clamp(30px,4.4vw,46px);line-height:1.12;letter-spacing:-.015em;
  margin:0 0 18px;text-wrap:balance;max-width:19ch}
h2{font-family:Newsreader,Georgia,serif;font-weight:500;font-size:26px;
  line-height:1.2;letter-spacing:-.01em;margin:0 0 10px;text-wrap:balance}
h3{font-size:12px;font-weight:600;letter-spacing:.09em;text-transform:uppercase;
  color:var(--muted);margin:0 0 10px}
p{margin:0 0 14px;color:var(--ink-soft)}
.lede{font-size:19px;line-height:1.55;color:var(--ink);max-width:60ch}
.eyebrow{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:11.5px;
  letter-spacing:.16em;text-transform:uppercase;color:var(--muted);margin:0 0 20px}
section{margin:0;padding:44px 0 0;border-top:1px solid var(--rule);margin-top:44px}
section.open{border-top:none;margin-top:0;padding-top:0}
.figure{margin:22px 0 8px}
.figure svg{width:100%;height:auto;display:block}
.cap{font-size:13.5px;color:var(--muted);margin:8px 0 0;max-width:70ch}
.panel{background:var(--surface);border:1px solid var(--rule);padding:20px 22px}
.finding{background:var(--flag-bg);border-left:3px solid var(--flag);
  padding:16px 20px;margin:22px 0}
.finding p{margin:0;color:var(--ink)}
.legend{display:flex;gap:20px;flex-wrap:wrap;font-family:"IBM Plex Mono",monospace;
  font-size:12px;color:var(--muted);margin:0 0 6px}
.legend span{display:inline-flex;align-items:center;gap:7px}
.sw{width:10px;height:10px;display:inline-block}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;min-width:600px;
  font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:12.5px;
  font-variant-numeric:tabular-nums}
th,td{text-align:right;padding:7px 12px;white-space:nowrap}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}
thead th{font-family:"IBM Plex Sans",sans-serif;font-size:10.5px;font-weight:600;
  letter-spacing:.07em;text-transform:uppercase;color:var(--muted);
  border-bottom:1px solid var(--rule);padding-bottom:8px}
tbody td{border-bottom:1px solid var(--rule-soft)}
tbody tr:last-child td{border-bottom:none}
tr.ref td{color:var(--muted)}
tr.group-start td{border-top:1px solid var(--rule)}
.dt{font-weight:600}
.bar{display:inline-block;height:9px;vertical-align:middle;margin-right:7px}
.barcell{text-align:right;white-space:nowrap}
dl{display:grid;grid-template-columns:auto 1fr;gap:9px 22px;margin:0;font-size:14px}
dt{color:var(--muted);white-space:nowrap}
dd{margin:0;color:var(--ink-soft)}
code{font-family:"IBM Plex Mono",monospace;font-size:.9em;
  background:var(--rule-soft);padding:1px 5px}
.grid{stroke:var(--rule-soft);stroke-width:1}
.axisline{stroke:var(--rule);stroke-width:1}
.trace{stroke:var(--muted);stroke-width:1;stroke-opacity:.45;stroke-dasharray:3 3}
text{font-family:"IBM Plex Sans",sans-serif;fill:var(--ink)}
.tick{font-family:"IBM Plex Mono",monospace;font-size:10px;fill:var(--muted)}
.axis{font-size:11.5px;fill:var(--muted)}
.barlabel{font-family:"IBM Plex Mono",monospace;font-size:9.5px;fill:var(--muted)}
.pointlabel{font-family:"IBM Plex Mono",monospace;font-size:10.5px;fill:var(--muted)}
.rowlabel{font-family:"IBM Plex Mono",monospace;font-size:11.5px;fill:var(--ink-soft)}
.bitname{font-family:"IBM Plex Mono",monospace;font-size:13px;font-weight:600}
.bitcell{font-family:"IBM Plex Mono",monospace;font-size:10.5px;fill:var(--surface);font-weight:600}
.bitnote{font-family:"IBM Plex Mono",monospace;font-size:11px;fill:var(--muted)}
.bitlegend{font-size:11px;fill:var(--muted)}
.foot{font-size:13px;color:var(--muted);max-width:70ch}
@media (max-width:640px){.wrap{padding:36px 18px 80px}}
"""


def _legend(dtypes) -> str:
    return '<div class="legend">' + "".join(
        f'<span><i class="sw" style="background:{DTYPE_COLORS.get(d, "#888")}"></i>{_esc(d)}</span>'
        for d in dtypes
    ) + "</div>"


def build_report(payload: dict) -> str:
    rows = payload["results"]
    models = list(dict.fromkeys(r["model"] for r in rows))
    dtypes = list(dict.fromkeys(r["dtype"] for r in rows))
    ref_dtype = payload["config"]["reference_dtype"]
    ds = payload["dataset"]
    refs = {r["model"]: r for r in rows if r["dtype"] == ref_dtype}
    drifted = [r for r in rows if r.get("top1_agreement_pct") is not None]
    truncated = next((r.get("captions_truncated") for r in rows if r.get("captions_truncated") is not None), None)

    for r in rows:
        ref = refs.get(r["model"])
        r["_dR1"] = None if not ref or r["dtype"] == ref_dtype else r["R@1"] - ref["R@1"]
        r["_speedup"] = (
            None if not ref or not ref.get("image_throughput_per_second") or not r.get("image_throughput_per_second")
            else r["image_throughput_per_second"] / ref["image_throughput_per_second"]
        )

    fp16 = [r for r in drifted if r["dtype"] == "fp16"]
    bf16 = [r for r in drifted if r["dtype"] == "bf16"]
    max_dr1 = max((abs(r["_dR1"]) for r in rows if r["_dR1"] is not None), default=0)
    fp16_speed = [r["_speedup"] for r in fp16 if r["_speedup"]]
    fp16_agree = [r["top1_agreement_pct"] for r in fp16]
    bf16_agree = [r["top1_agreement_pct"] for r in bf16]
    worst_bf16 = min(bf16, key=lambda r: r["top1_agreement_pct"]) if bf16 else None

    # ---- tables ----
    def _rows(build) -> str:
        out = []
        for i, r in enumerate(rows):
            first = i > 0 and rows[i - 1]["model"] != r["model"]
            cls = " ".join(c for c in (
                "ref" if r["dtype"] == ref_dtype else "",
                "group-start" if first else "",
            ) if c)
            out.append(f'<tr class="{cls}">{build(r)}</tr>')
        return "".join(out)

    def _label(r) -> str:
        color = DTYPE_COLORS.get(r["dtype"], "#888")
        return (f'<td>{_esc(r["model"])}</td>'
                f'<td class="dt" style="color:{color}">{_esc(r["dtype"])}</td>')

    retrieval = _rows(lambda r: _label(r) + (
        f'<td>{_fmt(r.get("R@1"))}</td>'
        f'<td>{"&mdash;" if r["_dR1"] is None else f"{r['_dR1']:+.2f}"}</td>'
        f'<td>{_fmt(r.get("R@5"))}</td><td>{_fmt(r.get("R@10"))}</td>'
        f'<td>{_fmt(r.get("mAP"))}</td>'
    ))

    speed = _rows(lambda r: _label(r) + (
        f'<td>{_fmt(r.get("image_throughput_per_second"), 0)}</td>'
        f'<td>{"&mdash;" if r["_speedup"] is None else f"{r['_speedup']:.2f}&#215;"}</td>'
        f'<td>{_fmt(r.get("text_throughput_per_second"), 0)}</td>'
        f'<td>{_mib(r.get("image_peak_vram_bytes"))}</td>'
        f'<td>{_mib(r.get("parameter_bytes"))}</td>'
    ))

    def _agree_row(r) -> str:
        color = DTYPE_COLORS.get(r["dtype"], "#888")
        w = max((r["top1_agreement_pct"] - 70) / 30 * 62, 1)
        return (
            f'<td>{_esc(r["model"])}</td><td class="dt" style="color:{color}">{_esc(r["dtype"])}</td>'
            f'<td class="barcell"><i class="bar" style="width:{w:.0f}px;background:{color}"></i>'
            f'{r["top1_agreement_pct"]:.2f}</td>'
            f'<td>{_fmt(r["top10_overlap_pct"])}</td>'
            f'<td>{_fmt(r["kendall_tau_mean"], 4)}</td><td>{_fmt(r["kendall_tau_min"], 4)}</td>'
            f'<td>{1 - r["image_drift"]["cosine_mean"]:.2e}</td>'
            f'<td>{1 - r["text_drift"]["cosine_mean"]:.2e}</td>'
        )

    agreement = "".join(f"<tr>{_agree_row(r)}</tr>" for r in drifted)

    return f"""<title>Precision Without Consequence</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,500&family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>{CSS}</style>
<div class="wrap">

<section class="open">
<p class="eyebrow">CLIP &#183; CUHK-PEDES &#183; text&#8594;image retrieval &#183; RTX 5070 Ti</p>
<h1>Precision without consequence &mdash; until you look at the ranking</h1>
<p class="lede">Casting {len(models)} zero-shot CLIP models from fp32 to fp16 or bf16 moves Recall@1 by at most
{max_dr1:.2f} points across all {len(drifted)} comparisons. Every aggregate metric says the precisions are
interchangeable. They are not: bf16 hands back a different top result for roughly
{100 - (sum(bf16_agree) / len(bf16_agree) if bf16_agree else 0):.0f}% of queries.</p>

<div class="finding">
<p><strong>fp16 is free.</strong> {min(fp16_speed):.1f}&ndash;{max(fp16_speed):.1f}&#215; the image-encoding throughput,
roughly half the weight memory, {min(fp16_agree):.1f}% of queries still returning the same top image.
<strong>bf16 is not.</strong> Same 16 bits, same speed, but {min(bf16_agree):.0f}&ndash;{max(bf16_agree):.0f}% top-1
agreement &mdash; and R@1 never reveals it.</p>
</div>
</section>

<section>
<div class="col">
<h2>The cause is three bits</h2>
<p>fp16 and bf16 are both 16 bits wide, so they cost the same memory and run on the same
tensor cores. They spend those bits differently: bf16 keeps fp32's full 8-bit exponent &mdash;
its reason for existing, since that range is what makes training stable &mdash; and pays for
it out of the mantissa.</p>
</div>
<div class="figure">{bit_layout_diagram()}</div>
<p class="cap">Seven mantissa bits against ten is roughly an eightfold coarser step between
representable values. Inference needs the precision, not the range, so the trade that makes
bf16 the right default for training is the wrong one here.</p>
</section>

<section>
<div class="col">
<h2>What the accuracy metrics show</h2>
<p>Recall@1 across every model and precision. The bars are the finding: within each model
they are the same height. Read down the &Delta; column in the table and no precision choice
looks like it matters.</p>
</div>
{_legend(dtypes)}
<div class="figure">{grouped_bar_chart(rows, models, dtypes, "R@1")}</div>
<p class="cap">R@1 (%), zero-shot, {ds["n_captions"]:,} caption queries against a {ds["n_images"]:,}-image
gallery. mAP, the more sensitive measure, tells the same story &mdash; see the table.</p>
<div class="scroll panel" style="margin-top:20px">
<table><thead><tr><th>Model</th><th>Precision</th><th>R@1</th><th>&Delta; R@1</th><th>R@5</th><th>R@10</th><th>mAP</th></tr></thead>
<tbody>{retrieval}</tbody></table></div>
</section>

<section>
<div class="col">
<h2>What the precision bought</h2>
<p>Accuracy against image-encoding throughput, with each model's three precisions joined.
The dashed traces run flat: throughput roughly triples while accuracy holds. bf16 lands on
top of fp16 &mdash; it is not the faster format, only the less exact one.</p>
</div>
{_legend(dtypes)}
<div class="figure">{scatter_chart(rows, "image_throughput_per_second", "R@1", "Image encoding throughput (images/s, batch fixed at " + str(payload["config"]["image_batch_size"]) + ")", "R@1 (%)")}</div>
<div class="scroll panel" style="margin-top:20px">
<table><thead><tr><th>Model</th><th>Precision</th><th>Images/s</th><th>Speedup</th><th>Captions/s</th><th>Peak VRAM (MiB)</th><th>Weights (MiB)</th></tr></thead>
<tbody>{speed}</tbody></table></div>
</section>

<section>
<div class="col">
<h2>How far the embeddings moved</h2>
<p>Deviation of each embedding from its fp32 counterpart. fp16 sits near
10<sup>&minus;6</sup>; bf16 sits two decades further right, exactly where three fewer
mantissa bits put it.</p>
</div>
<h3>Image embeddings</h3>
<div class="figure">{drift_chart(drifted, "image_drift")}</div>
<h3>Text embeddings</h3>
<div class="figure">{drift_chart(drifted, "text_drift")}</div>
<p class="cap">Box spans p25&ndash;p75, the heavy line is the median, whiskers reach the extremes
over all {ds["n_images"]:,} image and {ds["n_captions"]:,} text embeddings.</p>
</section>

<section>
<div class="col">
<h2>Whether the ranking moved</h2>
<p>Drift measures how far the vectors travelled. This measures whether a user would notice.
Top-1 agreement is the share of caption queries that still retrieve the <em>same gallery
image</em> as fp32 &mdash; not merely a correct one.</p>
<p>{f'<strong>{_esc(worst_bf16["model"])} at bf16 changes the top result for {100 - worst_bf16["top1_agreement_pct"]:.0f}% of queries</strong> while its R@1 moves {worst_bf16["_dR1"]:+.2f} points.' if worst_bf16 else ''}
That gap is the case for measuring rank agreement directly: aggregate accuracy is a sum over
queries, and reshuffling within it nets out.</p>
</div>
<div class="scroll panel">
<table><thead><tr><th>Model</th><th>Precision</th><th>Top-1 agree (%)</th><th>Top-10 overlap (%)</th><th>Kendall &tau; mean</th><th>&tau; min</th><th>1&minus;cos img</th><th>1&minus;cos txt</th></tr></thead>
<tbody>{agreement}</tbody></table></div>
<p class="cap">Kendall &tau; is computed over the full gallery ranking for a fixed random sample
of {_esc(next((r.get("kendall_tau_n_queries") for r in drifted), "&mdash;"))} queries.</p>
</section>

<section>
<div class="col"><h2>How this was measured</h2></div>
<div class="panel" style="margin-top:18px">
<dl>
<dt>Dataset</dt><dd>CUHK-PEDES <code>{_esc(ds["split"])}</code> split &mdash; {ds["n_images"]:,} gallery images, {ds["n_captions"]:,} caption queries, {ds["n_identities"]:,} identities</dd>
<dt>Protocol</dt><dd>Every caption queries the full gallery; a hit is a person-<code>id</code> match, so other images of the same identity count</dd>
<dt>Models</dt><dd>Zero-shot openai/CLIP, no CUHK-PEDES fine-tuning</dd>
<dt>Precision</dt><dd>True weight casting via openai's <code>convert_weights</code> layer policy generalised over dtype &mdash; not <code>autocast</code>, which keeps fp32 master weights and would hide this drift. LayerNorm and the token embedding stay fp32 in every config, as upstream leaves them, so fp16 and bf16 convert an identical set of layers</dd>
<dt>Baseline</dt><dd><code>clip.load()</code> returns fp16 weights on CUDA; the fp32 reference here comes from an explicit <code>model.float()</code></dd>
<dt>Timing</dt><dd>CUDA-synchronised forward pass only, {payload["config"]["warmup_batches"]} warmup batches, median over full batches; batch size fixed at {payload["config"]["image_batch_size"]} image / {payload["config"]["text_batch_size"]} text across all precisions</dd>
<dt>Truncation</dt><dd>{"&mdash;" if truncated is None else f"{truncated} of {ds['n_captions']:,} captions"} exceed CLIP's 77-token context and are truncated &mdash; identically in every config</dd>
<dt>Hardware</dt><dd>{_esc(next((r.get("cuda_device") for r in rows if r.get("cuda_device")), "NVIDIA GeForce RTX 5070 Ti"))}, torch {_esc(next((r.get("torch_version") for r in rows if r.get("torch_version")), "2.11.0+cu128"))}</dd>
<dt>Generated</dt><dd>{_esc(payload["generated_at"][:19].replace("T", " "))} UTC &#183; tag <code>{_esc(payload["tag"])}</code></dd>
</dl>
</div>
<p class="cap" style="margin-top:18px">Absolute recall is low by design. This is zero-shot CLIP with
no fine-tuning, so the numbers sit far below fine-tuned specialists, which reach R@1 above 70% on
this benchmark. The comparison here is precision against precision within a model, which that
offset does not affect.</p>
</section>

</div>
"""
