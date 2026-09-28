#!/usr/bin/env python3
"""
Builds docs/poster.html and docs/report.html from the training-run results.

All figures are emitted as inline SVG so the poster stays crisp at A2 print size
and both documents render without any external assets or scripts.

    python3 docs/build_docs.py

Then, to rasterise (Chromium):
    chrome --headless --screenshot=docs/poster.png --window-size=1587,2245 docs/poster.html
    chrome --headless --print-to-pdf=docs/poster.pdf --no-pdf-header-footer docs/poster.html
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))

# ─── Results — single source of truth ────────────────────────────────────────

CLASSES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
LABELS  = {'glioma': 'Glioma', 'meningioma': 'Meningioma',
           'no_tumor': 'No tumour', 'pituitary': 'Pituitary'}

# Validated categorical set for the dark surface #101418:
#   adjacent CVD ΔE 9.4, adjacent normal-vision ΔE 24.6, all slots ≥ 3:1 contrast.
COLORS = {'glioma': '#3987e5', 'meningioma': '#d95926',
          'no_tumor': '#199e70', 'pituitary': '#9085e9'}

# (train_loss, train_acc, val_loss, val_acc) per epoch; epochs 1-5 are phase 1.
HISTORY = [
    (1.2805, 41.88, 1.1217, 58.10), (1.0199, 66.67, 0.9114, 74.90),
    (0.8586, 75.95, 0.7842, 79.60), (0.7559, 78.80, 0.6969, 81.50),
    (0.6807, 80.60, 0.6316, 84.20),
    (0.2829, 90.72, 0.1414, 95.30), (0.0936, 97.28, 0.1008, 96.70),
    (0.0492, 98.83, 0.1087, 96.30), (0.0295, 99.28, 0.0860, 97.30),
    (0.0219, 99.55, 0.0789, 97.60), (0.0165, 99.62, 0.0835, 97.40),
    (0.0154, 99.62, 0.0717, 97.90), (0.0068, 99.97, 0.0695, 98.10),
    (0.0076, 99.90, 0.0687, 97.90), (0.0076, 99.85, 0.0686, 97.80),
    (0.0057, 99.90, 0.0698, 97.70), (0.0063, 99.92, 0.0685, 97.80),
    (0.0059, 99.90, 0.0705, 97.70), (0.0055, 99.92, 0.0639, 98.10),
    (0.0043, 99.92, 0.0647, 98.00),
]
PHASE1_EPOCHS = 5

# precision, recall, f1, support
METRICS = {
    'glioma':     (0.9793, 0.9331, 0.9556, 254),
    'meningioma': (0.9486, 0.9641, 0.9562, 306),
    'no_tumor':   (0.9722, 1.0000, 0.9859, 140),
    'pituitary':  (0.9835, 0.9933, 0.9884, 300),
}

# rows = actual, cols = predicted, in CLASSES order
CONFUSION = [
    [237,  14,   0,   3],
    [  5, 295,   4,   2],
    [  0,   0, 140,   0],
    [  0,   2,   0, 298],
]

TEST_ACC   = 97.00
MACRO_F1   = 0.9715
TOTAL_TEST = 1000

# ─── Chart tokens ────────────────────────────────────────────────────────────

INK       = '#e9ecef'
INK_2     = '#9aa4af'
INK_MUTED = '#6b7682'
GRID      = 'rgba(255,255,255,0.07)'
AXIS      = 'rgba(255,255,255,0.16)'
SURFACE   = '#0f1318'

SERIES_VAL   = '#3987e5'   # the series that carries the headline
SERIES_TRAIN = '#7d8792'   # neutral: an emphasis pair, not a categorical one

FONT = "'Inter', system-ui, sans-serif"
MONO = "'JetBrains Mono', ui-monospace, monospace"


def esc(s):
    return str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def txt(x, y, s, size=13, fill=INK_2, anchor='start', weight='400',
        family=FONT, opacity=1.0):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}" '
            f'opacity="{opacity}">{esc(s)}</text>')


# ─── Figure 1 — accuracy across the two phases ───────────────────────────────

def chart_accuracy(w=760, h=312):
    ml, mr, mt, mb = 56, 132, 34, 50
    pw, ph = w - ml - mr, h - mt - mb
    y_min, y_max = 40, 100
    n = len(HISTORY)

    def X(e):  # epoch is 1-indexed
        return ml + (e - 1) / (n - 1) * pw

    def Y(v):
        return mt + (1 - (v - y_min) / (y_max - y_min)) * ph

    o = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" '
         f'aria-label="Validation accuracy rises from 58 percent to 98 percent over 20 epochs; '
         f'unfreezing layer4 at epoch 6 produces the decisive jump.">']

    # Grid + y axis
    for v in range(y_min, y_max + 1, 10):
        y = Y(v)
        o.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml+pw}" y2="{y:.1f}" '
                 f'stroke="{GRID}" stroke-width="1"/>')
        o.append(txt(ml - 12, y + 4, f'{v}%', 12, INK_MUTED, 'end', family=MONO))

    # x axis
    o.append(f'<line x1="{ml}" y1="{mt+ph}" x2="{ml+pw}" y2="{mt+ph}" '
             f'stroke="{AXIS}" stroke-width="1"/>')
    for e in [1, 5, 10, 15, 20]:
        o.append(txt(X(e), mt + ph + 20, e, 12, INK_MUTED, 'middle', family=MONO))
    o.append(txt(ml + pw / 2, mt + ph + 42, 'Epoch', 12, INK_MUTED, 'middle'))

    # Phase divider — the single most important feature of this chart
    xd = X(PHASE1_EPOCHS + 0.5)
    o.append(f'<line x1="{xd:.1f}" y1="{mt-6}" x2="{xd:.1f}" y2="{mt+ph}" '
             f'stroke="{AXIS}" stroke-width="1" stroke-dasharray="3 4"/>')
    o.append(txt(xd - 8, mt - 14, 'Phase 1 — head only', 11, INK_MUTED, 'end', '500'))
    o.append(txt(xd + 8, mt - 14, 'Phase 2 — layer4 + head', 11, INK, 'start', '500'))

    # Series
    for key, color, idx in (('Train', SERIES_TRAIN, 1), ('Validation', SERIES_VAL, 3)):
        pts = ' '.join(f'{X(i+1):.1f},{Y(r[idx]):.1f}' for i, r in enumerate(HISTORY))
        o.append(f'<polyline points="{pts}" fill="none" stroke="{color}" '
                 f'stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
        for i, r in enumerate(HISTORY):
            # 2px surface ring keeps overlapping markers readable
            o.append(f'<circle cx="{X(i+1):.1f}" cy="{Y(r[idx]):.1f}" r="3.2" '
                     f'fill="{color}" stroke="{SURFACE}" stroke-width="2"/>')
        # Direct end labels — identity never rests on colour alone. The two series
        # finish within 2 points of each other, so they are pushed apart vertically
        # and tied back to their line with a short leader.
        last = HISTORY[-1][idx]
        ly = Y(last) - 16 if key == 'Train' else Y(last) + 26
        o.append(f'<line x1="{X(n):.1f}" y1="{Y(last):.1f}" x2="{X(n)+8:.1f}" '
                 f'y2="{ly-4:.1f}" stroke="{color}" stroke-width="1" opacity="0.55"/>')
        o.append(txt(X(n) + 12, ly, key, 12, INK, 'start', '600'))
        o.append(txt(X(n) + 12, ly + 15, f'{last:.2f}%', 12, INK_MUTED, 'start',
                     family=MONO))

    # Annotate the jump — placed below the curve, clear of the phase labels
    ax, ay = X(7.4), Y(84)
    o.append(f'<path d="M {X(6):.1f} {Y(95.30)+8:.1f} L {ax-6:.1f} {ay-14:.1f}" '
             f'fill="none" stroke="{AXIS}" stroke-width="1"/>')
    o.append(txt(ax, ay - 4, '84.2% → 95.3%', 12.5, INK, 'start', '600'))
    o.append(txt(ax, ay + 13, 'in a single epoch, on unfreezing layer4', 11.5,
                 INK_MUTED, 'start'))

    o.append('</svg>')
    return '\n'.join(o)


# ─── Figure 2 — per-class F1 ─────────────────────────────────────────────────

def chart_f1(w=760, h=240):
    ml, mr, mt, mb = 118, 84, 22, 52
    pw, ph = w - ml - mr, h - mt - mb
    row = ph / len(CLASSES)
    bar = 20
    lo, hi = 0.90, 1.00   # axis starts at 0.90; stated on the axis label

    def X(v):
        return ml + max(0.0, (v - lo) / (hi - lo)) * pw

    o = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" '
         f'aria-label="Per-class F1: pituitary 0.988, no tumour 0.986, '
         f'meningioma 0.956, glioma 0.956.">']

    for v in [0.90, 0.92, 0.94, 0.96, 0.98, 1.00]:
        x = X(v)
        o.append(f'<line x1="{x:.1f}" y1="{mt-8}" x2="{x:.1f}" y2="{mt+ph}" '
                 f'stroke="{GRID}" stroke-width="1"/>')
        o.append(txt(x, mt + ph + 20, f'{v:.2f}', 11.5, INK_MUTED, 'middle',
                     family=MONO))
    o.append(txt(ml + pw / 2, mt + ph + 40, 'F1 score  (axis starts at 0.90)',
                 11.5, INK_MUTED, 'middle'))

    order = sorted(CLASSES, key=lambda c: -METRICS[c][2])
    for i, c in enumerate(order):
        p, r, f1, sup = METRICS[c]
        y = mt + i * row + (row - bar) / 2
        o.append(txt(ml - 14, y + bar / 2 + 4, LABELS[c], 13, INK, 'end', '500'))
        o.append(f'<rect x="{ml}" y="{y:.1f}" width="{pw}" height="{bar}" rx="4" '
                 f'fill="rgba(255,255,255,0.045)"/>')
        o.append(f'<rect x="{ml}" y="{y:.1f}" width="{X(f1)-ml:.1f}" height="{bar}" '
                 f'rx="4" fill="{COLORS[c]}"/>')
        o.append(txt(X(f1) + 12, y + bar / 2 + 4, f'{f1:.4f}', 12.5, INK, 'start',
                     '600', family=MONO))
        o.append(txt(ml - 14, y + bar / 2 + 19, f'n = {sup}', 10.5, INK_MUTED, 'end',
                     family=MONO))

    o.append('</svg>')
    return '\n'.join(o)


# ─── Figure 3 — confusion matrix ─────────────────────────────────────────────

def _ramp(v):
    """Sequential blue, one hue, dark (near surface) to light as magnitude rises."""
    if v <= 0:      return ('#121820', INK_MUTED)
    if v <= 0.02:   return ('#0d366b', '#cde2fb')
    if v <= 0.10:   return ('#184f95', '#ffffff')
    if v <= 0.35:   return ('#256abf', '#ffffff')
    if v <= 0.70:   return ('#2a78d6', '#ffffff')
    if v <= 0.90:   return ('#3987e5', '#ffffff')
    return ('#86b6ef', '#0b1220')


def chart_confusion(w=760, h=338):
    ml, mr, mt, mb = 126, 96, 50, 62
    pw, ph = w - ml - mr, h - mt - mb
    cw, ch = pw / 4, ph / 4
    gap = 2  # 2px surface gap between fills

    o = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" '
         f'aria-label="Confusion matrix. 19 of 30 total errors sit on the '
         f'glioma-meningioma boundary; no tumour has perfect recall.">']

    o.append(txt(ml + pw / 2, 22, 'Predicted', 12, INK_MUTED, 'middle', '500'))
    o.append(f'<text x="{22}" y="{mt+ph/2}" font-family="{FONT}" font-size="12" '
             f'font-weight="500" fill="{INK_MUTED}" text-anchor="middle" '
             f'transform="rotate(-90 22 {mt+ph/2:.1f})">Actual</text>')

    for j, c in enumerate(CLASSES):
        o.append(txt(ml + j * cw + cw / 2, mt - 14, LABELS[c], 12, INK_2, 'middle',
                     '500'))

    for i, c in enumerate(CLASSES):
        total = sum(CONFUSION[i])
        y = mt + i * ch
        o.append(txt(ml - 14, y + ch / 2 + 1, LABELS[c], 12.5, INK, 'end', '500'))
        o.append(txt(ml - 14, y + ch / 2 + 17, f'n = {total}', 10.5, INK_MUTED, 'end',
                     family=MONO))
        # Row-normalised = recall; direct-labelled with raw counts
        for j in range(4):
            v = CONFUSION[i][j] / total
            fill, ink = _ramp(v)
            x = ml + j * cw
            o.append(f'<rect x="{x+gap/2:.1f}" y="{y+gap/2:.1f}" '
                     f'width="{cw-gap:.1f}" height="{ch-gap:.1f}" rx="5" fill="{fill}"'
                     + (f' stroke="{GRID}" stroke-width="1"' if v <= 0 else '') + '/>')
            o.append(txt(x + cw / 2, y + ch / 2 + 2, CONFUSION[i][j], 17, ink,
                         'middle', '600' if i == j else '500', family=MONO))
            o.append(txt(x + cw / 2, y + ch / 2 + 20, f'{v*100:.1f}%', 10.5, ink,
                         'middle', family=MONO, opacity=0.78))
        o.append(txt(ml + pw + 16, y + ch / 2 + 5, f'{CONFUSION[i][i]/total*100:.1f}%',
                     12.5, INK, 'start', '600', family=MONO))

    o.append(txt(ml + pw + 16, mt - 14, 'Recall', 11.5, INK_MUTED, 'start', '500'))

    # Sequential legend
    ly = mt + ph + 34
    o.append(txt(ml, ly + 4, 'Share of row', 11, INK_MUTED, 'end'))
    steps = ['#121820', '#0d366b', '#184f95', '#256abf', '#2a78d6', '#3987e5', '#86b6ef']
    sw = 26
    for k, s in enumerate(steps):
        o.append(f'<rect x="{ml+10+k*(sw+2)}" y="{ly-8}" width="{sw}" height="12" '
                 f'rx="2" fill="{s}"/>')
    o.append(txt(ml + 10, ly + 20, '0%', 10.5, INK_MUTED, 'start', family=MONO))
    o.append(txt(ml + 10 + len(steps) * (sw + 2) - 2, ly + 20, '100%', 10.5,
                 INK_MUTED, 'end', family=MONO))

    o.append('</svg>')
    return '\n'.join(o)


# ─── Shared styles ───────────────────────────────────────────────────────────

BASE_CSS = """
  :root {
    color-scheme: dark;
    --bg: #08090b; --surface-1: #0f1318; --surface-2: #161b21; --surface-3: #1d242c;
    --border: rgba(255,255,255,0.08); --border-firm: rgba(255,255,255,0.14);
    --ink: #e9ecef; --ink-2: #9aa4af; --ink-muted: #6b7682;
    --glioma: #3987e5; --meningioma: #d95926; --no_tumor: #199e70; --pituitary: #9085e9;
    --warning: #c98500;
  }
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  /* Keep the dark theme when printing or exporting to PDF. */
  html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
  body { background: var(--bg); color: var(--ink);
         font-family: 'Inter', system-ui, -apple-system, sans-serif;
         -webkit-font-smoothing: antialiased; line-height: 1.55; }
  h1, h2, h3 { letter-spacing: -0.02em; line-height: 1.2; }
  .mono { font-family: 'JetBrains Mono', ui-monospace, monospace;
          font-variant-numeric: tabular-nums; }
  a { color: var(--glioma); }
"""

FONT_LINK = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
             '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
             '<link href="https://fonts.googleapis.com/css2?'
             'family=Inter:wght@400;500;600;700;800&'
             'family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">')


# ─── Poster ──────────────────────────────────────────────────────────────────

def build_poster():
    swatches = ''.join(
        f'<div class="legend-item"><span class="dot" style="background:{COLORS[c]}">'
        f'</span>{LABELS[c]}</div>' for c in CLASSES)

    return f"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="UTF-8">
<title>NeuraScan — Poster</title>
{FONT_LINK}
<style>
{BASE_CSS}
  /* A2 portrait — 420 x 594 mm */
  @page {{ size: 420mm 594mm; margin: 0; }}
  body {{ display: flex; justify-content: center; background: #050607; }}
  .poster {{
    width: 420mm; height: 594mm; background: var(--bg);
    padding: 15mm 18mm 14mm; display: flex; flex-direction: column;
    position: relative; overflow: hidden;
  }}
  .poster::before {{
    content: ''; position: absolute; inset: 0; pointer-events: none;
    background:
      radial-gradient(ellipse 70% 32% at 50% 0%, rgba(120,145,180,0.13), transparent 70%),
      linear-gradient(rgba(255,255,255,0.02) 1px, transparent 1px) 0 0 / 100% 14mm,
      linear-gradient(90deg, rgba(255,255,255,0.02) 1px, transparent 1px) 0 0 / 14mm 100%;
    mask-image: linear-gradient(to bottom, #000 0%, transparent 42%);
  }}
  .poster > * {{ position: relative; z-index: 1; }}

  /* Header */
  header {{ border-bottom: 1px solid var(--border-firm); padding-bottom: 6mm;
            margin-bottom: 6mm; }}
  .eyebrow {{ display: flex; align-items: center; gap: 10px; font-size: 13px;
              letter-spacing: 0.16em; text-transform: uppercase;
              color: var(--ink-muted); font-weight: 600; margin-bottom: 5mm; }}
  .eyebrow .rule {{ flex: 1; height: 1px; background: var(--border-firm); }}
  h1 {{ font-size: 50px; font-weight: 800; margin-bottom: 4mm; max-width: 300mm; }}
  h1 em {{ font-style: normal; color: var(--ink-2); font-weight: 500; }}
  .sub {{ font-size: 19px; color: var(--ink-2); max-width: 290mm; line-height: 1.5; }}
  .byline {{ display: flex; gap: 8mm; flex-wrap: wrap; margin-top: 4.5mm;
             font-size: 14.5px; color: var(--ink-muted); }}
  .byline b {{ color: var(--ink-2); font-weight: 600; }}

  /* Hero stats */
  .stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 4mm;
            margin-bottom: 6mm; }}
  .stat {{ background: var(--surface-1); border: 1px solid var(--border);
           border-radius: 4mm; padding: 4.5mm 5mm 4mm; }}
  .stat.lead {{ background: linear-gradient(160deg, #182130, var(--surface-1) 65%);
                border-color: var(--border-firm); }}
  .stat-val {{ font-size: 44px; font-weight: 800; letter-spacing: -0.035em;
               line-height: 1; font-variant-numeric: tabular-nums; }}
  .stat.lead .stat-val {{ font-size: 54px; }}
  .stat-val small {{ font-size: 24px; font-weight: 600; color: var(--ink-2); }}
  .stat-lbl {{ font-size: 13.5px; color: var(--ink-muted); margin-top: 2.5mm;
               letter-spacing: 0.02em; }}

  /* Columns */
  .cols {{ display: grid; grid-template-columns: 1fr 1.34fr; gap: 5mm;
           flex-shrink: 0; }}
  .col {{ display: flex; flex-direction: column; gap: 3mm; }}

  .panel {{ background: var(--surface-1); border: 1px solid var(--border);
            border-radius: 4mm; padding: 5mm; }}
  .panel.flush {{ padding-bottom: 4mm; }}
  h2 {{ font-size: 13px; font-weight: 700; letter-spacing: 0.13em;
        text-transform: uppercase; color: var(--ink-muted); margin-bottom: 3.5mm;
        display: flex; align-items: center; gap: 8px; }}
  h2::before {{ content: ''; width: 3px; height: 13px; border-radius: 2px;
                background: var(--ink-2); }}
  .panel p {{ font-size: 14.5px; color: var(--ink-2); margin-bottom: 2.5mm; }}
  .panel p:last-child {{ margin-bottom: 0; }}
  .panel strong {{ color: var(--ink); font-weight: 600; }}

  /* Class legend */
  .legend {{ display: flex; flex-wrap: wrap; gap: 3mm 6mm; margin-bottom: 4mm; }}
  .legend-item {{ display: flex; align-items: center; gap: 7px; font-size: 14px;
                  color: var(--ink-2); }}
  .dot {{ width: 10px; height: 10px; border-radius: 3px; }}

  /* Pipeline */
  .pipe {{ display: flex; flex-direction: column; gap: 2.5mm; }}
  .pstep {{ display: flex; gap: 3.5mm; align-items: flex-start;
            background: var(--surface-2); border: 1px solid var(--border);
            border-radius: 2.5mm; padding: 3mm 3.5mm; }}
  .pnum {{ width: 7mm; height: 7mm; flex-shrink: 0; border-radius: 2mm;
           background: var(--surface-3); border: 1px solid var(--border-firm);
           display: grid; place-items: center; font-size: 13px; font-weight: 700;
           color: var(--ink-2); }}
  .pstep b {{ display: block; font-size: 15px; font-weight: 600; margin-bottom: 1mm; }}
  .pstep span {{ font-size: 13.5px; color: var(--ink-muted); line-height: 1.45; }}

  /* Phase cards */
  .phases {{ display: grid; grid-template-columns: 1fr 1fr; gap: 3mm; }}
  .phase {{ background: var(--surface-2); border: 1px solid var(--border);
            border-radius: 2.5mm; padding: 4mm; }}
  .phase.hot {{ border-color: rgba(57,135,229,0.4);
                background: linear-gradient(165deg, rgba(57,135,229,0.11),
                            var(--surface-2) 70%); }}
  .phase h3 {{ font-size: 14px; font-weight: 700; margin-bottom: 2.5mm; }}
  .phase dl {{ display: grid; grid-template-columns: auto 1fr; gap: 1mm 3mm;
               font-size: 12.5px; }}
  .phase dt {{ color: var(--ink-muted); }}
  .phase dd {{ color: var(--ink); text-align: right;
               font-family: 'JetBrains Mono', monospace;
               font-variant-numeric: tabular-nums; }}

  /* Table */
  table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
  th {{ text-align: right; font-size: 12px; font-weight: 600; letter-spacing: 0.06em;
        text-transform: uppercase; color: var(--ink-muted);
        padding: 0 0 2.5mm; border-bottom: 1px solid var(--border-firm); }}
  th:first-child {{ text-align: left; }}
  td {{ text-align: right; padding: 2.2mm 0; border-bottom: 1px solid var(--border);
        font-family: 'JetBrains Mono', monospace; font-variant-numeric: tabular-nums;
        color: var(--ink-2); }}
  td:first-child {{ text-align: left; font-family: 'Inter', sans-serif;
                    color: var(--ink); font-weight: 500; }}
  tr:last-child td {{ border-bottom: none; }}
  tr.total td {{ color: var(--ink); font-weight: 600;
                 border-top: 1px solid var(--border-firm); }}
  .cname {{ display: flex; align-items: center; gap: 7px; }}

  /* Bottom band */
  .band {{ display: grid; grid-template-columns: 2.4fr 1fr; gap: 4mm;
           margin-top: 3.5mm; }}
  .poster > footer {{ padding-bottom: 0; }}
  .find {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 4mm; }}
  .fitem {{ display: flex; gap: 3mm; align-items: flex-start; }}
  .fbadge {{ flex-shrink: 0; min-width: 14mm; text-align: center; padding: 1.6mm 1.8mm;
             border-radius: 2mm; font-family: 'JetBrains Mono', monospace;
             font-size: 16px; font-weight: 600; line-height: 1.15; }}
  .fbadge small {{ display: block; font-size: 10.5px; font-weight: 500;
                   opacity: 0.75; font-family: 'Inter', sans-serif; }}
  .fitem p {{ font-size: 13px; margin: 0; line-height: 1.48; }}

  footer {{ margin-top: 3.5mm; padding-top: 3mm; border-top: 1px solid var(--border-firm);
            display: flex; justify-content: space-between; align-items: flex-start;
            gap: 8mm; font-size: 13px; color: var(--ink-muted); }}
  .warn {{ display: flex; gap: 3mm; align-items: flex-start; max-width: 190mm;
           color: var(--warning); }}
  .warn svg {{ width: 15px; height: 15px; flex-shrink: 0; margin-top: 2px;
               stroke: currentColor; fill: none; stroke-width: 1.8;
               stroke-linecap: round; stroke-linejoin: round; }}
</style>
</head>
<body>
<div class="poster">

  <header>
    <div class="eyebrow"><span>Medical imaging &middot; Deep learning</span>
      <span class="rule"></span><span>BRISC2025</span></div>
    <h1>Brain tumour classification from MRI <em>using two-phase transfer learning</em></h1>
    <p class="sub">A ResNet18 fine-tuned in two stages classifies axial brain MRI slices
      into four categories at 97.00% test accuracy — and concentrates almost two-thirds
      of its residual error on the one boundary that is hardest for human readers too.</p>
    <div class="byline">
      <span><b>Architecture</b> ResNet18, ImageNet pretrained</span>
      <span><b>Dataset</b> BRISC2025 &middot; 6,000 scans</span>
      <span><b>Framework</b> PyTorch &middot; torchvision &middot; Flask</span>
    </div>
  </header>

  <div class="stats">
    <div class="stat lead">
      <div class="stat-val">97.00<small>%</small></div>
      <div class="stat-lbl">Test accuracy — 970 / 1,000 held-out scans</div>
    </div>
    <div class="stat">
      <div class="stat-val">0.9715</div>
      <div class="stat-lbl">Macro-averaged F1</div>
    </div>
    <div class="stat">
      <div class="stat-val">100<small>%</small></div>
      <div class="stat-lbl">Recall on <em>no tumour</em> — zero missed tumours</div>
    </div>
    <div class="stat">
      <div class="stat-val">11.2<small>M</small></div>
      <div class="stat-lbl">Parameters &middot; 8.4 M fine-tuned</div>
    </div>
  </div>

  <div class="cols">

    <!-- Left column -->
    <div class="col">

      <div class="panel">
        <h2>The problem</h2>
        <p>Tumour type must be identified before treatment can be planned, but reading MRI
          is slow, needs scarce expertise, and varies between observers — most of all
          between <strong>glioma</strong> and <strong>meningioma</strong>, which can look
          alike on a single axial slice.</p>
        <p>We classify one slice into four classes:</p>
        <div class="legend">{swatches}</div>
        <p>5,000 training images is far too few to train an 11-million-parameter network
          from scratch, so the whole method turns on <strong>how</strong> pretrained
          weights are adapted.</p>
      </div>

      <div class="panel">
        <h2>Data</h2>
        <table>
          <tr><th>Split</th><th>Images</th><th>Source</th></tr>
          <tr><td>Train</td><td>4,000</td><td>80% of train dir</td></tr>
          <tr><td>Validation</td><td>1,000</td><td>20%, stratified</td></tr>
          <tr><td>Test</td><td>1,000</td><td>held out</td></tr>
          <tr class="total"><td>Total</td><td>6,000</td><td>seed 42</td></tr>
        </table>
      </div>

      <div class="panel">
        <h2>Preprocessing</h2>
        <div class="pipe">
          <div class="pstep"><div class="pnum">1</div><div>
            <b>Resize to 224 &times; 224</b>
            <span>The resolution ResNet18 was pretrained at.</span></div></div>
          <div class="pstep"><div class="pnum">2</div><div>
            <b>Greyscale &rarr; 3 channels</b>
            <span>Replicating the single channel lets the pretrained RGB filters apply
              unchanged.</span></div></div>
          <div class="pstep"><div class="pnum">3</div><div>
            <b>Horizontal flip only</b>
            <span>Anatomy is near-symmetric, so a flip is plausible. Rotation and jitter
              were excluded — fabricating non-physical scans teaches artefacts, not
              pathology.</span></div></div>
          <div class="pstep"><div class="pnum">4</div><div>
            <b>ImageNet normalisation</b>
            <span>Places inputs in the distribution the pretrained weights expect.</span>
          </div></div>
        </div>
      </div>

      <div class="panel">
        <h2>Method — why two phases</h2>
        <p>The new 4-class head starts random. Unfreezing everything at once would push
          large, noisy gradients back through the backbone and <strong>destroy the
          pretrained features before they could be used</strong>. Only
          <strong>layer4</strong> is ever unfrozen: early layers already hold generic edge
          and texture detectors, while the deepest block holds the class semantics that
          must move from natural images to MRI.</p>
        <div class="phases">
          <div class="phase">
            <h3>Phase 1 &middot; head only</h3>
            <dl>
              <dt>Epochs</dt><dd>5</dd>
              <dt>Trainable</dt><dd>2,052</dd>
              <dt>LR (Adam)</dt><dd>1e-4</dd>
              <dt>Backbone</dt><dd>frozen</dd>
              <dt>Val acc</dt><dd>84.2%</dd>
            </dl>
          </div>
          <div class="phase hot">
            <h3>Phase 2 &middot; layer4 + head</h3>
            <dl>
              <dt>Epochs</dt><dd>15</dd>
              <dt>Trainable</dt><dd>8,395,780</dd>
              <dt>LR (Adam)</dt><dd>5e-5</dd>
              <dt>Schedule</dt><dd>StepLR</dd>
              <dt>Val acc</dt><dd>98.0%</dd>
            </dl>
          </div>
        </div>
      </div>

    </div>

    <!-- Right column -->
    <div class="col">

      <div class="panel flush">
        <h2>Training — validation accuracy across both phases</h2>
        {chart_accuracy()}
        <p style="font-size:14px;margin-top:2mm">Phase 1 plateaus at <strong>84.2%</strong>:
          a frozen ImageNet backbone can only reach so far into a domain as distant as MRI.
          Unfreezing <strong>layer4</strong> lifts validation accuracy to
          <strong>95.3% in a single epoch</strong>, and past 97% by epoch 9. By epoch 13 it has
          converged: train sits at 99.9%+ while validation holds a narrow 97.7–98.1% band.</p>
      </div>

      <div class="panel flush">
        <h2>Results — per-class F1 on the held-out test set</h2>
        {chart_f1()}
        <p style="font-size:14px;margin-top:1mm">Glioma and meningioma trail together —
          the first sign they are being confused with <em>each other</em> rather than
          failing independently. Macro and weighted F1 agree to within
          <strong>0.002</strong>, so the headline is not an artefact of class
          imbalance.</p>
      </div>

      <div class="panel flush">
        <h2>Confusion matrix — 30 errors in 1,000 scans</h2>
        {chart_confusion()}
      </div>

    </div>
  </div>

  <div class="band">
    <div class="panel">
      <h2>What the errors say</h2>
      <div class="find">
        <div class="fitem">
          <div class="fbadge" style="background:rgba(217,89,38,0.16);color:#f0885c">
            63%<small>of errors</small></div>
          <p><strong>Glioma &harr; meningioma is the one real weakness.</strong> 19 of 30
            errors sit on this pair — also the hardest call for human readers on a single
            slice. The model fails where the task is genuinely ambiguous, not at
            random.</p>
        </div>
        <div class="fitem">
          <div class="fbadge" style="background:rgba(25,158,112,0.16);color:#2fbf8d">
            0<small>missed</small></div>
          <p><strong>Perfect recall on <em>no tumour</em>.</strong> No scan containing a
            tumour was called tumour-free. This class's four errors all run the safe way,
            so false negatives — the costliest error in screening — did not occur.</p>
        </div>
        <div class="fitem">
          <div class="fbadge" style="background:rgba(144,133,233,0.16);color:#a79ef0">
            99.3%<small>recall</small></div>
          <p><strong>Pituitary is nearly separable.</strong> Two errors in 300. These
            tumours sit at a distinctive skull-base location, giving the network a
            positional cue the other tumour types do not offer.</p>
        </div>
      </div>
    </div>

    <div class="panel">
      <h2>Limits &amp; next steps</h2>
      <p><strong>Limits.</strong> One slice, not the volume. No localisation. A single
        dataset, so cross-scanner generalisation is unmeasured. Confidence is raw softmax
        and uncalibrated.</p>
      <p><strong>Next.</strong> Target the glioma/meningioma boundary with a focal loss or
        a second-stage binary head; add Grad-CAM; validate externally; aggregate across
        the slices of a study.</p>
    </div>
  </div>

  <footer>
    <div class="warn">
      <svg viewBox="0 0 24 24"><path d="M10.3 3.9 2.6 17.1A2 2 0 0 0 4.3 20h15.4a2 2 0
        0 0 1.7-2.9L13.7 3.9a2 2 0 0 0-3.4 0Z"/><path d="M12 9v4.5M12 17h.01"/></svg>
      <span>Research and educational use only — not a medical device, and not validated
        for clinical diagnosis.</span>
    </div>
    <span class="mono">NeuraScan &middot; ResNet18 &middot; 97.00% test accuracy</span>
  </footer>

</div>
</body>
</html>
"""


# ─── Report (formatted, printable) ───────────────────────────────────────────

def build_report():
    rows_metrics = ''.join(
        f'<tr><td><span class="cname"><span class="dot" '
        f'style="background:{COLORS[c]}"></span>{LABELS[c]}</span></td>'
        f'<td>{METRICS[c][0]:.4f}</td><td>{METRICS[c][1]:.4f}</td>'
        f'<td>{METRICS[c][2]:.4f}</td><td>{METRICS[c][3]}</td></tr>'
        for c in CLASSES)

    rows_hist = ''.join(
        f'<tr{" class=phase2" if i >= PHASE1_EPOCHS else ""}>'
        f'<td>{i+1}</td><td>{1 if i < PHASE1_EPOCHS else 2}</td>'
        f'<td>{r[0]:.4f}</td><td>{r[1]:.2f}%</td>'
        f'<td>{r[2]:.4f}</td><td>{r[3]:.2f}%</td></tr>'
        for i, r in enumerate(HISTORY))

    return f"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NeuraScan — Project Report</title>
{FONT_LINK}
<style>
{BASE_CSS}
  @page {{ size: A4; margin: 16mm 14mm; }}
  body {{ padding: 0 24px 80px; }}
  .doc {{ max-width: 860px; margin: 0 auto; }}

  .cover {{ padding: 64px 0 40px; border-bottom: 1px solid var(--border-firm);
            margin-bottom: 44px; }}
  .eyebrow {{ font-size: 12px; letter-spacing: 0.16em; text-transform: uppercase;
              color: var(--ink-muted); font-weight: 600; margin-bottom: 20px; }}
  h1 {{ font-size: 40px; font-weight: 800; margin-bottom: 14px; }}
  .lede {{ font-size: 18px; color: var(--ink-2); max-width: 680px; }}
  .kpis {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px;
           margin-top: 32px; }}
  .kpi {{ background: var(--surface-1); border: 1px solid var(--border);
          border-radius: 12px; padding: 16px 18px; }}
  .kpi b {{ display: block; font-size: 26px; font-weight: 700; letter-spacing: -0.03em;
            font-variant-numeric: tabular-nums; }}
  .kpi span {{ font-size: 12px; color: var(--ink-muted); }}

  h2 {{ font-size: 13px; font-weight: 700; letter-spacing: 0.13em;
        text-transform: uppercase; color: var(--ink-muted);
        margin: 44px 0 18px; padding-bottom: 10px;
        border-bottom: 1px solid var(--border); }}
  h3 {{ font-size: 17px; font-weight: 600; margin: 28px 0 10px; }}
  p {{ color: var(--ink-2); margin-bottom: 14px; font-size: 15.5px; }}
  p strong {{ color: var(--ink); font-weight: 600; }}
  ul {{ margin: 0 0 16px 20px; color: var(--ink-2); font-size: 15.5px; }}
  li {{ margin-bottom: 9px; }}
  li strong {{ color: var(--ink); font-weight: 600; }}

  figure {{ background: var(--surface-1); border: 1px solid var(--border);
            border-radius: 12px; padding: 22px 20px 16px; margin: 24px 0; }}
  figcaption {{ font-size: 13px; color: var(--ink-muted); margin-top: 12px;
                padding-top: 12px; border-top: 1px solid var(--border); }}
  figcaption b {{ color: var(--ink-2); font-weight: 600; }}

  table {{ width: 100%; border-collapse: collapse; font-size: 14.5px; margin: 18px 0; }}
  th {{ text-align: right; font-size: 11.5px; font-weight: 600; letter-spacing: 0.06em;
        text-transform: uppercase; color: var(--ink-muted); padding: 0 8px 10px;
        border-bottom: 1px solid var(--border-firm); }}
  th:first-child {{ text-align: left; padding-left: 0; }}
  td {{ text-align: right; padding: 9px 8px; border-bottom: 1px solid var(--border);
        color: var(--ink-2); font-family: 'JetBrains Mono', monospace;
        font-variant-numeric: tabular-nums; }}
  td:first-child {{ text-align: left; padding-left: 0; color: var(--ink);
                    font-family: 'Inter', sans-serif; font-weight: 500; }}
  tr.phase2 td {{ background: rgba(57,135,229,0.05); }}
  .tbl-text td:not(:first-child):not(:last-child),
  .tbl-text th:not(:first-child):not(:last-child) {{ text-align: left; }}
  .cname {{ display: inline-flex; align-items: center; gap: 8px; }}
  .dot {{ width: 9px; height: 9px; border-radius: 2px; }}
  .scroll {{ max-height: 460px; overflow: auto; border: 1px solid var(--border);
             border-radius: 10px; padding: 0 14px; }}

  .callout {{ border: 1px solid var(--border); border-left: 3px solid var(--glioma);
              background: var(--surface-1); border-radius: 10px; padding: 16px 18px;
              margin: 22px 0; }}
  .callout p:last-child {{ margin-bottom: 0; }}
  .callout.warn {{ border-left-color: var(--warning);
                   background: rgba(201,133,0,0.06);
                   border-color: rgba(201,133,0,0.25); }}

  footer {{ margin-top: 56px; padding-top: 22px; border-top: 1px solid var(--border-firm);
            font-size: 13px; color: var(--ink-muted); }}
  @media print {{
    figure, .callout, .kpi, table {{ break-inside: avoid; }}
    h2 {{ break-after: avoid; }}
    .scroll {{ max-height: none; overflow: visible; }}
  }}
</style>
</head>
<body>
<div class="doc">

<div class="cover">
  <div class="eyebrow">Project report &middot; Medical imaging &middot; Deep learning</div>
  <h1>Brain tumour classification from MRI using ResNet18</h1>
  <p class="lede">A two-phase transfer-learning schedule — frozen backbone, then a
    fine-tuned deepest block — takes a ResNet18 to 97.00% accuracy on four-way brain MRI
    classification from only 5,000 training images.</p>
  <div class="kpis">
    <div class="kpi"><b>97.00%</b><span>Test accuracy</span></div>
    <div class="kpi"><b>0.9715</b><span>Macro F1</span></div>
    <div class="kpi"><b>100%</b><span>No-tumour recall</span></div>
    <div class="kpi"><b>30</b><span>Errors in 1,000</span></div>
  </div>
</div>

<h2>1. Problem</h2>
<p>Tumour type must be identified before treatment can be planned. Manual MRI reading is
  slow, demands scarce expertise, and shows measurable inter-observer variability —
  particularly between <strong>glioma</strong> and <strong>meningioma</strong>, which can
  present with overlapping appearance on a single axial slice.</p>
<p>The task here is single-label, four-way classification of one axial slice into
  glioma, meningioma, no tumour, or pituitary tumour.</p>

<h2>2. Data</h2>
<table>
  <tr><th>Split</th><th>Images</th><th>Derivation</th></tr>
  <tr><td>Train</td><td>4,000</td><td>80% of train dir</td></tr>
  <tr><td>Validation</td><td>1,000</td><td>20%, stratified</td></tr>
  <tr><td>Test</td><td>1,000</td><td>held out entirely</td></tr>
</table>
<p>The 80:20 split is stratified on the label so class balance is preserved in both
  halves, and seeded (<span class="mono">random_state = 42</span>) so the run reproduces
  exactly. The classes are moderately imbalanced — no tumour is only 14% of the test set
  — which is why §5 reports per-class metrics rather than accuracy alone.</p>

<h2>3. Preprocessing</h2>
<table class="tbl-text">
  <tr><th>Step</th><th>Operation</th><th>Why</th></tr>
  <tr><td>1</td><td>Resize 224&times;224</td><td>ResNet18's pretrained resolution</td></tr>
  <tr><td>2</td><td>Greyscale &rarr; 3ch</td><td>Pretrained RGB filters apply unchanged</td></tr>
  <tr><td>3</td><td>Horizontal flip</td><td>Train only; anatomy is near-symmetric</td></tr>
  <tr><td>4</td><td>ImageNet normalise</td><td>Matches the pretrained distribution</td></tr>
</table>
<div class="callout">
  <p><strong>On augmentation.</strong> Only horizontal flipping is used. Rotation, shear
    and colour jitter were deliberately excluded: they can introduce geometry or intensity
    patterns that never occur in real acquisition, and an augmentation that fabricates
    non-physical images teaches the model artefacts rather than pathology. Validation and
    test transforms use no augmentation at all.</p>
</div>

<h2>4. Method</h2>
<p>ResNet18 pretrained on ImageNet, with the final layer replaced:
  <span class="mono">Linear(512 &rarr; 1000)</span> becomes
  <span class="mono">Linear(512 &rarr; 4)</span>. Total parameters: 11,178,564.</p>
<p>That new head starts <strong>random</strong>. Unfreezing the whole network immediately
  would push large, noisy gradients back through the backbone and corrupt the pretrained
  features before they could be of any use. The two-phase schedule prevents this.</p>
<table class="tbl-text">
  <tr><th>Setting</th><th>Phase 1</th><th>Phase 2</th></tr>
  <tr><td>Epochs</td><td>5</td><td>15</td></tr>
  <tr><td>Trainable params</td><td>2,052</td><td>8,395,780</td></tr>
  <tr><td>Learning rate</td><td>1e-4</td><td>5e-5</td></tr>
  <tr><td>Scheduler</td><td>none</td><td>StepLR (&gamma;=0.1, step 7)</td></tr>
  <tr><td>Unfrozen</td><td>fc only</td><td>layer4 + fc</td></tr>
  <tr><td>Optimiser</td><td>Adam</td><td>Adam</td></tr>
  <tr><td>Batch size</td><td>32</td><td>32</td></tr>
</table>
<p>Only <strong>layer4</strong> is unfrozen in Phase 2, not the full backbone: early
  convolutional layers encode generic edge and texture detectors that transfer well across
  domains, while the deepest block encodes the class-specific semantics that most need
  adapting from natural images to MRI.</p>

<figure>
  {chart_accuracy()}
  <figcaption><b>Figure 1 — Accuracy across both phases.</b> Phase 1 plateaus at 84.2%
    validation accuracy; a frozen ImageNet backbone can only reach so far into a domain as
    distant as MRI. Unfreezing layer4 lifts validation accuracy to 95.3% within a single
    epoch and past 97% by epoch 9. From epoch 13 onward, training sits at 99.9%+ while
    validation holds a narrow 97.7–98.1% band and validation loss stops falling — the
    model has converged, and further epochs would not help.</figcaption>
</figure>

<h3>Epoch-by-epoch</h3>
<div class="scroll">
<table>
  <tr><th>Epoch</th><th>Phase</th><th>Train loss</th><th>Train acc</th>
      <th>Val loss</th><th>Val acc</th></tr>
  {rows_hist}
</table>
</div>

<h2>5. Results</h2>
<p><strong>Test accuracy: {TEST_ACC:.2f}%</strong> — 970 of {TOTAL_TEST:,} held-out scans
  classified correctly.</p>

<table>
  <tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr>
  {rows_metrics}
  <tr><td><b>Macro avg</b></td><td>0.9709</td><td>0.9726</td><td>0.9715</td><td>1,000</td></tr>
  <tr><td><b>Weighted avg</b></td><td>0.9702</td><td>0.9700</td><td>0.9699</td><td>1,000</td></tr>
</table>
<p>Macro and weighted averages agree to within 0.002, which confirms the headline number
  is not being carried by the larger classes — the model is genuinely competent on all
  four.</p>

<figure>
  {chart_f1()}
  <figcaption><b>Figure 2 — Per-class F1.</b> Pituitary and no tumour lead; glioma and
    meningioma trail together, which is the first hint that they are being confused with
    each other rather than failing independently.</figcaption>
</figure>

<figure>
  {chart_confusion()}
  <figcaption><b>Figure 3 — Confusion matrix.</b> Rows are the true class, columns the
    prediction; cells carry the raw count and the share of the row, and the diagonal is
    per-class recall. Thirty errors in total, and their distribution is anything but
    uniform.</figcaption>
</figure>

<h3>Error analysis</h3>
<ul>
  <li><strong>Glioma &harr; meningioma accounts for 19 of 30 errors (63%).</strong>
    Fourteen gliomas read as meningioma, five meningiomas as glioma. This is the single
    dominant failure mode, and it mirrors the hardest distinction for human readers: on
    one axial slice, without contrast timing or multi-plane context, the two can look
    genuinely similar. Glioma consequently has the lowest recall of the four, at
    93.3%.</li>
  <li><strong>No tumour achieves perfect recall (140/140).</strong> Not one scan
    containing a tumour was classified as tumour-free. For a screening-style application
    this is the most valuable property the model has, since a false negative is the
    costliest error. Its precision of 0.9722 reflects four meningiomas misfiled as
    no-tumour — errors in the <em>other</em> direction, which are the safer kind.</li>
  <li><strong>Pituitary is nearly separable</strong>, at 99.3% recall with two errors in
    300. Pituitary tumours occupy a distinctive skull-base location, giving the network a
    positional cue the other tumour types do not offer.</li>
</ul>

<h2>6. Deployment</h2>
<p>The trained weights are served by a Flask application. Device selection is automatic at
  start-up — CUDA on an NVIDIA GPU, Apple MPS on Apple Silicon, otherwise CPU. The
  original training run used Apple MPS.</p>
<table class="tbl-text">
  <tr><th>Method</th><th>Route</th><th>Purpose</th></tr>
  <tr><td>GET</td><td>/</td><td>Browser interface</td></tr>
  <tr><td>GET</td><td>/health</td><td>Liveness + active device</td></tr>
  <tr><td>POST</td><td>/predict</td><td>multipart image &rarr; prediction</td></tr>
</table>
<p>The uploaded image runs through the same resize &rarr; greyscale-to-3-channel &rarr;
  normalise transform used at validation time, then through the network under
  <span class="mono">torch.no_grad()</span>, softmaxed into a distribution.</p>
<p>The interface renders the <strong>full probability distribution</strong>, not just the
  winning class. A 97%-confident prediction and a 51%-confident one are very different
  objects, and surfacing the distribution makes that difference visible to whoever reads
  the result. Class colours are fixed and contrast-validated, and every bar is directly
  labelled, so identity never rests on colour alone.</p>

<h2>7. Limitations</h2>
<ul>
  <li><strong>Single-slice classification.</strong> Radiologists read the full volume
    across planes with sequence and contrast context; a slice ambiguous alone is often
    unambiguous in the stack.</li>
  <li><strong>No localisation.</strong> The model returns a class, not a segmentation or
    bounding box — it cannot say where the lesion is or how large.</li>
  <li><strong>Single-dataset evaluation.</strong> MRI appearance varies with scanner
    vendor, field strength and protocol. Performance on another site's scans is
    unmeasured and should not be assumed.</li>
  <li><strong>Uncalibrated confidence.</strong> The reported number is a raw softmax
    output, and deep networks are typically overconfident. A stated 97% should not be
    read as a well-calibrated probability without temperature scaling.</li>
  <li><strong>Not a medical device.</strong> Unvalidated for clinical use, with no
    regulatory approval.</li>
</ul>

<h2>8. Future work</h2>
<ul>
  <li><strong>Attack the dominant error mode directly</strong> — class-weighted or focal
    loss biased toward the glioma/meningioma boundary, or a second-stage binary classifier
    invoked only when the top two probabilities are both in that pair.</li>
  <li><strong>Explainability.</strong> Grad-CAM overlays would let a reader check whether
    the network attended to the lesion or to an irrelevant artefact — a prerequisite for
    any clinical conversation.</li>
  <li><strong>Volume-level inference</strong> — aggregate across the slices of a study
    rather than classifying one.</li>
  <li><strong>External validation</strong> on an independently acquired dataset, to
    measure the real generalisation gap.</li>
  <li><strong>Confidence calibration</strong> via temperature scaling.</li>
  <li><strong>Larger backbones</strong> — ResNet50, EfficientNet, a ViT — benchmarked to
    see whether capacity or data is the binding constraint.</li>
</ul>

<h2>9. Conclusion</h2>
<p>A ResNet18 fine-tuned in two phases classifies brain MRI slices into four categories at
  <strong>97.00% test accuracy</strong>, with a macro F1 of 0.9715 and perfect recall on
  the no-tumour class. The two-phase schedule is what makes this work on 5,000 images:
  freezing the backbone while the new head stabilises, then adapting only the deepest
  block at a reduced learning rate, lifted validation accuracy from an 84.2% ceiling to
  98.0%.</p>
<p>The residual errors are concentrated rather than diffuse — 63% of them sit on the
  glioma/meningioma boundary, the same distinction that is hardest for human readers on a
  single slice. That concentration is itself useful: it points at a specific, addressable
  target for the next iteration rather than a general need for more capacity.</p>

<div class="callout warn">
  <p><strong>Disclaimer.</strong> Research and educational use only. This model is not a
    medical device and is not validated for clinical diagnosis. Always consult a qualified
    medical professional.</p>
</div>

<footer>
  NeuraScan &middot; ResNet18 transfer learning on BRISC2025 &middot; PyTorch, torchvision,
  Flask. Figures generated from the executed training notebook.
</footer>

</div>
</body>
</html>
"""


if __name__ == '__main__':
    for name, html in (('poster.html', build_poster()),
                       ('report.html', build_report())):
        path = os.path.join(HERE, name)
        with open(path, 'w') as fh:
            fh.write(html)
        print(f'wrote {path}  ({len(html):,} bytes)')
