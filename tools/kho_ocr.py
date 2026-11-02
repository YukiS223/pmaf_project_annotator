#!/usr/bin/env python3
"""Per-frame UT timestamp OCR for KHO all-sky AVI/MP4 clips.

Reads the burned-in HH:MM:SS clock (top-right, fixed bitmap font) from every
frame and returns per-frame times with confidences.  Two template banks cover
the two font weights seen in the archive (thin ~2017, bold ~2024); the bank is
chosen per clip by aggregate match score.

Pure stdlib.  If numpy is importable it is used transparently for speed, but
results are identical without it.  Validated on 2700 real frames across both
eras (zero read errors, including a mid-clip midnight rollover).
"""

import base64
import gzip
import json

ALGO_VERSION = 1

# Strip geometry (validated): x 504..608, y 6..24 of the 608x608 frame.
STRIP_X, STRIP_Y, STRIP_W, STRIP_H = 504, 6, 104, 18
CELL_W, CELL_H = 16, 22
DIGIT_GLYPHS = (0, 1, 3, 4, 6, 7)   # glyph indices of HHMMSS among the 10 groups
SEG_THRESHOLDS = (100, 110, 120, 130, 140, 150, 160, 90, 80, 70)
CONF_ACCEPT = 0.85    # greedy accept threshold for the expected next digit
CONF_LOW = 0.60       # below this a read is flagged low-confidence

_TEMPLATES_B64 = "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"

try:
    import numpy as _np
except Exception:      # numpy is optional everywhere in this package
    _np = None


def _load_banks():
    raw = json.loads(gzip.decompress(base64.b64decode(_TEMPLATES_B64)))
    banks = {}
    for era, digits in raw.items():
        bank = {}
        for d, t in digits.items():
            data = base64.b64decode(t["data"])
            px = [b / 255.0 for b in data]
            bank[d] = _Template(px, t["w"], t["h"])
        banks[era] = bank
    return banks


class _Template:
    __slots__ = ("px", "w", "h", "mean", "var")

    def __init__(self, px, w, h):
        self.px, self.w, self.h = px, w, h
        n = len(px)
        s = sum(px)
        self.mean = s / n
        self.var = sum(v * v for v in px) - s * s / n


def segment(strip):
    """strip: list of STRIP_H rows, each a bytes/bytearray of STRIP_W.
    Returns list of 10 (x0, x1) glyph column groups, or None."""
    for thr in SEG_THRESHOLDS:
        lit = [False] * STRIP_W
        for row in strip:
            for x in range(STRIP_W):
                if row[x] > thr:
                    lit[x] = True
        groups, s = [], None
        for x in range(STRIP_W):
            if lit[x] and s is None:
                s = x
            elif not lit[x] and s is not None:
                groups.append((s, x))
                s = None
        if s is not None:
            groups.append((s, STRIP_W))
        if len(groups) == 10:
            return groups
    return None


def cut_cell(strip, group):
    """Fixed vertical band (baseline preserved), horizontally centered cell."""
    x0, x1 = group
    w = min(x1 - x0, CELL_W)
    h = min(STRIP_H, CELL_H)
    cell = [0.0] * (CELL_W * CELL_H)
    y_off = (CELL_H - h) // 2
    x_off = (CELL_W - w) // 2
    mx = 0
    for y in range(h):
        row = strip[y]
        base = (y + y_off) * CELL_W + x_off
        for x in range(w):
            v = row[x0 + x]
            cell[base + x] = v
            if v > mx:
                mx = v
    if mx > 0:
        inv = 1.0 / mx
        cell = [v * inv for v in cell]
    return cell


def _ncc_at(cell, tmpl, dy, dx):
    """NCC between cell (CELL_HxCELL_W list) and template at integer offset."""
    tw, th = tmpl.w, tmpl.h
    # overlapping window of the template inside the cell, shifted by (dy,dx)
    sa = saa = sab = sb = sbb = 0.0
    n = 0
    tpx = tmpl.px
    for y in range(th):
        cy = y + dy
        if cy < 0 or cy >= CELL_H:
            continue
        trow = y * tw
        crow = cy * CELL_W
        for x in range(tw):
            cx = x + dx
            if cx < 0 or cx >= CELL_W:
                continue
            a = cell[crow + cx]
            b = tpx[trow + x]
            sa += a
            saa += a * a
            sb += b
            sbb += b * b
            sab += a * b
            n += 1
    if n == 0:
        return 0.0
    cov = sab - sa * sb / n
    va = saa - sa * sa / n
    vb = sbb - sb * sb / n
    if va <= 0 or vb <= 0:
        return 0.0
    return cov / (va * vb) ** 0.5


if _np is not None:
    def _ncc_at(cell, tmpl, dy, dx):  # noqa: F811 (numpy fast path)
        c = _np.asarray(cell).reshape(CELL_H, CELL_W)
        t = _np.asarray(tmpl.px).reshape(tmpl.h, tmpl.w)
        cy0, cy1 = max(0, dy), min(CELL_H, tmpl.h + dy)
        cx0, cx1 = max(0, dx), min(CELL_W, tmpl.w + dx)
        ty0, tx0 = cy0 - dy, cx0 - dx
        a = c[cy0:cy1, cx0:cx1].ravel()
        b = t[ty0:ty0 + (cy1 - cy0), tx0:tx0 + (cx1 - cx0)].ravel()
        if a.size == 0:
            return 0.0
        a = a - a.mean()
        b = b - b.mean()
        d = float(_np.linalg.norm(a) * _np.linalg.norm(b))
        return float((a * b).sum() / d) if d > 0 else 0.0


def ncc_shift(cell, tmpl, r=2):
    best = -1.0
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            s = _ncc_at(cell, tmpl, dy, dx)
            if s > best:
                best = s
    return best


def classify_cell(cell, bank, expected=None, r=2):
    """Returns (digit, confidence). Tries `expected` first (greedy accept)."""
    if expected is not None:
        s = ncc_shift(cell, bank[expected], r)
        if s >= CONF_ACCEPT:
            return expected, s
    best, best_d = -1.0, "?"
    for d, t in bank.items():
        s = ncc_shift(cell, t, r)
        if s > best:
            best, best_d = s, d
    return best_d, best


def _sad(strip_a, strip_b, group):
    x0, x1 = group
    total = 0
    for y in range(STRIP_H):
        ra, rb = strip_a[y], strip_b[y]
        for x in range(x0, x1):
            total += abs(ra[x] - rb[x])
    return total / ((x1 - x0) * STRIP_H)


def read_clip(strips, progress=None):
    """strips: sequence of frames, each a list of STRIP_H rows (bytes of STRIP_W).
    Returns dict with per-frame results and the chosen era bank."""
    banks = _load_banks()

    # Choose bank on a spread of sample frames by mean full-search confidence.
    n = len(strips)
    sample_idx = [int(i * (n - 1) / 9) for i in range(10)] if n > 1 else [0]
    era_scores = {}
    for era, bank in banks.items():
        confs = []
        for i in sample_idx:
            groups = segment(strips[i])
            if not groups:
                continue
            cs = [classify_cell(cut_cell(strips[i], groups[g]), bank, None, 2)[1]
                  for g in DIGIT_GLYPHS]
            confs.append(min(cs))
        era_scores[era] = sum(confs) / len(confs) if confs else 0.0
    era = max(era_scores, key=era_scores.get)
    bank = banks[era]

    results = []          # per frame: dict(text, conf, method) or None
    prev_strip = None
    prev_groups = None
    prev_digits = None    # list of 6 chars
    prev_confs = None
    for i in range(n):
        strip = strips[i]
        groups = segment(strip)
        if groups is None:
            results.append(None)
            prev_strip = prev_groups = prev_digits = prev_confs = None
            if progress:
                progress(i, n)
            continue
        digits = [None] * 6
        confs = [0.0] * 6
        reuse = (prev_strip is not None and prev_groups is not None
                 and len(prev_groups) == 10)
        for k, g in enumerate(DIGIT_GLYPHS):
            grp = groups[g]
            if reuse and prev_digits and prev_groups[g] == grp \
                    and _sad(strip, prev_strip, grp) < 6.0:
                digits[k] = prev_digits[k]
                confs[k] = prev_confs[k]
                continue
            expected = None
            if prev_digits is not None and k == 5:
                expected = str((int(prev_digits[5]) + 1) % 10)
            cell = cut_cell(strip, grp)
            digits[k], confs[k] = classify_cell(cell, bank, expected, 2)
        results.append({"text": "".join(digits), "conf": min(confs)})
        prev_strip, prev_groups = strip, groups
        prev_digits, prev_confs = digits, confs
        if progress:
            progress(i, n)

    return {"era": era, "era_scores": era_scores, "frames": results}


def to_seconds(text):
    return int(text[:2]) * 3600 + int(text[2:4]) * 60 + int(text[4:])


def repair_times(results, strips, progress=None):
    """Monotonicity validation & repair.  Marks each frame with method:
    'ocr' (confident, consistent), 'interp' (interpolated between confident
    neighbours), or 'nominal' (no anchor).  Returns list of
    dict(seconds, text, conf, method) aligned with frames; seconds may exceed
    86400 across midnight rollovers (monotonic, not wrapped)."""
    banks = _load_banks()
    bank = banks[results["era"]]
    frames = results["frames"]
    n = len(frames)

    # Unwrap to monotonic seconds (rollover adds 86400).
    raw = [None] * n
    for i, f in enumerate(frames):
        if f is not None and f["conf"] >= CONF_LOW:
            raw[i] = to_seconds(f["text"])

    # Detect suspects.  True cadence is ~1.0145 s/frame and the stamp has 1 s
    # resolution, so over a gap of g frames the stamp delta must lie in
    # [g, g*1.1 + 2].  A delta of 0 (stuck/duplicated clock) or an off-model
    # jump is always wrong even when monotonicity holds.
    def unwrap(vals):
        out, offset, prev = [None] * n, 0, None
        for i, v in enumerate(vals):
            if v is None:
                out[i] = None
                continue
            if prev is not None and v + offset < prev - 43200:
                offset += 86400
            out[i] = v + offset
            prev = out[i]
        return out

    mono = unwrap(raw)
    suspects = set()
    last = None
    for i in range(n):
        if mono[i] is None:
            continue
        if last is not None:
            gap = i - last[0]
            d = mono[i] - last[1]
            if d < gap or d > gap * 1.1 + 2:
                suspects.add(i)
                suspects.add(last[0])
                continue
        last = (i, mono[i])

    # Full re-OCR (no greedy shortcut, wider shift) for suspects.
    for i in sorted(suspects):
        if strips[i] is None:      # no pixel data to re-verify: keep the read
            continue
        groups = segment(strips[i])
        if groups is None:
            raw[i] = None
            continue
        digits, confs = [], []
        for g in DIGIT_GLYPHS:
            d, c = classify_cell(cut_cell(strips[i], groups[g]), bank, None, 3)
            digits.append(d)
            confs.append(c)
        conf = min(confs)
        frames[i] = {"text": "".join(digits), "conf": conf}
        raw[i] = to_seconds(frames[i]["text"]) if conf >= CONF_LOW else None
        if progress:
            progress(i, n)

    mono = unwrap(raw)
    # Final pass: anything still violating monotonicity vs its neighbours is
    # dropped to None (will be interpolated).
    good = [i for i in range(n) if mono[i] is not None]
    keep = set()
    prev_i = None
    for i in good:
        if prev_i is None:
            keep.add(i)
            prev_i = i
            continue
        gap = i - prev_i
        d = mono[i] - mono[prev_i]
        if d >= gap and d <= gap * 1.1 + 2:
            keep.add(i)
            prev_i = i
        elif d > gap * 1.1 + 2 and frames[i]["conf"] >= 0.95:
            # fast jump confirmed by full re-OCR: a real capture pause —
            # re-anchor here rather than smearing the gap by interpolation
            keep.add(i)
            prev_i = i
        # d < gap (duplicate/backwards second) is physically impossible at
        # ~1.0145 s cadence: drop, will be interpolated between anchors
    anchors = [i for i in good if i in keep]

    out = []
    for i in range(n):
        if i in keep:
            out.append({"seconds": mono[i], "text": frames[i]["text"],
                        "conf": frames[i]["conf"], "method": "ocr"})
        else:
            out.append(None)

    # Interpolate gaps between anchors; extrapolate ends at mean cadence.
    if anchors:
        mean_cad = 1.0145
        if len(anchors) > 1:
            span = mono[anchors[-1]] - mono[anchors[0]]
            mean_cad = span / (anchors[-1] - anchors[0])
        for i in range(n):
            if out[i] is not None:
                continue
            lo = max((a for a in anchors if a < i), default=None)
            hi = min((a for a in anchors if a > i), default=None)
            if lo is not None and hi is not None:
                sec = mono[lo] + (mono[hi] - mono[lo]) * (i - lo) / (hi - lo)
            elif lo is not None:
                sec = mono[lo] + (i - lo) * mean_cad
            elif hi is not None:
                sec = mono[hi] - (hi - i) * mean_cad
            else:
                sec = i * mean_cad
            method = "interp" if (lo is not None or hi is not None) else "nominal"
            f = frames[i]
            out[i] = {"seconds": sec, "text": f["text"] if f else None,
                      "conf": f["conf"] if f else 0.0, "method": method}
    else:
        for i in range(n):
            out[i] = {"seconds": i * 1.0145, "text": None, "conf": 0.0,
                      "method": "nominal"}
    return out
