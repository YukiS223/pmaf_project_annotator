#!/usr/bin/env python3
"""Canonical mask RLE codec shared by the annotator and exporters.

Format (string, JSON-safe):  "<w>x<h>:<runs>"
  runs = '.'-separated base-36 run lengths, row-major (x fastest),
  alternating background,foreground,background,...  The first run is
  background and may be 0.  Run lengths must sum to w*h.

The JS implementation in app/index.html mirrors this exactly.
"""


def encode(mask, width, height):
    """mask: iterable of 0/1 ints, row-major, length w*h -> RLE string."""
    runs = []
    cur_val = 0
    cur_len = 0
    for v in mask:
        v = 1 if v else 0
        if v == cur_val:
            cur_len += 1
        else:
            runs.append(cur_len)
            cur_val = v
            cur_len = 1
    runs.append(cur_len)
    total = sum(runs)
    if total != width * height:
        raise ValueError(f"mask length {total} != {width}x{height}")
    return f"{width}x{height}:" + ".".join(_b36(r) for r in runs)


def decode(rle):
    """RLE string -> (mask bytearray of 0/1, width, height)."""
    head, _, body = rle.partition(":")
    w, _, h = head.partition("x")
    width, height = int(w), int(h)
    mask = bytearray(width * height)
    pos = 0
    val = 0
    if body:
        for tok in body.split("."):
            # strict base-36: int(tok, 36) alone would accept '-5', '+5',
            # '1_0' and whitespace, silently shifting every later run
            if not tok or any(c not in _B36 for c in tok):
                raise ValueError(f"bad RLE token {tok!r}")
            n = int(tok, 36)
            if val:
                for i in range(pos, pos + n):
                    mask[i] = 1
            pos += n
            val ^= 1
    if pos != width * height:
        raise ValueError(f"RLE runs sum {pos} != {width}x{height}")
    return mask, width, height


def to_coco_counts(mask, width, height):
    """Row-major 0/1 mask -> COCO uncompressed counts (column-major,
    background first)."""
    counts = []
    cur_val = 0
    cur_len = 0
    for x in range(width):
        for y in range(height):
            v = mask[y * width + x]
            if v == cur_val:
                cur_len += 1
            else:
                counts.append(cur_len)
                cur_val = v
                cur_len = 1
    counts.append(cur_len)
    return counts


def from_coco_counts(counts, width, height):
    """COCO uncompressed counts -> row-major 0/1 mask bytearray."""
    mask = bytearray(width * height)
    pos = 0
    val = 0
    for n in counts:
        if val:
            for k in range(pos, pos + n):
                x, y = divmod(k, height)
                mask[y * width + x] = 1
        pos += n
        val ^= 1
    if pos != width * height:
        raise ValueError("counts do not sum to w*h")
    return mask


def bbox_area(mask, width, height):
    """Returns ([x, y, bw, bh], area) of a row-major 0/1 mask; bbox is
    [0,0,0,0] for an empty mask."""
    x0 = y0 = None
    x1 = y1 = -1
    area = 0
    for y in range(height):
        base = y * width
        for x in range(width):
            if mask[base + x]:
                area += 1
                if x0 is None or x < x0:
                    x0 = x
                if x1 < x:
                    x1 = x
                if y0 is None:
                    y0 = y
                y1 = y
    if area == 0:
        return [0, 0, 0, 0], 0
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1], area


_B36 = "0123456789abcdefghijklmnopqrstuvwxyz"


def _b36(n):
    if n == 0:
        return "0"
    out = []
    while n:
        n, r = divmod(n, 36)
        out.append(_B36[r])
    return "".join(reversed(out))
