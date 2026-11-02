#!/usr/bin/env python3
"""Generate pmaf.ico for the Windows launcher. Stdlib only.

Design: dark night-sky rounded square with a green->teal aurora arc.
Rendered analytically (rounded-rect SDF + distance to a quadratic bezier)
at each icon size with 4x4 supersampling — no resampling artifacts.

Entries: 256 px (PNG-compressed) + 48/32/16 px (classic BGRA DIBs).
"""
import math
import struct
import sys
import zlib
from pathlib import Path

# unit-square coordinates, y down
BEZ = ((0.10, 0.74), (0.42, 0.20), (0.92, 0.44))   # aurora arc control points
BG_TOP, BG_BOT = (16, 22, 46), (5, 8, 19)          # night-sky gradient
CORE_A, CORE_B = (64, 255, 160), (70, 205, 255)    # green -> teal along the arc
STARS = ((0.22, 0.30), (0.78, 0.18), (0.64, 0.68), (0.30, 0.55), (0.85, 0.80))


def bezier(t):
    (x0, y0), (x1, y1), (x2, y2) = BEZ
    u = 1.0 - t
    return (u * u * x0 + 2 * u * t * x1 + t * t * x2,
            u * u * y0 + 2 * u * t * y1 + t * t * y2)


BEZ_PTS = [bezier(i / 63.0) for i in range(64)]


def arc_dist(x, y):
    """(distance to arc, parameter t of nearest point)"""
    best, bt = 10.0, 0.0
    for i, (bx, by) in enumerate(BEZ_PTS):
        d = math.hypot(x - bx, y - by)
        if d < best:
            best, bt = d, i / 63.0
    return best, bt


def shade(x, y, px):
    """RGBA (0..255 floats) for unit coords x,y; px = pixel size in units."""
    # rounded-square signed distance
    r = 0.16
    qx = abs(x - 0.5) - (0.5 - r)
    qy = abs(y - 0.5) - (0.5 - r)
    d = math.hypot(max(qx, 0.0), max(qy, 0.0)) - r
    alpha = min(max(0.5 - d / px, 0.0), 1.0)   # 1-px anti-aliased edge
    if alpha == 0.0:
        return 0.0, 0.0, 0.0, 0.0

    # background gradient
    g = min(max(y, 0.0), 1.0)
    rr = BG_TOP[0] + (BG_BOT[0] - BG_TOP[0]) * g
    gg = BG_TOP[1] + (BG_BOT[1] - BG_TOP[1]) * g
    bb = BG_TOP[2] + (BG_BOT[2] - BG_TOP[2]) * g

    # stars (skip at tiny sizes where they would just be noise)
    if px < 1 / 40.0:
        for sx, sy in STARS:
            sd = math.hypot(x - sx, y - sy)
            s = math.exp(-(sd / 0.008) ** 2) * 180
            rr += s; gg += s; bb += s

    # aurora arc: bright core + wide soft glow, green fading to teal
    dist, t = arc_dist(x, y)
    core = math.exp(-(dist / 0.075) ** 2)
    glow = 0.35 * math.exp(-(dist / 0.22) ** 2)
    cr = CORE_A[0] + (CORE_B[0] - CORE_A[0]) * t
    cg = CORE_A[1] + (CORE_B[1] - CORE_A[1]) * t
    cb = CORE_A[2] + (CORE_B[2] - CORE_A[2]) * t
    w = core + glow
    rr += cr * w
    gg += cg * w
    bb += cb * w

    clamp = lambda v: min(max(v, 0.0), 255.0)
    return clamp(rr), clamp(gg), clamp(bb), alpha * 255.0


def render(size):
    """-> list of rows, each row a list of (r,g,b,a) ints, top-down."""
    ss = 4
    px = 1.0 / size
    rows = []
    for j in range(size):
        row = []
        for i in range(size):
            ar = ag = ab = aa = 0.0
            for sj in range(ss):
                for si in range(ss):
                    x = (i + (si + 0.5) / ss) / size
                    y = (j + (sj + 0.5) / ss) / size
                    r, g, b, a = shade(x, y, px)
                    ar += r * a; ag += g * a; ab += b * a; aa += a
            n = ss * ss
            if aa > 0:
                row.append((round(ar / aa), round(ag / aa), round(ab / aa),
                            round(aa / n)))
            else:
                row.append((0, 0, 0, 0))
        rows.append(row)
    return rows


def to_png(rows):
    size = len(rows)
    raw = b"".join(b"\x00" + bytes(v for p in row for v in p) for row in rows)
    def chunk(tag, data):
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c))
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9))
            + chunk(b"IEND", b""))


def to_dib(rows):
    """BGRA 32-bpp bottom-up DIB + all-zero AND mask (alpha channel rules)."""
    size = len(rows)
    hdr = struct.pack("<IiiHHIIiiII", 40, size, size * 2, 1, 32, 0,
                      size * size * 4, 0, 0, 0, 0)
    xor = b"".join(bytes(v for (r, g, b, a) in row for v in (b, g, r, a))
                   for row in reversed(rows))
    mask_stride = ((size + 31) // 32) * 4
    return hdr + xor + b"\x00" * (mask_stride * size)


def main():
    out = Path(__file__).parent / "pmaf.ico"
    sizes = (256, 48, 32, 16)
    images = []
    for s in sizes:
        rows = render(s)
        images.append(to_png(rows) if s == 256 else to_dib(rows))
        print(f"  {s}x{s}: {len(images[-1])} bytes")
    header = struct.pack("<HHH", 0, 1, len(sizes))
    offset = len(header) + 16 * len(sizes)
    entries = b""
    for s, img in zip(sizes, images):
        entries += struct.pack("<BBBBHHII", s % 256, s % 256, 0, 0, 1, 32,
                               len(img), offset)
        offset += len(img)
    out.write_bytes(header + entries + b"".join(images))
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    sys.exit(main())
