#!/usr/bin/env python3
"""Minimal stdlib PNG writer/reader used by this package.

Writer: 8-bit RGB, 8-bit grayscale, 16-bit grayscale (big-endian, per spec).
Reader: handles non-interlaced PNGs with bit depth 8/16, gray or RGB, all
five scanline filters — sufficient for files written here and by ffmpeg.
"""

import struct
import zlib

_SIG = b"\x89PNG\r\n\x1a\n"


def _chunk(tag, data):
    c = struct.pack(">I", len(data)) + tag + data
    return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)


def write_png(path, width, height, mode, data):
    """mode: 'L8' | 'L16' | 'RGB8'.
    data: bytes/bytearray — L8: w*h; L16: w*h uint16 values as big-endian
    2*w*h bytes; RGB8: 3*w*h."""
    if mode == "L8":
        color, depth, bpp = 0, 8, 1
    elif mode == "L16":
        color, depth, bpp = 0, 16, 2
    elif mode == "RGB8":
        color, depth, bpp = 2, 8, 3
    else:
        raise ValueError(mode)
    stride = width * bpp
    if len(data) != stride * height:
        raise ValueError(f"bad data length {len(data)} != {stride * height}")
    raw = bytearray()
    for y in range(height):
        raw.append(0)  # filter type 0 (None)
        raw += data[y * stride:(y + 1) * stride]
    ihdr = struct.pack(">IIBBBBB", width, height, depth, color, 0, 0, 0)
    with open(path, "wb") as f:
        f.write(_SIG)
        f.write(_chunk(b"IHDR", ihdr))
        f.write(_chunk(b"IDAT", zlib.compress(bytes(raw), 6)))
        f.write(_chunk(b"IEND", b""))


def read_png(path):
    """Returns (width, height, mode, data) with the same conventions as
    write_png.  Raises ValueError on unsupported formats."""
    with open(path, "rb") as f:
        blob = f.read()
    if blob[:8] != _SIG:
        raise ValueError("not a PNG")
    pos = 8
    width = height = depth = color = None
    idat = bytearray()
    while pos < len(blob):
        (ln,) = struct.unpack(">I", blob[pos:pos + 4])
        tag = blob[pos + 4:pos + 8]
        data = blob[pos + 8:pos + 8 + ln]
        pos += 12 + ln
        if tag == b"IHDR":
            width, height, depth, color, comp, filt, inter = \
                struct.unpack(">IIBBBBB", data)
            if inter != 0:
                raise ValueError("interlaced PNG unsupported")
            if color not in (0, 2) or depth not in (8, 16):
                raise ValueError(f"unsupported color={color} depth={depth}")
        elif tag == b"IDAT":
            idat += data
        elif tag == b"IEND":
            break
    channels = 1 if color == 0 else 3
    bpp = channels * (depth // 8)
    stride = width * bpp
    raw = zlib.decompress(bytes(idat))
    out = bytearray(stride * height)
    prev = bytearray(stride)
    p = 0
    for y in range(height):
        ftype = raw[p]
        p += 1
        line = bytearray(raw[p:p + stride])
        p += stride
        if ftype == 1:      # Sub
            for i in range(bpp, stride):
                line[i] = (line[i] + line[i - bpp]) & 0xFF
        elif ftype == 2:    # Up
            for i in range(stride):
                line[i] = (line[i] + prev[i]) & 0xFF
        elif ftype == 3:    # Average
            for i in range(stride):
                a = line[i - bpp] if i >= bpp else 0
                line[i] = (line[i] + ((a + prev[i]) >> 1)) & 0xFF
        elif ftype == 4:    # Paeth
            for i in range(stride):
                a = line[i - bpp] if i >= bpp else 0
                b = prev[i]
                c = prev[i - bpp] if i >= bpp else 0
                pa, pb, pc = abs(b - c), abs(a - c), abs(a + b - 2 * c)
                pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                line[i] = (line[i] + pr) & 0xFF
        elif ftype != 0:
            raise ValueError(f"bad filter {ftype}")
        out[y * stride:(y + 1) * stride] = line
        prev = line
    mode = {(1, 8): "L8", (1, 16): "L16", (3, 8): "RGB8"}[(channels, depth)]
    return width, height, mode, bytes(out)


def u16_grid_to_bytes(values, width, height):
    """values: iterable of ints (row-major) -> big-endian uint16 bytes."""
    return struct.pack(f">{width * height}H", *values)


def bytes_to_u16_grid(data, width, height):
    return list(struct.unpack(f">{width * height}H", data))
