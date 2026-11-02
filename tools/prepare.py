#!/usr/bin/env python3
"""Project preparation: video -> frames + timing + backgrounds + manifest.

Steps (all stdlib, ffmpeg does the decoding):
  1. extract   — every frame to frames/%06d.png (lossless, authoritative)
  2. ocr       — per-frame UT from the burned-in clock (tools/kho_ocr.py)
  3. background— per-pixel median of 9 raw frames for each of 6 time segments
  4. circle    — sky-disk (cx, cy, r) from the mid background
  5. manifest  — manifest.json with provenance, timing, hashes

Used as a library by serve.py and as a CLI:
  python3 tools/prepare.py <video> [--name PROJECT] [--projects-dir DIR]
"""

import hashlib
import json
import os
import platform
import re
import shutil
import statistics
import struct
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kho_ocr  # noqa: E402
import pngio    # noqa: E402

TOOL_VERSION = "2.0.0"
SCHEMA_VERSION = 2
BG_SEGMENTS = 6
BG_SAMPLES = 9
PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VIDEO_EXTS = (".avi", ".mp4", ".mov", ".mkv")


def find_ffmpeg():
    """Bundled binary first, then PATH."""
    sysname = platform.system()
    arch = platform.machine().lower()
    cands = []
    if sysname == "Darwin" and arch in ("arm64", "aarch64"):
        cands.append(os.path.join(PKG_ROOT, "bin", "ffmpeg-macos-arm64"))
    elif sysname == "Windows":
        cands.append(os.path.join(PKG_ROOT, "bin", "ffmpeg-windows-x64.exe"))
    for c in cands:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    found = shutil.which("ffmpeg")
    if found:
        return found
    raise RuntimeError(
        "No usable ffmpeg: bundled binary missing for this platform "
        f"({sysname}/{arch}) and none on PATH. See INSTRUCTIONS.pdf, "
        "'Troubleshooting'.")


def ffmpeg_version(ffmpeg):
    try:
        out = subprocess.run([ffmpeg, "-version"], capture_output=True,
                             text=True, timeout=15).stdout
        return out.splitlines()[0].strip() if out else "unknown"
    except Exception:
        return "unknown"


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def png_size(path):
    """Fast width/height from IHDR."""
    with open(path, "rb") as f:
        head = f.read(33)
    if head[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path}: not a PNG")
    w, h = struct.unpack(">II", head[16:24])
    return w, h


def extract_frames(ffmpeg, src, frames_dir, progress=None):
    os.makedirs(frames_dir, exist_ok=True)
    # A previous failed/partial run may have left frames behind; a shorter new
    # video would otherwise interleave with stale ones.
    for stale in os.listdir(frames_dir):
        if re.fullmatch(r"\d{6}\.png", stale):
            os.remove(os.path.join(frames_dir, stale))
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", src,
           "-vsync", "0", "-start_number", "0",
           "-progress", "pipe:1",
           os.path.join(frames_dir, "%06d.png"), "-y"]
    # stderr goes to a file: a PIPE we don't drain while looping over stdout
    # can fill and deadlock ffmpeg if it gets chatty
    import tempfile as _tf
    with _tf.TemporaryFile(mode="w+") as errf:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=errf,
                                text=True)
        for line in proc.stdout:
            m = re.match(r"frame=(\d+)", line.strip())
            if m and progress:
                progress(int(m.group(1)))
        proc.wait()
        if proc.returncode != 0:
            errf.seek(0)
            err = errf.read()[-2000:]
            raise RuntimeError(f"ffmpeg frame extraction failed: {err}")
    names = sorted(n for n in os.listdir(frames_dir)
                   if re.fullmatch(r"\d{6}\.png", n))
    if not names:
        raise RuntimeError("ffmpeg produced no frames")
    # Frames must be a dense 0..n-1 sequence.
    for i, n in enumerate(names):
        if int(n[:6]) != i:
            raise RuntimeError(f"frame index gap at {n} (expected {i:06d}.png)")
    return names


def read_strips(ffmpeg, src, nframes):
    K = kho_ocr
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", src,
           "-vf", f"crop={K.STRIP_W}:{K.STRIP_H}:{K.STRIP_X}:{K.STRIP_Y},format=gray",
           "-vsync", "0", "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    fsz = K.STRIP_W * K.STRIP_H
    n = len(raw) // fsz
    if n != nframes:
        raise RuntimeError(f"OCR pass decoded {n} frames, extraction saw {nframes}")
    return [[raw[f * fsz + y * K.STRIP_W:f * fsz + (y + 1) * K.STRIP_W]
             for y in range(K.STRIP_H)] for f in range(n)]


def date_base_from_name(filename):
    m = re.search(r"(\d{2})(\d{2})(\d{4})_(\d{2})(\d{2})(\d{2})", filename)
    if not m:
        return None, None
    dd, mm, yyyy, H, M, S = (int(g) for g in m.groups())
    try:
        return (datetime(yyyy, mm, dd, tzinfo=timezone.utc).date().isoformat(),
                H * 3600 + M * 60 + S)
    except ValueError:
        return None, None


def compute_timing(ffmpeg, src, nframes, filename, progress=None):
    strips = read_strips(ffmpeg, src, nframes)
    res = kho_ocr.read_clip(strips, progress=progress)
    times = kho_ocr.repair_times(res, strips)
    date_base, name_start = date_base_from_name(filename)
    frames = []
    rollovers = []
    for i, t in enumerate(times):
        rec = {"seconds": round(t["seconds"], 3), "conf": round(t["conf"], 4),
               "method": t["method"]}
        if t["text"]:
            rec["ut"] = f'{t["text"][:2]}:{t["text"][2:4]}:{t["text"][4:]}'
        if date_base:
            dt = (datetime.fromisoformat(date_base + "T00:00:00+00:00")
                  + timedelta(seconds=t["seconds"]))
            rec["iso"] = dt.isoformat().replace("+00:00", "Z")
        frames.append(rec)
        if i > 0 and times[i - 1]["seconds"] < 86400 <= times[i]["seconds"]:
            rollovers.append(i)
    warn = None
    if name_start is not None and frames and frames[0]["method"] == "ocr":
        drift = abs(frames[0]["seconds"] - name_start)
        if drift > 120:
            warn = (f"filename start time differs from first OCR time by "
                    f"{drift:.0f}s")
    counts = {}
    for f in frames:
        counts[f["method"]] = counts.get(f["method"], 0) + 1
    return {"era": res["era"], "frames": frames, "rollovers": rollovers,
            "date_base": date_base, "method_counts": counts,
            "ocr_algo_version": kho_ocr.ALGO_VERSION, "warning": warn}


def bg_sample_indices(nframes):
    seg_len = nframes / BG_SEGMENTS
    segs = []
    for s in range(BG_SEGMENTS):
        idxs = [min(nframes - 1, int(s * seg_len + (k + 0.5) * seg_len / BG_SAMPLES))
                for k in range(BG_SAMPLES)]
        segs.append(sorted(set(idxs)))
    return segs


def compute_backgrounds(ffmpeg, src, nframes, width, height, out_dir,
                        progress=None):
    """Per-segment per-pixel median over raw RGB samples pulled via ffmpeg."""
    os.makedirs(out_dir, exist_ok=True)
    segs = bg_sample_indices(nframes)
    all_idx = sorted({i for seg in segs for i in seg})
    sel = "+".join(f"eq(n\\,{i})" for i in all_idx)
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", src,
           "-vf", f"select='{sel}',format=rgb24", "-vsync", "0",
           "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    fsz = width * height * 3
    if len(raw) != fsz * len(all_idx):
        raise RuntimeError(
            f"background pass decoded {len(raw) // fsz} frames, "
            f"expected {len(all_idx)}")
    frame_at = {idx: raw[k * fsz:(k + 1) * fsz]
                for k, idx in enumerate(all_idx)}
    try:
        import numpy as np
    except Exception:
        np = None
    out = []
    for s, seg in enumerate(segs):
        bufs = [frame_at[i] for i in seg]
        if np is not None:
            stack = np.stack([np.frombuffer(b, np.uint8) for b in bufs])
            med = np.median(stack, axis=0).astype(np.uint8).tobytes()
        else:
            m = len(bufs)
            half = m // 2
            med_arr = bytearray(fsz)
            cols = [bytearray(m) for _ in range(1)]  # reusable scratch
            scratch = bytearray(m)
            for p in range(fsz):
                for j in range(m):
                    scratch[j] = bufs[j][p]
                med_arr[p] = sorted(scratch)[half]
            med = bytes(med_arr)
        fn = f"bg_{s}.png"
        pngio.write_png(os.path.join(out_dir, fn), width, height, "RGB8", med)
        out.append({"file": fn, "samples": seg,
                    "sha256": hashlib.sha256(med).hexdigest()})
        if progress:
            progress(s + 1, BG_SEGMENTS)
    return {"algo_version": 1, "segments_count": BG_SEGMENTS,
            "samples_per_segment": BG_SAMPLES, "segments": out}


def detect_circle(bg_png_path, width, height):
    """Sky-disk estimate from a background image: threshold between corner
    dark level and disk level, then robust bbox of the bright region."""
    w, h, mode, data = pngio.read_png(bg_png_path)
    lum = bytearray(w * h)
    for p in range(w * h):
        r, g, b = data[3 * p], data[3 * p + 1], data[3 * p + 2]
        lum[p] = (r * 77 + g * 150 + b * 29) >> 8
    pk = 40
    corners = []
    for cy in (0, h - pk):
        for cx in (0, w - pk):
            corners += [lum[(cy + y) * w + (cx + x)]
                        for y in range(0, pk, 4) for x in range(0, pk, 4)]
    corner_level = statistics.median(corners)
    cx0, cy0 = w // 2, h // 2
    center = [lum[(cy0 + y) * w + (cx0 + x)]
              for y in range(-40, 40, 4) for x in range(-40, 40, 4)]
    center_level = statistics.median(center)
    if center_level <= corner_level + 8:
        return {"cx": w / 2, "cy": h / 2, "r": min(w, h) * 0.42,
                "method": "default"}
    # Flood fill from the image centre: burned-in text and compass labels are
    # bright too, but they are not connected to the sky disk, so the filled
    # component's bbox is the disk alone.
    thr = corner_level + 0.5 * (center_level - corner_level)
    seen = bytearray(w * h)
    start = (h // 2) * w + (w // 2)
    if lum[start] <= thr:
        return {"cx": w / 2, "cy": h / 2, "r": min(w, h) * 0.42,
                "method": "default"}
    stack = [start]
    seen[start] = 1
    x0, x1, y0, y1 = w, -1, h, -1
    while stack:
        p = stack.pop()
        y, x = divmod(p, w)
        if x < x0:
            x0 = x
        if x > x1:
            x1 = x
        if y < y0:
            y0 = y
        if y > y1:
            y1 = y
        if x > 0 and not seen[p - 1] and lum[p - 1] > thr:
            seen[p - 1] = 1
            stack.append(p - 1)
        if x < w - 1 and not seen[p + 1] and lum[p + 1] > thr:
            seen[p + 1] = 1
            stack.append(p + 1)
        if y > 0 and not seen[p - w] and lum[p - w] > thr:
            seen[p - w] = 1
            stack.append(p - w)
        if y < h - 1 and not seen[p + w] and lum[p + w] > thr:
            seen[p + w] = 1
            stack.append(p + w)
    if x1 - x0 < w * 0.3:   # implausibly small disk -> fall back
        return {"cx": w / 2, "cy": h / 2, "r": min(w, h) * 0.42,
                "method": "default"}
    return {"cx": round((x0 + x1) / 2, 1), "cy": round((y0 + y1) / 2, 1),
            "r": round(max(x1 - x0, y1 - y0) / 2, 1),
            "method": "flood_fill"}


def prepare_project(src_video, project_dir, progress=None, keep_source_name=None):
    """Full pipeline. progress(phase, done, total) if given."""
    def report(phase, done, total):
        if progress:
            progress(phase, done, total)

    t_start = time.time()
    ffmpeg = find_ffmpeg()
    os.makedirs(project_dir, exist_ok=True)
    frames_dir = os.path.join(project_dir, "frames")
    src_name = keep_source_name or os.path.basename(src_video)

    report("hash", 0, 1)
    src_sha = sha256_file(src_video)
    src_bytes = os.path.getsize(src_video)
    report("hash", 1, 1)

    report("extract", 0, 0)
    names = extract_frames(ffmpeg, src_video, frames_dir,
                           progress=lambda n: report("extract", n, 0))
    nframes = len(names)
    width, height = png_size(os.path.join(frames_dir, names[0]))
    report("extract", nframes, nframes)

    report("ocr", 0, nframes)
    timing = compute_timing(ffmpeg, src_video, nframes, src_name,
                            progress=lambda i, n: report("ocr", i + 1, n))

    report("background", 0, BG_SEGMENTS)
    bg_dir = os.path.join(project_dir, "backgrounds")
    backgrounds = compute_backgrounds(
        ffmpeg, src_video, nframes, width, height, bg_dir,
        progress=lambda d, t: report("background", d, t))

    report("circle", 0, 1)
    mid = os.path.join(bg_dir, backgrounds["segments"][BG_SEGMENTS // 2]["file"])
    sky = detect_circle(mid, width, height)
    report("circle", 1, 1)

    sizes = [os.path.getsize(os.path.join(frames_dir, n)) for n in names[:20]]
    # Dataset-split grouping key: clips from the same observation night are
    # correlated and must land in the same train/val split.  Night = UT date
    # of the clip start (a post-midnight clip belongs to the night that began
    # the previous UT day only if the PI decides so — date_base is objective
    # and recorded; regroup offline if needed).
    night_id = (f"KHO_{timing['date_base']}" if timing.get("date_base")
                else f"KHO_unknown_{src_sha[:12]}")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "tool_version": TOOL_VERSION,
        "project": os.path.basename(project_dir),
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "prepare_seconds": round(time.time() - t_start, 1),
        "source": {"filename": src_name, "sha256": src_sha,
                   "bytes": src_bytes},
        "video": {"width": width, "height": height, "nframes": nframes},
        "extract": {"ffmpeg": ffmpeg_version(ffmpeg),
                    "cmd": "-vsync 0 -start_number 0 frames/%06d.png",
                    "png_avg_bytes": int(statistics.mean(sizes))},
        "site": {"name": "KHO Kjell Henriksen Observatory",
                 "location": "N78.148 E16.043, Svalbard"},
        "night_id": night_id,
        "sky": sky,
        "timing": timing,
        "backgrounds": backgrounds,
    }
    mpath = os.path.join(project_dir, "manifest.json")
    with open(mpath + ".tmp", "w") as f:
        json.dump(manifest, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(mpath + ".tmp", mpath)   # never leave a partial manifest
    return manifest


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video")
    ap.add_argument("--name", default=None)
    ap.add_argument("--projects-dir",
                    default=os.path.join(PKG_ROOT, "projects"))
    args = ap.parse_args()
    if not os.path.isfile(args.video):
        sys.exit(f"no such file: {args.video}")
    name = args.name or re.sub(r"[^A-Za-z0-9_-]", "_",
                               os.path.splitext(os.path.basename(args.video))[0])
    project_dir = os.path.join(args.projects_dir, name)
    if os.path.exists(os.path.join(project_dir, "manifest.json")):
        sys.exit(f"project '{name}' already prepared ({project_dir})")

    last = {"phase": None}

    def prog(phase, done, total):
        if phase != last["phase"]:
            print(f"\n[{phase}]", end="", flush=True)
            last["phase"] = phase
        if total:
            print(f"\r[{phase}] {done}/{total}", end="", flush=True)
        elif done and done % 200 == 0:
            print(f"\r[{phase}] {done}", end="", flush=True)

    m = prepare_project(args.video, project_dir, progress=prog)
    print(f"\nprepared '{name}': {m['video']['nframes']} frames, "
          f"timing methods {m['timing']['method_counts']}, "
          f"sky circle {m['sky']}")
    if m["timing"].get("warning"):
        print("WARNING:", m["timing"]["warning"])


if __name__ == "__main__":
    main()
