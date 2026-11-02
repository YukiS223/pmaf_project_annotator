#!/usr/bin/env python3
"""Export annotations to training-ready artifacts.  Fail-closed.

Given a prepared project directory (manifest.json + annotations.json),
produces export/<UTC-stamp>/ containing:

  audit.json          every frame, every state, provenance — the full record
  train_cocovis.json  COCO-VIS (YouTube-VIS style) with ONLY fully reviewed
                      frames; original frame indices and UT times preserved
  masks/%06d.png      uint16 instance-ID maps (0 = background), only for
                      fully reviewed frames
  lut.json            pixel value -> track uuid/name/color
  report.json         machine-readable validator report
  checksums.txt       sha256 of every emitted file

A frame is "fully reviewed" when it is not marked bad and EVERY track in the
project has state accepted or absent on it.  Unreviewed/proposal/edited
frames are excluded entirely — they are never exported as negatives.

The export is built in a temporary directory, validated, and only moved into
place if the validator passes.  CLI:
  python3 tools/export_coco.py <project_dir>
"""

import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pngio  # noqa: E402
import rle    # noqa: E402

EXPORT_SCHEMA_VERSION = 2
ACCEPT_STATES = ("accepted", "absent")


class ExportError(Exception):
    pass


def _load(project_dir):
    with open(os.path.join(project_dir, "manifest.json")) as f:
        manifest = json.load(f)
    anno_path = os.path.join(project_dir, "annotations.json")
    if not os.path.exists(anno_path):
        raise ExportError("no annotations.json — nothing to export")
    with open(anno_path) as f:
        doc = json.load(f)
    return manifest, doc


def frame_review_status(doc, nframes):
    """Returns (fully_reviewed set, per-frame dict of track states)."""
    tracks = doc.get("tracks", [])
    track_ids = [str(t["id"]) for t in tracks]
    frames = doc.get("frames", {})
    reviewed = set()
    states = {}
    for i in range(nframes):
        rec = frames.get(str(i), {})
        if not isinstance(rec, dict):
            continue
        st = {}
        for tid in track_ids:
            trec = (rec.get("tracks") or {}).get(tid)
            st[tid] = trec.get("state") if trec else "unreviewed"
        states[i] = {"bad": bool(rec.get("bad")), "tracks": st}
        if track_ids and not rec.get("bad") \
                and all(s in ACCEPT_STATES for s in st.values()):
            reviewed.add(i)
    return reviewed, states


def run_export(project_dir, out_root=None, progress=None):
    manifest, doc = _load(project_dir)
    w = manifest["video"]["width"]
    h = manifest["video"]["height"]
    nframes = manifest["video"]["nframes"]
    timing = manifest["timing"]["frames"]
    tracks = doc.get("tracks", [])
    frames = doc.get("frames", {})
    if not tracks:
        raise ExportError("no tracks defined — nothing to export")

    reviewed, states = frame_review_status(doc, nframes)
    included = sorted(reviewed)
    if not included:
        raise ExportError(
            "no fully reviewed frames: a frame exports only when every track "
            "is accepted or marked absent on it (and it is not a bad frame)")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_root = out_root or os.path.join(project_dir, "export")
    os.makedirs(out_root, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(prefix=f".{stamp}-", dir=out_root)
    final_dir = os.path.join(out_root, stamp)
    n_uniq = 1
    while (os.path.exists(final_dir)
           or os.path.exists(final_dir + ".FAILED")):
        n_uniq += 1
        final_dir = os.path.join(out_root, f"{stamp}-{n_uniq}")
    masks_dir = os.path.join(tmp_dir, "masks")
    os.makedirs(masks_dir)

    report = {"schema_version": EXPORT_SCHEMA_VERSION,
              "project": manifest["project"],
              "annotations_rev": doc.get("rev"),
              "created": datetime.now(timezone.utc).isoformat(),
              "frames_total": nframes,
              "frames_fully_reviewed": len(included),
              "checks": [], "errors": [], "warnings": []}

    def check(name, ok, detail=""):
        report["checks"].append({"check": name, "ok": bool(ok),
                                 "detail": detail})
        if not ok:
            report["errors"].append(f"{name}: {detail}")

    # ---- integrity: no orphaned or malformed track keys anywhere -----------
    # A frame record keyed by a track id that is not in doc.tracks would be
    # invisible to the completeness gate: its accepted mask would silently
    # vanish from the training set.  Fail closed instead.
    known_tids = {str(t["id"]) for t in tracks}
    orphans = []
    for k, rec in frames.items():
        if not isinstance(rec, dict):
            continue
        for tid in (rec.get("tracks") or {}):
            if tid not in known_tids:
                orphans.append((k, tid))
    check("no_orphan_track_records", not orphans, str(orphans[:5]))
    if orphans:
        raise ExportError(
            f"frame records exist for unknown track ids {orphans[:5]} — "
            "these annotations would be silently dropped; fix or delete them")

    id_by_tid = {str(t["id"]): int(t["id"]) for t in tracks}
    if any(v <= 0 or v > 65535 for v in id_by_tid.values()):
        raise ExportError("track ids must be in 1..65535 for uint16 masks")
    if len(set(id_by_tid.values())) != len(id_by_tid):
        raise ExportError("duplicate track ids after normalization")

    # ---- single streaming pass over included frames ------------------------
    # Decode -> disjointness -> uint16 PNG -> per-track COCO arrays, one frame
    # at a time.  Never hold all masks at once: 3 dense tracks x 3600 frames
    # of raw masks would be ~4 GB.
    ordered_tids = [str(t["id"]) for t in tracks]
    per_track = {tid: {"segs": [], "bboxes": [], "areas": [], "present": []}
                 for tid in ordered_tids}
    for k, fi in enumerate(included):
        rec = frames.get(str(fi), {})
        per = {}
        for tid in ordered_tids:
            trec = (rec.get("tracks") or {}).get(tid)
            if not trec or trec.get("state") != "accepted":
                continue
            r = trec.get("rle")
            if not r:
                raise ExportError(
                    f"frame {fi} track {tid}: accepted without a mask")
            m, mw, mh = rle.decode(r)
            if (mw, mh) != (w, h):
                raise ExportError(
                    f"frame {fi} track {tid}: mask {mw}x{mh} != video {w}x{h}")
            per[tid] = m

        grid = [0] * (w * h)
        for tid, m in per.items():
            v = id_by_tid[tid]
            for p in range(w * h):
                if m[p]:
                    if grid[p]:
                        check("masks_disjoint", False,
                              f"frame {fi}: tracks {grid[p]} and {v} overlap")
                        raise ExportError(report["errors"][-1])
                    grid[p] = v
        pngio.write_png(os.path.join(masks_dir, f"{fi:06d}.png"),
                        w, h, "L16", pngio.u16_grid_to_bytes(grid, w, h))

        for tid in ordered_tids:
            pt = per_track[tid]
            m = per.get(tid)
            if m is None:
                pt["segs"].append(None)
                pt["bboxes"].append(None)
                pt["areas"].append(None)
                pt["present"].append(False)
                continue
            bb, area = rle.bbox_area(m, w, h)
            if area == 0:
                raise ExportError(
                    f"frame {fi} track {tid}: accepted mask has zero area — "
                    "clear it or mark the track absent instead")
            pt["segs"].append({"size": [h, w],
                               "counts": rle.to_coco_counts(m, w, h)})
            pt["bboxes"].append(bb)
            pt["areas"].append(area)
            pt["present"].append(True)
        del per
        if progress:
            progress("masks", k + 1, len(included))

    # ---- COCO-VIS ----------------------------------------------------------
    file_names = [f"frames/{fi:06d}.png" for fi in included]
    frame_times = []
    for fi in included:
        t = timing[fi]
        frame_times.append({"index": fi, "seconds": t["seconds"],
                            "ut": t.get("ut"), "iso": t.get("iso"),
                            "method": t["method"], "conf": t["conf"]})
    video_entry = {
        "id": 1, "name": manifest["project"],
        "width": w, "height": h, "length": len(included),
        "file_names": file_names,
        "frame_indices": included,          # original 0-based frame indices
        "frame_times": frame_times,
        "source_sha256": manifest["source"]["sha256"],
        "source_filename": manifest["source"]["filename"],
        # file_names are relative to the project directory, NOT this export
        "image_root": os.path.abspath(project_dir),
        "night_id": manifest.get("night_id", manifest["project"]),
    }
    annotations = []
    aid = 1
    for t in tracks:
        tid = str(t["id"])
        pt = per_track[tid]
        if not any(pt["present"]):
            raise ExportError(
                f"track {tid} ({t.get('name')}) has no accepted mask on any "
                "fully reviewed frame — delete the track (right panel) or "
                "accept at least one mask; an all-empty track would poison "
                "tracker training")
        annotations.append({
            "id": aid, "video_id": 1, "category_id": 1,
            "track_id": int(t["id"]), "track_uuid": t.get("uuid"),
            "track_name": t.get("name"),
            "segmentations": pt["segs"], "bboxes": pt["bboxes"],
            "areas": pt["areas"], "present": pt["present"],
            "lifecycle_events": t.get("events", []),
        })
        aid += 1
    cocovis = {
        "info": {
            "description": "PMAF video instance segmentation (KHO all-sky)",
            "schema": "youtube-vis-style; segmentations are uncompressed "
                      "COCO RLE (column-major counts, background first); "
                      "null segmentation = track reviewed-absent on that "
                      "frame (a true negative, not a gap)",
            "export_schema_version": EXPORT_SCHEMA_VERSION,
            "tool_version": manifest.get("tool_version"),
            "annotations_rev": doc.get("rev"),
            "date_created": report["created"],
            "grouping_key_for_dataset_splits":
                manifest.get("night_id", manifest["project"]),
            "image_root_note": "file_names resolve against videos[].image_root"
                               " (the project directory), not this export dir",
        },
        "videos": [video_entry],
        "categories": [{"id": 1, "name": "PMAF", "supercategory": "aurora"}],
        "annotations": annotations,
    }
    with open(os.path.join(tmp_dir, "train_cocovis.json"), "w") as f:
        json.dump(cocovis, f)

    # ---- audit.json (full record) ------------------------------------------
    audit = {
        "export_schema_version": EXPORT_SCHEMA_VERSION,
        "manifest": manifest,
        "annotations": doc,
        "review_summary": {
            "fully_reviewed": included,
            "per_frame_states": {str(k): v for k, v in states.items()},
        },
    }
    with open(os.path.join(tmp_dir, "audit.json"), "w") as f:
        json.dump(audit, f)

    # ---- LUT ---------------------------------------------------------------
    lut = {"0": None}
    for t in tracks:
        lut[str(t["id"])] = {"uuid": t.get("uuid"), "name": t.get("name"),
                             "color": t.get("color")}
    with open(os.path.join(tmp_dir, "lut.json"), "w") as f:
        json.dump(lut, f, indent=1)

    # ---- validate ----------------------------------------------------------
    _validate(tmp_dir, project_dir, manifest, doc, included, id_by_tid,
              check, report, progress)

    with open(os.path.join(tmp_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=1)

    # checksums (report.json included; checksums.txt lists all other files)
    sums = []
    for root, _, files in os.walk(tmp_dir):
        for fn in sorted(files):
            if fn == "checksums.txt":
                continue
            p = os.path.join(root, fn)
            rel = os.path.relpath(p, tmp_dir)
            sums.append(f"{_sha256(p)}  {rel}")
    with open(os.path.join(tmp_dir, "checksums.txt"), "w") as f:
        f.write("\n".join(sums) + "\n")

    if report["errors"]:
        failed_dir = final_dir + ".FAILED"
        shutil.move(tmp_dir, failed_dir)
        raise ExportError(
            "export validation FAILED — nothing usable was produced. "
            f"Details: {failed_dir}/report.json :: "
            + "; ".join(report["errors"][:5]))
    shutil.move(tmp_dir, final_dir)
    return final_dir, report


def _sampled_masks(doc, fi, w, h):
    """Re-decode the accepted masks of one frame from the annotations doc."""
    rec = doc.get("frames", {}).get(str(fi), {})
    out = {}
    for tid, trec in (rec.get("tracks") or {}).items():
        if trec.get("state") == "accepted" and trec.get("rle"):
            m, _, _ = rle.decode(trec["rle"])
            out[tid] = m
    return out


def _validate(tmp_dir, project_dir, manifest, doc, included, id_by_tid,
              check, report, progress=None):
    w = manifest["video"]["width"]
    h = manifest["video"]["height"]
    timing = manifest["timing"]["frames"]

    # RLE round-trip: canonical string -> mask -> string, plus COCO counts
    # round-trip, on a deterministic sample (first/last/every 37th included).
    sample = sorted(set(included[::37]) | {included[0], included[-1]})
    ok_rt = True
    detail = ""
    for fi in sample:
        for tid, m in _sampled_masks(doc, fi, w, h).items():
            s = rle.encode(m, w, h)
            m2, _, _ = rle.decode(s)
            if bytes(m2) != bytes(m):
                ok_rt, detail = False, f"canonical RLE frame {fi} track {tid}"
                break
            counts = rle.to_coco_counts(m, w, h)
            if sum(counts) != w * h:
                ok_rt, detail = False, f"COCO counts sum frame {fi}"
                break
            m3 = rle.from_coco_counts(counts, w, h)
            if bytes(m3) != bytes(m):
                ok_rt, detail = False, f"COCO round-trip frame {fi} track {tid}"
                break
        if not ok_rt:
            break
    check("rle_round_trip", ok_rt, detail or f"{len(sample)} frames sampled")

    # uint16 PNG re-read equals independently re-decoded masks.
    ok_png = True
    detail = ""
    for k, fi in enumerate(sample):
        pw, ph, mode, data = pngio.read_png(
            os.path.join(tmp_dir, "masks", f"{fi:06d}.png"))
        if (pw, ph, mode) != (w, h, "L16"):
            ok_png, detail = False, f"frame {fi}: {pw}x{ph} {mode}"
            break
        grid = pngio.bytes_to_u16_grid(data, w, h)
        vals = set(grid) - {0}
        if not vals.issubset(set(id_by_tid.values())):
            ok_png, detail = False, f"frame {fi}: values {vals} not in LUT"
            break
        for tid, m in _sampled_masks(doc, fi, w, h).items():
            v = id_by_tid[tid]
            for p in range(0, w * h, 997):   # strided pixel check
                if (grid[p] == v) != bool(m[p]):
                    ok_png, detail = False, f"frame {fi} px {p}"
                    break
            if not ok_png:
                break
        if not ok_png:
            break
        if progress:
            progress("validate", k + 1, len(sample))
    check("uint16_png_round_trip", ok_png,
          detail or f"{len(sample)} frames sampled")

    # Every referenced source frame PNG must exist with plausible size.
    missing = [fi for fi in included
               if not os.path.isfile(
                   os.path.join(project_dir, "frames", f"{fi:06d}.png"))]
    check("source_frames_exist", not missing,
          f"missing frame PNGs: {missing[:5]}" if missing else
          f"{len(included)} checked")

    # Timing monotonic over included frames (rollover already unwrapped).
    secs = [timing[fi]["seconds"] for fi in included]
    mono = all(b >= a for a, b in zip(secs, secs[1:]))
    check("timestamps_monotonic", mono, "")

    # Cadence sanity: consecutive OCR'd frames should step ~1.0145 s.  A
    # stuck clock or confident misread shows up as off-model residuals even
    # when monotonicity holds.
    offs = dups = pairs = 0
    all_frames = manifest["timing"]["frames"]
    for i in range(1, len(all_frames)):
        a, b = all_frames[i - 1], all_frames[i]
        if a["method"] != "ocr" or b["method"] != "ocr":
            continue
        pairs += 1
        d = b["seconds"] - a["seconds"]
        if d == 0:
            dups += 1
        elif not (0.5 <= d <= 2.5):
            offs += 1
    if pairs:
        frac = (offs + dups) / pairs
        if frac > 0.01:
            report["warnings"].append(
                f"timing cadence anomaly: {dups} duplicate and {offs} "
                f"off-model steps in {pairs} OCR pairs — inspect the clip's "
                "burned-in clock before trusting per-frame times")

    # Alignment: every parallel array in the video entry has equal length.
    with open(os.path.join(tmp_dir, "train_cocovis.json")) as f:
        cv = json.load(f)
    v = cv["videos"][0]
    n = v["length"]
    aligned = (len(v["file_names"]) == len(v["frame_indices"])
               == len(v["frame_times"]) == n
               and all(len(a["segmentations"]) == len(a["present"])
                       == len(a["bboxes"]) == len(a["areas"]) == n
                       for a in cv["annotations"]))
    check("cocovis_arrays_aligned", aligned, "")
    inc = v["frame_indices"]
    check("frame_indices_strictly_increasing",
          all(b > a for a, b in zip(inc, inc[1:])), "")

    # Track ids unique.
    ids = [t["id"] for t in doc.get("tracks", [])]
    check("track_ids_unique", len(ids) == len(set(ids)), "")

    # Fail-closed: no frame outside `included` appears anywhere in training
    # arrays, and no non-terminal state made it in.
    frames_map = doc.get("frames", {})
    leak = []
    for fi in inc:
        rec = frames_map.get(str(fi), {})
        for tid, trec in (rec.get("tracks") or {}).items():
            if trec.get("state") not in ("accepted", "absent"):
                leak.append((fi, tid, trec.get("state")))
    check("no_nonterminal_states_exported", not leak, str(leak[:5]))

    # State counts for the report.
    sc = {}
    for rec in frames_map.values():
        for trec in (rec.get("tracks") or {}).values():
            s = trec.get("state", "?")
            sc[s] = sc.get(s, 0) + 1
    report["state_counts"] = sc


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("project_dir")
    args = ap.parse_args()
    pd = args.project_dir.rstrip("/")
    if not os.path.isfile(os.path.join(pd, "manifest.json")):
        sys.exit(f"{pd}: not a prepared project (no manifest.json)")

    def prog(phase, done, total):
        print(f"\r[{phase}] {done}/{total}", end="", flush=True)

    try:
        out_dir, report = run_export(pd, progress=prog)
    except ExportError as e:
        print(f"\nEXPORT FAILED: {e}")
        sys.exit(2)
    print(f"\nexport OK -> {out_dir}")
    print(f"  fully reviewed frames: {report['frames_fully_reviewed']}"
          f" / {report['frames_total']}")
    for c in report["checks"]:
        print(f"  [{'ok' if c['ok'] else 'FAIL'}] {c['check']} {c['detail']}")
    for wmsg in report["warnings"]:
        print("  warning:", wmsg)


if __name__ == "__main__":
    main()
