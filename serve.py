#!/usr/bin/env python3
"""PMAF Annotator local server (stdlib only).

Serves the single-file UI, prepares projects (ffmpeg frame extraction, UT
OCR, background bank), stores annotations with compare-and-swap revisioned
delta saves, and runs the fail-closed exporter.

Binds to 127.0.0.1 only.  Start via START_HERE.command / START_HERE.bat, or:
  python3 serve.py [--port 8798] [--no-browser]
"""

import hashlib
import json
import os
import platform
import re
import shutil
import sys
import threading
import time
import uuid
import webbrowser

try:
    import fcntl
except ImportError:          # Windows
    fcntl = None
try:
    import msvcrt
except ImportError:          # POSIX
    msvcrt = None
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PKG_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PKG_ROOT, "tools"))
import export_coco  # noqa: E402
import prepare      # noqa: E402

SERVER_VERSION = prepare.TOOL_VERSION
PROJECTS_DIR = os.path.join(PKG_ROOT, "projects")
APP_INDEX = os.path.join(PKG_ROOT, "app", "index.html")
NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
FRAME_RE = re.compile(r"^\d{6}\.png$")
BG_RE = re.compile(r"^bg_\d\.png$")
MAX_PUT_BYTES = 200 * 1024 * 1024
BACKUP_KEEP = 12
BACKUP_MAX_BYTES = 500 * 1024 * 1024
PREV_REFRESH_S = 60
DISK_ESTIMATE_FACTOR = 130   # bytes of PNG frames per byte of source video

_lock = threading.Lock()                  # guards the dicts below
_project_locks = {}                       # name -> Lock
_annos = {}                               # name -> {"doc":..., "rev":...}
_jobs = {}                                # (kind, name) -> status dict
_scan_cache = {"videos": [], "at": 0}


def _plock(name):
    with _lock:
        if name not in _project_locks:
            _project_locks[name] = threading.Lock()
        return _project_locks[name]


def _pdir(name):
    return os.path.join(PROJECTS_DIR, name)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _new_doc(name):
    return {"schema_version": 2, "project": name, "rev": uuid.uuid4().hex,
            "updated_at": _now(), "algo_version": 1, "tracks": [],
            "frames": {}}


def _anno_path(name):
    return os.path.join(_pdir(name), "annotations.json")


def _load_annotations(name):
    with _lock:
        if name in _annos:
            return _annos[name]
    path = _anno_path(name)
    if os.path.exists(path):
        with open(path) as f:
            doc = json.load(f)
        if "rev" not in doc:
            doc["rev"] = uuid.uuid4().hex
    else:
        doc = _new_doc(name)
    with _lock:
        _annos[name] = doc
    return doc


def _save_annotations(name, doc):
    """Atomic write with .prev safety copy and time-bucketed backups."""
    pd = _pdir(name)
    path = _anno_path(name)
    prev = path + ".prev"
    backups = os.path.join(pd, "backups")
    os.makedirs(backups, exist_ok=True)
    if os.path.exists(path):
        try:
            if (not os.path.exists(prev)
                    or time.time() - os.path.getmtime(prev) > PREV_REFRESH_S):
                shutil.copy2(path, prev + ".tmp")
                os.replace(prev + ".tmp", prev)
            bucket = datetime.now(timezone.utc).strftime("%Y%m%d%H%M")[:11]
            bpath = os.path.join(backups, f"anno-{bucket}0.json")
            if not os.path.exists(bpath):
                shutil.copy2(path, bpath + ".tmp")
                os.replace(bpath + ".tmp", bpath)
                # cap by count AND total bytes (oldest evicted first)
                keep = sorted(os.listdir(backups))
                for old in keep[:-BACKUP_KEEP]:
                    os.remove(os.path.join(backups, old))
                keep = sorted(os.listdir(backups))
                total = sum(os.path.getsize(os.path.join(backups, b))
                            for b in keep)
                while keep[:-1] and total > BACKUP_MAX_BYTES:
                    victim = os.path.join(backups, keep.pop(0))
                    total -= os.path.getsize(victim)
                    os.remove(victim)
        except OSError as e:
            print(f"[warn] backup rotation failed: {e}")
    tmp = path + f".tmp-{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(doc, f)
        f.flush()
        os.fsync(f.fileno())     # survive power loss, not just crashes
    os.replace(tmp, path)


def _manifest(name):
    p = os.path.join(_pdir(name), "manifest.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def _list_projects():
    out = []
    if not os.path.isdir(PROJECTS_DIR):
        return out
    for name in sorted(os.listdir(PROJECTS_DIR)):
        if not NAME_RE.match(name) or not os.path.isdir(_pdir(name)):
            continue
        m = _manifest(name)
        job = _jobs.get(("prepare", name))
        entry = {"name": name, "prepared": m is not None}
        if m:
            entry["nframes"] = m["video"]["nframes"]
            entry["source"] = m["source"]["filename"]
            entry["timing_methods"] = m["timing"]["method_counts"]
            with _plock(name):   # a concurrent PUT may be mutating the doc
                doc = _load_annotations(name)
                entry["tracks"] = len(doc.get("tracks", []))
                states = {}
                for rec in doc.get("frames", {}).values():
                    for tr in (rec.get("tracks") or {}).values():
                        s = tr.get("state", "?")
                        states[s] = states.get(s, 0) + 1
            entry["state_counts"] = states
        if job and not job.get("finished"):
            entry["preparing"] = {k: job[k] for k in
                                  ("phase", "done", "total") if k in job}
        if job and job.get("error"):
            entry["prepare_error"] = job["error"]
        out.append(entry)
    return out


def _scan_videos():
    """Video files next to the package (parent dir) and inside it."""
    vids = []
    seen = set()
    for d in (os.path.dirname(PKG_ROOT), PKG_ROOT):
        try:
            for fn in sorted(os.listdir(d)):
                p = os.path.join(d, fn)
                if (os.path.isfile(p) and not fn.startswith(".")
                        and fn.lower().endswith(prepare.VIDEO_EXTS)
                        and p not in seen):
                    seen.add(p)
                    # id derived from the path: stable across rescans, so a
                    # stale client can never import the wrong file
                    vid = hashlib.sha1(p.encode()).hexdigest()[:12]
                    vids.append({"id": vid, "name": fn,
                                 "bytes": os.path.getsize(p), "path": p})
        except OSError:
            pass
    _scan_cache["videos"] = vids
    _scan_cache["at"] = time.time()
    return [{k: v[k] for k in ("id", "name", "bytes")} for v in vids]


def _disk_free(path):
    try:
        return shutil.disk_usage(path).free
    except OSError:
        return None


def _start_prepare(name, src_path, src_name, force=False):
    key = ("prepare", name)
    with _lock:
        if key in _jobs and not _jobs[key].get("finished"):
            return _jobs[key], "already preparing"
        # a fresh project must never inherit a cached doc from a
        # previously deleted project of the same name
        _annos.pop(name, None)
        est = os.path.getsize(src_path) * DISK_ESTIMATE_FACTOR
        free = _disk_free(PROJECTS_DIR)
        if free is not None and free < est * 1.5 and not force:
            return None, (f"low disk: need ~{est // 2**30 + 1} GB for frames, "
                          f"only {free // 2**30} GB free (retry with force)")
        job = {"phase": "queued", "done": 0, "total": 0, "started": _now(),
               "finished": False, "error": None}
        _jobs[key] = job

    def work():
        try:
            def prog(phase, done, total):
                job["phase"], job["done"], job["total"] = phase, done, total
            prepare.prepare_project(src_path, _pdir(name), progress=prog,
                                    keep_source_name=src_name)
            job["phase"] = "done"
        except Exception as e:   # surfaced to the UI
            job["error"] = f"{type(e).__name__}: {e}"
            job["phase"] = "error"
        finally:
            job["finished"] = True

    threading.Thread(target=work, daemon=True).start()
    return job, None


def _start_export(name):
    key = ("export", name)
    with _lock:
        if key in _jobs and not _jobs[key].get("finished"):
            return _jobs[key], "already exporting"
        job = {"phase": "starting", "done": 0, "total": 0, "started": _now(),
               "finished": False, "error": None, "out_dir": None,
               "report": None}
        _jobs[key] = job

    def work():
        try:
            def prog(phase, done, total):
                job["phase"], job["done"], job["total"] = phase, done, total
            out_dir, report = export_coco.run_export(_pdir(name),
                                                     progress=prog)
            job["out_dir"] = out_dir
            job["report"] = report
            job["phase"] = "done"
        except export_coco.ExportError as e:
            job["error"] = str(e)
            job["phase"] = "error"
        except Exception as e:
            job["error"] = f"{type(e).__name__}: {e}"
            job["phase"] = "error"
        finally:
            job["finished"] = True

    threading.Thread(target=work, daemon=True).start()
    return job, None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = f"PMAFAnnotator/{SERVER_VERSION}"

    def log_message(self, fmt, *args):
        pass  # quiet; errors are reported to the UI

    # ---- helpers -----------------------------------------------------------
    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path, ctype, immutable=False):
        # Open before committing headers: an error after end_headers() would
        # otherwise write a second response into the same stream.
        try:
            f = open(path, "rb")
        except OSError:
            return self._json({"error": "not found"}, 404)
        try:
            size = os.fstat(f.fileno()).st_size
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(size))
            self.send_header("Cache-Control",
                             "public, max-age=31536000, immutable"
                             if immutable else "no-store")
            self.end_headers()
            shutil.copyfileobj(f, self.wfile)
        except (BrokenPipeError, ConnectionResetError, OSError):
            self.close_connection = True
        finally:
            f.close()

    def _body(self, cap=MAX_PUT_BYTES):
        ln = int(self.headers.get("Content-Length", 0))
        if ln > cap:
            raise ValueError(f"body too large ({ln} bytes)")
        return self.rfile.read(ln)

    def _q(self):
        from urllib.parse import parse_qs, urlparse
        u = urlparse(self.path)
        return u.path, {k: v[0] for k, v in parse_qs(u.query).items()}

    def _project(self, q):
        name = q.get("name", "")
        if not NAME_RE.match(name):
            raise ValueError("bad project name")
        return name

    # ---- GET ---------------------------------------------------------------
    def do_GET(self):
        try:
            path, q = self._q()
            if path in ("/", "/index.html"):
                return self._file(APP_INDEX, "text/html; charset=utf-8")
            if path == "/api/health":
                try:
                    ff = prepare.find_ffmpeg()
                    ffv = prepare.ffmpeg_version(ff)
                except RuntimeError as e:
                    ff, ffv = None, str(e)
                return self._json({
                    "ok": ff is not None, "version": SERVER_VERSION,
                    "ffmpeg": ff, "ffmpeg_version": ffv,
                    "platform": f"{platform.system()} {platform.machine()}",
                    "python": sys.version.split()[0],
                    "disk_free_gb": round((_disk_free(PROJECTS_DIR) or 0)
                                          / 2**30, 1)})
            if path == "/api/projects":
                return self._json({"projects": _list_projects()})
            if path == "/api/scan_videos":
                return self._json({"videos": _scan_videos()})
            if path == "/api/prepare_status":
                name = self._project(q)
                job = _jobs.get(("prepare", name))
                return self._json(job or {"phase": "none"})
            if path == "/api/export_status":
                name = self._project(q)
                job = _jobs.get(("export", name))
                return self._json(job or {"phase": "none"})
            if path == "/api/annotations":
                name = self._project(q)
                with _plock(name):
                    doc = _load_annotations(name)
                return self._json({"doc": doc})
            m = re.match(r"^/p/([A-Za-z0-9_-]{1,64})/(.+)$", path)
            if m:
                name, rest = m.group(1), m.group(2)
                pd = _pdir(name)
                if rest == "manifest.json":
                    return self._file(os.path.join(pd, "manifest.json"),
                                      "application/json")
                fm = re.match(r"^frames/(\d{6}\.png)$", rest)
                if fm:
                    return self._file(os.path.join(pd, "frames", fm.group(1)),
                                      "image/png", immutable=True)
                bm = re.match(r"^backgrounds/(bg_\d\.png)$", rest)
                if bm:
                    return self._file(
                        os.path.join(pd, "backgrounds", bm.group(1)),
                        "image/png", immutable=True)
            self._json({"error": "not found"}, 404)
        except ValueError as e:
            self._json({"error": str(e)}, 400)
        except (BrokenPipeError, ConnectionResetError):
            pass

    # ---- POST / PUT --------------------------------------------------------
    def do_POST(self):
        try:
            path, q = self._q()
            if path == "/api/upload":
                name = self._project(q)
                fname = os.path.basename(q.get("filename", "video"))
                ext = os.path.splitext(fname)[1].lower()
                if ext not in prepare.VIDEO_EXTS:
                    raise ValueError(f"unsupported extension {ext} "
                                     f"(want {prepare.VIDEO_EXTS})")
                if os.path.exists(os.path.join(_pdir(name), "manifest.json")):
                    raise ValueError("project already exists")
                job = _jobs.get(("prepare", name))
                if job and not job.get("finished"):
                    raise ValueError("project is being prepared — wait for "
                                     "it to finish before re-uploading")
                os.makedirs(_pdir(name), exist_ok=True)
                dst = os.path.join(_pdir(name), "source" + ext)
                ln = int(self.headers.get("Content-Length", 0))
                free = _disk_free(PROJECTS_DIR) or 0
                if ln * (1 + DISK_ESTIMATE_FACTOR * 1.5) > free > 0:
                    raise ValueError("not enough disk for video + frames")
                read = 0
                with open(dst, "wb") as f:
                    while read < ln:
                        chunk = self.rfile.read(min(1 << 20, ln - read))
                        if not chunk:
                            break
                        f.write(chunk)
                        read += len(chunk)
                if read != ln:
                    os.remove(dst)
                    raise ValueError("upload truncated")
                job, err = _start_prepare(name, dst, fname,
                                          force=q.get("force") == "1")
                if err:
                    return self._json({"error": err}, 409)
                return self._json({"ok": True, "stored": os.path.basename(dst)})
            if path == "/api/import":
                name = self._project(q)
                body = json.loads(self._body(1 << 20) or b"{}")
                vid = next((v for v in _scan_cache["videos"]
                            if v["id"] == body.get("video_id")), None)
                if vid is None:
                    raise ValueError("unknown video_id (rescan and retry)")
                if os.path.exists(os.path.join(_pdir(name), "manifest.json")):
                    raise ValueError("project already exists")
                job = _jobs.get(("prepare", name))
                if job and not job.get("finished"):
                    raise ValueError("project is being prepared")
                os.makedirs(_pdir(name), exist_ok=True)
                ext = os.path.splitext(vid["name"])[1].lower()
                dst = os.path.join(_pdir(name), "source" + ext)
                if not os.path.exists(dst):
                    try:
                        os.link(vid["path"], dst)
                    except OSError:
                        shutil.copy2(vid["path"], dst)
                job, err = _start_prepare(name, dst, vid["name"],
                                          force=body.get("force") is True)
                if err:
                    return self._json({"error": err}, 409)
                return self._json({"ok": True})
            if path == "/api/export":
                name = self._project(q)
                job, err = _start_export(name)
                if err:
                    return self._json({"error": err}, 409)
                return self._json({"ok": True})
            self._json({"error": "not found"}, 404)
        except ValueError as e:
            self._json({"error": str(e)}, 400)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_PUT(self):
        try:
            path, q = self._q()
            if path != "/api/annotations":
                return self._json({"error": "not found"}, 404)
            name = self._project(q)
            delta = json.loads(self._body())
            base_rev = delta.get("base_rev")
            manifest = _manifest(name)
            if manifest is None:
                raise ValueError("project not prepared")
            nframes = manifest["video"]["nframes"]
            with _plock(name):
                doc = _load_annotations(name)
                if base_rev != doc["rev"]:
                    return self._json({"error": "revision conflict",
                                       "rev": doc["rev"]}, 409)
                # Validate the ENTIRE delta before touching the shared doc —
                # a partially applied delta would diverge memory from disk
                # and a later save would persist the divergence silently.
                if "tracks" in delta and not isinstance(delta["tracks"], list):
                    raise ValueError("tracks must be a list")
                norm = {}
                for k, rec in (delta.get("frames") or {}).items():
                    if not k.isdigit() or not (0 <= int(k) < nframes):
                        raise ValueError(f"bad frame index {k!r}")
                    if rec is not None and not isinstance(rec, dict):
                        raise ValueError("frame record must be object or null")
                    norm[str(int(k))] = rec
                if "tracks" in delta:
                    doc["tracks"] = delta["tracks"]
                for fk, rec in norm.items():
                    if rec is None:
                        doc["frames"].pop(fk, None)
                    else:
                        doc["frames"][fk] = rec
                doc["rev"] = uuid.uuid4().hex
                doc["updated_at"] = _now()
                try:
                    _save_annotations(name, doc)
                except OSError as e:
                    # memory now diverges from disk: drop the cache so the
                    # next access reloads the last durable state; the client
                    # keeps its dirty set and retries against the disk rev
                    with _lock:
                        _annos.pop(name, None)
                    return self._json({"error": f"disk write failed: {e}"},
                                      500)
                return self._json({"rev": doc["rev"]})
        except ValueError as e:
            self._json({"error": str(e)}, 400)
        except (BrokenPipeError, ConnectionResetError):
            pass


_LOCK_FH = None   # must outlive main() — the OS lock dies with the fd


def acquire_instance_lock():
    """One server per package. Two processes over the same projects/ would
    each cache annotation revisions independently and silently overwrite
    each other's saves — in-process CAS cannot see across processes."""
    global _LOCK_FH
    path = os.path.join(PKG_ROOT, ".server.lock")
    _LOCK_FH = open(path, "a+")
    try:
        if fcntl is not None:
            fcntl.flock(_LOCK_FH, fcntl.LOCK_EX | fcntl.LOCK_NB)
        elif msvcrt is not None:
            msvcrt.locking(_LOCK_FH.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        return False
    _LOCK_FH.truncate(0)
    _LOCK_FH.write(str(os.getpid()))
    _LOCK_FH.flush()
    return True


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8798)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    if not acquire_instance_lock():
        print("PMAF Annotator is already running for this package — "
              "use the existing window (check your browser), or close the "
              "other terminal first.")
        sys.exit(1)
    os.makedirs(PROJECTS_DIR, exist_ok=True)
    httpd = None
    port = args.port
    for cand in range(args.port, args.port + 10):
        try:
            httpd = ThreadingHTTPServer(("127.0.0.1", cand), Handler)
            port = cand
            break
        except OSError:
            continue
    if httpd is None:
        sys.exit(f"no free port in {args.port}..{args.port + 9}")
    url = f"http://127.0.0.1:{port}/"
    try:
        ff = prepare.find_ffmpeg()
        print(f"ffmpeg: {ff}")
    except RuntimeError as e:
        print(f"WARNING: {e}")
    print(f"PMAF Annotator v{SERVER_VERSION}  ->  {url}")
    print("Keep this window open while annotating.  Ctrl+C to stop.")
    if not args.no_browser:
        threading.Timer(0.7, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
