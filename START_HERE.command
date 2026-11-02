#!/bin/zsh
# PMAF Annotator launcher (macOS).  First launch: right-click -> Open.
cd "$(dirname "$0")"

echo "PMAF Annotator — starting up"

# Clear the quarantine flag Safari/Chrome put on downloaded files so the
# bundled ffmpeg/python can run.  (We cannot clear our own flag — that is why
# the very first launch needs right-click -> Open.)
xattr -dr com.apple.quarantine bin runtime tools app serve.py 2>/dev/null

# --- pick a Python ----------------------------------------------------------
PY=""
if [ -x "runtime/macos-arm64/bin/python3" ]; then
  if runtime/macos-arm64/bin/python3 -c "print()" >/dev/null 2>&1; then
    PY="runtime/macos-arm64/bin/python3"
  fi
fi
if [ -z "$PY" ]; then
  if command -v python3 >/dev/null 2>&1 && python3 -c "print()" >/dev/null 2>&1; then
    PY="python3"
    echo "note: using system python3 (bundled runtime unavailable on this Mac)"
  fi
fi
if [ -z "$PY" ]; then
  echo ""
  echo "ERROR: no working Python found."
  echo "  The bundled runtime targets Apple-silicon Macs.  On this machine,"
  echo "  install Python 3 first:  https://www.python.org/downloads/"
  echo "  (or: xcode-select --install)"
  read -k 1 "?press any key to close"
  exit 1
fi

# --- sanity: bundled ffmpeg runs -------------------------------------------
if [ -x "bin/ffmpeg-macos-arm64" ]; then
  if ! bin/ffmpeg-macos-arm64 -version >/dev/null 2>&1; then
    echo "warning: bundled ffmpeg failed to run (Gatekeeper?)."
    echo "  Fix: System Settings -> Privacy & Security -> Allow,"
    echo "  or install ffmpeg via Homebrew (brew install ffmpeg)."
  fi
fi

df -g . | awk 'NR==2 {print "free disk: " $4 " GB (a new 1-hour clip needs ~2 GB)"}'

exec "$PY" serve.py
