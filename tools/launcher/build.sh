#!/bin/sh
# Cross-compile START_HERE.exe for Windows x64 from macOS/Linux.
# Requires mingw-w64 (macOS: brew install mingw-w64) and python3.
set -e
cd "$(dirname "$0")"

[ -f pmaf.ico ] || python3 gen_icon.py

x86_64-w64-mingw32-windres launcher.rc launcher_res.o
x86_64-w64-mingw32-gcc -O2 -municode -mconsole -static -s \
    launcher.c launcher_res.o -o ../../START_HERE.exe
rm -f launcher_res.o

echo "built ../../START_HERE.exe:"
ls -la ../../START_HERE.exe
