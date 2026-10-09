#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
SDK="$ROOT/local/libecw"
OUT="$ROOT/local/bin"

mkdir -p "$OUT"

clang++ -std=c++11 -O2 \
  -I"$SDK/include" \
  "$ROOT/native/ecw_window.cpp" \
  "$SDK/lib/libecwj2.a" \
  -lz -lm -lpthread \
  -o "$OUT/ecw_window"

file "$OUT/ecw_window"
