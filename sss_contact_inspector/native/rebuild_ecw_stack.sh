#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
SOURCE="$ROOT/vendor-src/libecw"
PREFIX="$ROOT/local/libecw"

cd "$SOURCE"
./configure \
  --target=arm-apple-darwin \
  --prefix="$PREFIX" \
  --disable-shared \
  --enable-static
make -j2

# The 2006 install script is not compatible with current macOS install(1),
# so copy the already-built static library and public headers explicitly.
mkdir -p "$PREFIX/lib" "$PREFIX/include"
cp lib/libecwj2.a "$PREFIX/lib/libecwj2.a"
cp Source/include/*.h "$PREFIX/include/"
cp Source/NCSBuildGnu/NCSGnuConfig.h "$PREFIX/include/"
cp Source/NCSBuildGnu/NCSTypesWithGnuDefs.h "$PREFIX/include/NCSTypes.h"
cp License.txt "$PREFIX/License.txt"

"$ROOT/native/build_ecw_window.sh"
