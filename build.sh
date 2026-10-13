#!/usr/bin/env bash
# build.sh ─ assemble the cortex, link the console, compile the C++26 tools.
#
#   build/libxn.a   the engine: every .asm except xn.asm, exporting the C ABI
#   build/xn        the console: xn.asm + libxn.a, static, no libc
#   build/xnc       C++26 instrument (inspect · forge · sweep · table)
#   build/xn_test   C++26 test suite against the SDK
#
# env: CXX (default g++), CXXFLAGS (appended), NASM (default nasm)
set -euo pipefail
cd "$(dirname "$0")"

NASM=${NASM:-nasm}
CXX=${CXX:-g++}
OUT=build
OBJ=$OUT/obj
mkdir -p "$OBJ"

say() { printf '  \342\212\225 %s\n' "$*"; }

say "assembling engine"
ENGINE=(xor io layer inference train options model agents mnist)
for m in "${ENGINE[@]}" xn; do
    "$NASM" -f elf64 -I engine/ -o "$OBJ/$m.o" "engine/$m.asm"
done

say "archiving libxn.a"
rm -f "$OUT/libxn.a"
ar rcs "$OUT/libxn.a" $(printf "$OBJ/%s.o " "${ENGINE[@]}")

say "linking xn (static, no libc)"
ld -static -nostdlib -z noexecstack -z max-page-size=4096 --gc-sections -s -o "$OUT/xn" "$OBJ/xn.o" "$OUT/libxn.a"

CXXSTD=(-std=c++26 -O2 -Wall -Wextra -Wpedantic -Isdk/include)
say "compiling xnc (c++26)"
"$CXX" "${CXXSTD[@]}" ${CXXFLAGS:-} tools/xnc.cpp "$OUT/libxn.a" -o "$OUT/xnc"
say "compiling xnist (c++26)"
"$CXX" "${CXXSTD[@]}" ${CXXFLAGS:-} tools/xnist.cpp "$OUT/libxn.a" -o "$OUT/xnist"
say "compiling redact (c++26)"
"$CXX" "${CXXSTD[@]}" -Iapps/redact ${CXXFLAGS:-} apps/redact/redact.cpp -o "$OUT/redact"
say "compiling unredact (c++26)"
"$CXX" "${CXXSTD[@]}" -Iapps/redact ${CXXFLAGS:-} apps/redact/unredact.cpp -o "$OUT/unredact"
say "compiling xn_test (c++26)"
"$CXX" "${CXXSTD[@]}" ${CXXFLAGS:-} tests/xn_test.cpp "$OUT/libxn.a" -o "$OUT/xn_test"



size=$(stat -c %s "$OUT/xn")
say "done · xn is $size bytes"
