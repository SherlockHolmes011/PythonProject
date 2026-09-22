bas#!/bin/bash
# find_ram_hogs2.sh
#
# Part 1: re-check the sketch-local .o files, sorted correctly this time
# (previous version's sort broke on the padded columns).
#
# Part 2: the real suspect -- WiFi/lwIP/mbedTLS ship as precompiled
# .a static libraries bundled with the ESP32 core package, not as
# source recompiled per-sketch, so Part 1 could never have seen them.
# This searches the whole core package folder for .a files and sizes
# each one.

TOOLBIN="/c/Users/reddy/AppData/Local/Arduino15/packages/esp32/tools/esp-x32/2601/bin"
SKETCH_BUILD="/c/Users/reddy/AppData/Local/arduino/sketches/673074CBE24BDB5445C7E9E90300B3EA"
CORE_PKG="/c/Users/reddy/AppData/Local/Arduino15/packages/esp32/hardware/esp32/3.3.11"
SIZE_TOOL="$TOOLBIN/xtensa-esp-elf-size.exe"

echo "=== Part 1: sketch-local object files (fixed sort) ==="
echo "text      data      bss       dec       file"
find "$SKETCH_BUILD" -name "*.o" 2>/dev/null | while read -r f; do
  "$SIZE_TOOL" "$f" 2>/dev/null | tail -1 | awk -v file="$f" '{printf "%-10s%-10s%-10s%-10s%s\n", $1, $2, $3, $4, file}'
done | sort -k3 -nr | head -15

echo ""
echo "=== Part 2: precompiled core libraries (.a files) -- the likely real source ==="
echo "text      data      bss       dec       file"
find "$CORE_PKG" -name "*.a" 2>/dev/null | while read -r f; do
  "$SIZE_TOOL" "$f" 2>/dev/null | tail -1 | awk -v file="$f" '{printf "%-10s%-10s%-10s%-10s%s\n", $1, $2, $3, $4, file}'
done | sort -k3 -nr | head -15

echo ""
echo "If Part 2 is empty, the core libs live somewhere other than expected --"
echo "run this to locate them instead:"
echo "  find \"/c/Users/reddy/AppData/Local/Arduino15/packages/esp32\" -name \"*.a\" | head -20"
