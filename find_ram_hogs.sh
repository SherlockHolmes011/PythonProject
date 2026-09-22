#!/bin/bash
# find_ram_hogs.sh
#
# Since the link failed, there's no final .elf to inspect -- but every
# individual .o file the compiler already produced still exists in the
# sketch build cache. This runs the toolchain's `size` tool on each one
# and sorts by BSS (uninitialized static RAM) usage, largest first, so
# you can see exactly which source file or library is actually eating
# your DRAM budget instead of guessing.
#
# Usage (from Git Bash, same terminal you're already using):
#   bash find_ram_hogs.sh

TOOLBIN="/c/Users/reddy/AppData/Local/Arduino15/packages/esp32/tools/esp-x32/2601/bin"
SKETCH_BUILD="/c/Users/reddy/AppData/Local/arduino/sketches/673074CBE24BDB5445C7E9E90300B3EA"
SIZE_TOOL="$TOOLBIN/xtensa-esp-elf-size.exe"

if [ ! -f "$SIZE_TOOL" ]; then
  echo "size tool not found at: $SIZE_TOOL"
  echo "Check the exact folder name under .../esp32/tools/esp-x32/ -- it may"
  echo "differ from 2601 if you're on a different core version. List it with:"
  echo "  ls \"/c/Users/reddy/AppData/Local/Arduino15/packages/esp32/tools/esp-x32/\""
  exit 1
fi

echo "text    data    bss     dec     file"
echo "----    ----    ---     ---     ----"

find "$SKETCH_BUILD" -name "*.o" 2>/dev/null | while read -r f; do
  "$SIZE_TOOL" "$f" 2>/dev/null | tail -1 | awk -v file="$f" '{printf "%-8s%-8s%-8s%-8s%s\n", $1, $2, $3, $4, file}'
done | sort -t' ' -k3 -n -r | head -25

echo ""
echo "Sorted by BSS (3rd column) descending -- the top few entries are"
echo "your biggest static-RAM contributors. Files under .../libraries/ or"
echo ".../cores/esp32/ point to WiFi/lwIP/library code; a file matching"
echo "kws_firmware.ino.cpp.o points to your own sketch code."
