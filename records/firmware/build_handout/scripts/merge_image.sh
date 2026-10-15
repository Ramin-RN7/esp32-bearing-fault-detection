#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.."; pwd)"
PORT_DIR="$ROOT/firmware/micropython/ports/esp32"
python3 -m pip install --user "esptool>=4.0" >/dev/null
mkdir -p "$HOME/Desktop"; TS="$(date +%Y%m%d-%H%M%S)"
OUT="$HOME/Desktop/micropython-thonny-$TS.bin"
python3 -m esptool --chip esp32 merge_bin -o "$OUT"   --flash_mode dio --flash_freq 40m --flash_size 4MB   0x1000  "$PORT_DIR/build/bootloader/bootloader.bin"   0x8000  "$PORT_DIR/build/partition_table/partition-table.bin"   0x10000 "$PORT_DIR/build/micropython.bin"
ls -lh "$OUT"; sha256sum "$OUT" || true
