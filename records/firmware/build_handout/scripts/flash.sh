#!/usr/bin/env bash
set -euo pipefail
PORT="${PORT:-/dev/ttyACM0}"; BAUD="${BAUD:-460800}"
BIN="${BIN:-$(ls -1t "$HOME"/Desktop/micropython-thonny-*.bin 2>/dev/null | head -n1)}"
if [[ -n "$BIN" && -f "$BIN" ]]; then
  python3 -m esptool --chip esp32 --port "$PORT" --baud "$BAUD"     write_flash -z --flash_size keep 0x0 "$BIN"
else
  ROOT="$(cd "$(dirname "$0")/.."; pwd)"
  PD="$ROOT/firmware/micropython/ports/esp32"
  python3 -m esptool --chip esp32 --port "$PORT" --baud "$BAUD" write_flash -z     0x1000  "$PD/build/bootloader/bootloader.bin"     0x8000  "$PD/build/partition_table/partition-table.bin"     0x10000 "$PD/build/micropython.bin"
fi
