#!/usr/bin/env bash
set -euo pipefail

# ---- Settings ----
PROJ_ROOT="${PROJ_ROOT:-$HOME/m5-firmware/firmware}"
PORTS="$PROJ_ROOT/micropython/ports/esp32"
USERMODS="$PROJ_ROOT/usermods"
STUB="$PROJ_ROOT/ses_modules/emlearn-micropython/src/emlearn_iir/mod_emlearn_iir.c"

TS=$(date +%Y%m%d-%H%M%S)
OUTDIR="$HOME/Desktop/m5-core2-firmware-release-$TS"
ZIP="$HOME/Desktop/m5-core2-firmware-release-$TS.zip"

mkdir -p "$OUTDIR"/{artifacts,source}

echo ">> Copying source additions"
install -D -m 0644 "$STUB"                       "$OUTDIR/source/mod_emlearn_iir.c"
install -D -m 0644 "$USERMODS/micropython.cmake" "$OUTDIR/source/micropython.cmake"

echo ">> Ensuring build outputs exist"
if [ ! -f "$PORTS/build/micropython.bin" ]; then
  echo "micropython.bin missing, building..."
  cd "$PORTS"
  idf.py build
fi

echo ">> Collecting artifacts"
install -D -m 0644 "$PORTS/build/micropython.bin"                         "$OUTDIR/artifacts/micropython.bin"
install -D -m 0644 "$PORTS/build/bootloader/bootloader.bin"               "$OUTDIR/artifacts/bootloader.bin"
install -D -m 0644 "$PORTS/build/partition_table/partition-table.bin"     "$OUTDIR/artifacts/partition-table.bin"

MERGED="$HOME/Desktop/micropython-thonny.bin"
if [ ! -f "$MERGED" ]; then
  echo "Creating merged image at $MERGED"
  /usr/bin/python3 -m esptool --chip esp32 merge_bin -o "$MERGED" \
    --flash_mode dio --flash_freq 40m --flash_size 4MB \
    0x1000  "$OUTDIR/artifacts/bootloader.bin" \
    0x8000  "$OUTDIR/artifacts/partition-table.bin" \
    0x10000 "$OUTDIR/artifacts/micropython.bin"
fi
install -D -m 0644 "$MERGED" "$OUTDIR/artifacts/micropython-thonny.bin"

echo ">> Creating checksums"
( cd "$OUTDIR/artifacts" && sha256sum * > SHA256SUMS.txt )

echo ">> Writing BUILD_INFO.txt"
cat > "$OUTDIR/BUILD_INFO.txt" <<EOF
Target: M5Stack Core2 (spiram-16m) / ESP32
MicroPython: v1.20.0
Toolchain: ESP-IDF 4.4 (xtensa-esp32-elf-gcc 8.4.0)
Includes: ulab, emlearn_iir stub, ubluetooth, uasyncio, network, machine/umachine, esp
EOF

echo ">> Zipping to $ZIP"
( cd "$OUTDIR/.." && zip -r "$(basename "$ZIP")" "$(basename "$OUTDIR")" >/dev/null )
echo ">> Done: $ZIP"
