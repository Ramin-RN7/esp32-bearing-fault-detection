# M5Stack Core2 MicroPython Firmware (emlearn/ulab) — Release Skeleton

**Target**: M5Stack Core2 (ESP32, spiram-16m)  
**MicroPython**: v1.20.0  
**ESP-IDF toolchain**: 4.4 (xtensa-esp32-elf-gcc 8.4.0)

This ZIP contains:
- `ses_modules/emlearn-micropython/src/emlearn_iir/mod_emlearn_iir.c` — minimal stub so the builtins table can link (`mp_module_emlearn_iir`).
- `usermods/micropython.cmake` — integrates the stub via CMake and links it to `usermod`.
- `make_release_zip.sh` — a helper script you can run **locally** to collect binaries and produce the final deliverable ZIP.
- `modules_list.txt` — module inventory observed on the device.
- `Release_Notes.pdf` — brief notes.

## The 7 important modules to verify in the final firmware

1. `ulab` (with `ulab.numpy` / `ulab.scipy` available)
2. `emlearn_iir` (stub added here to satisfy builtins table)
3. `ubluetooth`
4. `uasyncio`
5. `network`
6. `machine` / `umachine`
7. `esp`

(Also present: `emlearn_fft_c`, `emlearn_trees_c`.)

## Build: Use the repo's ESP-IDF 4.4 toolchain

```bash
# Ensure IDF 4.4 environment (example):
export IDF_PATH="$HOME/m5-firmware/firmware/esp-idf"
export IDF_TOOLS_PATH="$HOME/.espressif-idf44"
source "$IDF_PATH/export.sh"

# gcc should be 8.4.0:
xtensa-esp32-elf-gcc --version | head -1

# Build
cd "$HOME/m5-firmware/firmware/micropython/ports/esp32"
idf.py -DMICROPY_BOARD=CORE2        -DMICROPY_FROZEN_MANIFEST="$HOME/combined_manifest.py"        -DUSER_C_MODULES="$HOME/m5-firmware/firmware/usermods/micropython.cmake" build
```

### Common issues & fixes

- **`esp_adc_cal` missing** during CMake configure — happens if an IDF v6 path leaks into the env.  
  **Fix**: re-source **IDF 4.4** (`$IDF_PATH/export.sh`) and ensure `xtensa-esp32-elf-gcc` is **8.4.0**.

- **Linker error** `undefined reference to mp_module_emlearn_iir` — the builtins table expects this symbol.  
  **Fix**: add `mod_emlearn_iir.c` (this stub) and the CMake usermod block (included here).

- **`MP_REGISTER_MODULE` macro form** — for MicroPython v1.20 / IDF4 use the **2-argument** form:  
  `MP_REGISTER_MODULE(MP_QSTR_emlearn_iir, mp_module_emlearn_iir)`

- **esptool not found** when using the IDF's Python venv — install esptool to **system Python** and call with `/usr/bin/python3 -m esptool`.

- **Serial port busy/missing** — use `/dev/serial/by-id/*` symlink, add user to `dialout`, stop ModemManager.

## Merge a single Thonny-friendly image

```bash
# Using system Python where esptool is installed:
/usr/bin/python3 -m esptool --chip esp32 merge_bin -o "$HOME/Desktop/micropython-thonny.bin"   --flash_mode dio --flash_freq 40m --flash_size 4MB   0x1000  build/bootloader/bootloader.bin   0x8000  build/partition_table/partition-table.bin   0x10000 build/micropython.bin
```

## Flash it

```bash
PORT=$(readlink -f /dev/serial/by-id/usb-*-if00)  # adjust if needed
BAUD=460800  # 921600 if stable

/usr/bin/python3 -m esptool --chip esp32 --port "$PORT" --baud $BAUD   write_flash -z --flash_size keep 0x0 "$HOME/Desktop/micropython-thonny.bin"
```

## Verify on REPL

```python
import sys, os
print(sys.implementation)
print(os.uname())

for m in ["ulab","emlearn_iir","ubluetooth","uasyncio","network","machine","esp"]:
    __import__(m); print("OK:", m)
```

## Create the **final** deliverable ZIP (run locally)

Use the `make_release_zip.sh` script included here, which packages:
- build outputs (`bootloader.bin`, `partition-table.bin`, `micropython.bin`)
- merged image `micropython-thonny-*.bin` (it will create one if missing)
- this stub & cmake file
- checksum manifest + README
