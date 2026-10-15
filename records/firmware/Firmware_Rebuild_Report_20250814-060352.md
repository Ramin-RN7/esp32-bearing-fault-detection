
# Rebuilding & Flashing Custom MicroPython for M5Stack Core2 (ESP32, SPIRAM-16M)
**Author:** (your name)  
**Date:** 2025-08-14

This document records how we rebuilt MicroPython firmware for the **M5Stack Core2** board, integrated the **emlearn** stub module and **ulab**, resolved build issues, produced a **single merged image** suitable for Thonny, and verified the final firmware on device. It also lists the _important modules_ required for the project and the decisions/rationale behind key steps. This text is intentionally detailed so it can be summarized for a thesis report.

---

## 1) Target, Versions, and Goal

**Hardware:** M5Stack Core2 (ESP32, PSRAM)  
**Firmware:** MicroPython **v1.20.0** (board: `CORE2` / `spiram-16m`)  
**Toolchain (working):**
- ESP-IDF **4.4.x** (the copy _in the firmware repository_)
- `xtensa-esp32-elf-gcc` **8.4.0** (`esp-2020r3-8.4.0`)
- Python **3.10**
- `esptool` **4.7.0** (installed into the **system** Python via `--user`)

**Project goal:** ship a MicroPython build that includes the 7 must-have modules and links cleanly with an **`emlearn_iir`** stub to satisfy the C-side module table:

- `ulab` (with `ulab.numpy` / `ulab.scipy`)
- `emlearn_iir` (custom stub, builtin)
- `ubluetooth`
- `uasyncio`
- `network`
- `machine` / `umachine`
- `esp`

_Also available in our final build_: `emlearn_fft_c`, `emlearn_trees_c` (from `ses_modules/emlearn-micropython`), plus standard stdlib modules.

---

## 2) The Key Pitfall: ESP-IDF Mismatch

We initially hit this CMake error when CMake picked up a **newer** IDF (v6) from the global system:

```
Failed to resolve component 'esp_adc_cal' required by component 'main': unknown name.
HINT: The component 'esp_adc_cal' could not be found...
```

**Root cause:** running `idf.py` with **ESP-IDF v6** instead of **v4.4**. MicroPython port + our board config depends on the v4.4 layout (and its components).

**Fix:** explicitly use the **IDF 4.4** copy inside the repo and its toolchain:

```bash
# Clean slate (optional)
unset IDF_PATH IDF_TOOLS_PATH IDF_PYTHON_ENV_PATH

# Use the IDF shipped with the firmware (4.4.x)
export IDF_PATH="$HOME/m5-firmware/firmware/esp-idf"
export IDF_TOOLS_PATH="$HOME/.espressif-idf44"
source "$IDF_PATH/export.sh"   # adds 8.4.0 toolchain to PATH

# Sanity checks
which xtensa-esp32-elf-gcc       # .../esp-2020r3-8.4.0/xtensa-esp32-elf-gcc
xtensa-esp32-elf-gcc --version   # 8.4.0
```

This alone resolves the `esp_adc_cal` error and ensures the **correct libraries** are linked.

---

## 3) Fixing the Linker Error (Builtin Module Stub)

We saw (link stage):
```
undefined reference to `mp_module_emlearn_iir`
```

**Root cause:** `objmodule.c` registers `emlearn_iir` as a builtin, but no C module with that symbol was linked.

**Solution:** add a minimal C stub (`mod_emlearn_iir.c`) and wire it in via `usermod`.

### 3.1 C stub (`ses_modules/emlearn-micropython/src/emlearn_iir/mod_emlearn_iir.c`)
```c
#include "py/obj.h"
#include "py/runtime.h"
#include "py/objmodule.h"

STATIC const mp_rom_map_elem_t emlearn_iir_module_globals_table[] = {
    { MP_ROM_QSTR(MP_QSTR___name__), MP_ROM_QSTR(MP_QSTR_emlearn_iir) },
    // TODO: add real functions here later.
};

STATIC MP_DEFINE_CONST_DICT(emlearn_iir_module_globals, emlearn_iir_module_globals_table);

const mp_obj_module_t mp_module_emlearn_iir = {
    .base = { &mp_type_module },
    .globals = (mp_obj_dict_t*)&emlearn_iir_module_globals,
};

// IMPORTANT: use the 2-argument form for MicroPython v1.20's generator
MP_REGISTER_MODULE(MP_QSTR_emlearn_iir, mp_module_emlearn_iir);
```

> ⚠️ We briefly used a 3-argument form (`..., 1)`) which caused:
>
> `ERROR: Call to MP_REGISTER_MODULE(...) should be MP_REGISTER_MODULE(name, obj)`
>
> **Fix:** use the **2-argument** macro exactly as shown above.

### 3.2 Wire the stub into the build (`firmware/usermods/micropython.cmake`)
```cmake
# --- emlearn_iir stub (added) ---
add_library(usermod_emlearn_iir INTERFACE)
target_sources(usermod_emlearn_iir INTERFACE
    ${CMAKE_CURRENT_LIST_DIR}/../ses_modules/emlearn-micropython/src/emlearn_iir/mod_emlearn_iir.c
)
target_include_directories(usermod_emlearn_iir INTERFACE
    ${CMAKE_CURRENT_LIST_DIR}/../ses_modules/emlearn-micropython/src/emlearn_iir
)
target_link_libraries(usermod INTERFACE usermod_emlearn_iir)
# --- end emlearn_iir stub ---
```

This ensures the symbol `mp_module_emlearn_iir` is compiled and linked.

**Optional verification (during a successful build):**
```bash
# Confirm the stub got into the build graph
grep -F "mod_emlearn_iir.c" build/build.ninja

# Confirm the symbol is defined (D/T), not undefined (U)
nm build/*/*.a | grep mp_module_emlearn_iir
```

---

## 4) Building MicroPython (IDF 4.4 toolchain)

From the ESP32 port dir:

```bash
cd ~/m5-firmware/firmware/micropython/ports/esp32
rm -rf build build-CORE2 CMakeCache.txt

idf.py -DMICROPY_BOARD=CORE2 \
       -DMICROPY_FROZEN_MANIFEST="$HOME/combined_manifest.py" \
       -DUSER_C_MODULES="$HOME/m5-firmware/firmware/usermods/micropython.cmake" build
```

**Outputs of interest:**
- `build/micropython.bin`
- `build/bootloader/bootloader.bin`
- `build/partition_table/partition-table.bin`

**Non-fatal warnings observed:**
- `FastSpi.c: initialization discards 'const' qualifier...` (benign for us)
- `ili934x.c: comparison is always true due to limited range...` (benign)

---

## 5) Create a Single Merged Image (Thonny-friendly)

We used the **system** Python for `esptool` to avoid the IDF venv missing `esptool`:

```bash
# 1) Ensure binaries exist
cd ~/m5-firmware/firmware/micropython/ports/esp32
[ -f build/micropython.bin ] || idf.py build

# 2) Merge with system Python's esptool (installed via: python3 -m pip install --user esptool>=4.0)
SYS_PY=/usr/bin/python3
TS=$(date +%Y%m%d-%H%M%S)
OUT="$HOME/Desktop/micropython-thonny-$TS.bin"

$SYS_PY -m esptool --chip esp32 merge_bin -o "$OUT" \
  --flash_mode dio --flash_freq 40m --flash_size 4MB \
  0x1000  build/bootloader/bootloader.bin \
  0x8000  build/partition_table/partition-table.bin \
  0x10000 build/micropython.bin

# Tip: to silence the size warning, use --flash_size keep
```

This produces a **single** `.bin` that can be flashed at **0x0**.

---

## 6) Flashing

**Find the serial port:**

```bash
ls -l /dev/serial/by-id/      # stable symlink (preferred)
ls /dev/ttyUSB* /dev/ttyACM*  # quick check
dmesg | tail -n 50            # look for cdc_acm / CH340 / CP210x
```

**Give user access (one-time):**

```bash
sudo usermod -aG dialout "$USER"
newgrp dialout
# or temporarily:
sudo chmod a+rw /dev/ttyACM0
```

**Stop port grabbers (if any):**

```bash
sudo systemctl stop ModemManager
```

**Flash (merged at 0x0):**

```bash
PORT=/dev/ttyACM0
BAUD=460800

/usr/bin/python3 -m esptool --chip esp32 --port "$PORT" --baud $BAUD \
  write_flash -z --flash_size keep 0x0 "$OUT"
```

Expected log excerpt:
```
Features: WiFi, BT, Dual Core, 240MHz...
Wrote 1748000 bytes ... at 0x00000000 ...
Hash of data verified.
Leaving...
Hard resetting via RTS pin...
```

---

## 7) On-Device Verification

Open a REPL (screen/Thonny) and check:

```python
import sys, os
print(sys.implementation)
print(os.uname())
```

Observed:
```
(name='micropython', version=(1, 20, 0), _machine='Core2 (spiram-16m) with ESP32', _mpy=10502)
(sysname='esp32', nodename='esp32', release='1.20.0', version='v1.20.0 on 2025-08-14', machine='Core2 (spiram-16m) with ESP32')
```

**Import the 7 important modules:**
```python
mods = ["ulab","emlearn_iir","ubluetooth","uasyncio","network","machine","esp"]
for m in mods:
    try:
        __import__(m)
        print("✓", m)
    except Exception as e:
        print("✗", m, e)
```

**Quick `ulab` FFT sanity check:**
```python
from ulab import numpy as np
x = np.arange(8)
print(np.fft.fft(x))
```

**Module list snapshot (subset):**
```
__main__  emlearn_fft_c  emlearn_iir  emlearn_trees_c  ulab  ubluetooth  ...
```

_Note:_ `machine` may print as `umachine` in some lists; both resolve to the same core module namespace on ESP32 builds.

---

## 8) Issues & Fixes (Summary)

1. **CMake “esp_adc_cal unknown name”**  
   - **Cause:** IDF v6 path active.  
   - **Fix:** switch to repo’s **ESP-IDF v4.4**, toolchain gcc 8.4.0.

2. **Linker: `undefined reference to mp_module_emlearn_iir`**  
   - **Cause:** builtin table referenced a module not compiled.  
   - **Fix:** add `mod_emlearn_iir.c` stub and link via `usermod` CMake.

3. **`MP_REGISTER_MODULE` macro error**  
   - **Cause:** used 3-arg macro.  
   - **Fix:** use 2-argument form: `MP_REGISTER_MODULE(MP_QSTR_emlearn_iir, mp_module_emlearn_iir);`

4. **`esptool` not found in IDF venv**  
   - **Fix:** install/use system Python’s `esptool` and call with `/usr/bin/python3 -m esptool`.

5. **Serial port busy/missing**  
   - **Fixes:** use `/dev/serial/by-id/...`, add user to `dialout`, stop ModemManager, ensure correct port (`/dev/ttyACM0` in our case).

6. **Non-fatal warnings**  
   - `FastSpi.c` const-qualifier & `ili934x.c` type-limits: acknowledged and ignored.

---

## 9) Reproducibility Checklist

- [ ] IDF 4.4 selected (`xtensa-esp32-elf-gcc 8.4.0` in PATH)  
- [ ] `mod_emlearn_iir.c` present and registered (2-arg macro)  
- [ ] `usermods/micropython.cmake` links the stub into `usermod`  
- [ ] `idf.py build` produces `micropython.bin`  
- [ ] Merged image created with system `esptool`  
- [ ] Device flashed successfully at `0x0` with merged image  
- [ ] REPL imports for 7 modules succeed; `ulab` FFT works

---

## 10) Appendix A — Full `mod_emlearn_iir.c`

```c
#include "py/obj.h"
#include "py/runtime.h"
#include "py/objmodule.h"

STATIC const mp_rom_map_elem_t emlearn_iir_module_globals_table[] = {
    { MP_ROM_QSTR(MP_QSTR___name__), MP_ROM_QSTR(MP_QSTR_emlearn_iir) },
};

STATIC MP_DEFINE_CONST_DICT(emlearn_iir_module_globals, emlearn_iir_module_globals_table);

const mp_obj_module_t mp_module_emlearn_iir = {
    .base = { &mp_type_module },
    .globals = (mp_obj_dict_t*)&emlearn_iir_module_globals,
};

MP_REGISTER_MODULE(MP_QSTR_emlearn_iir, mp_module_emlearn_iir);
```

## 11) Appendix B — `usermods/micropython.cmake`

```cmake
add_library(usermod_emlearn_iir INTERFACE)
target_sources(usermod_emlearn_iir INTERFACE
    ${CMAKE_CURRENT_LIST_DIR}/../ses_modules/emlearn-micropython/src/emlearn_iir/mod_emlearn_iir.c
)
target_include_directories(usermod_emlearn_iir INTERFACE
    ${CMAKE_CURRENT_LIST_DIR}/../ses_modules/emlearn-micropython/src/emlearn_iir
)
target_link_libraries(usermod INTERFACE usermod_emlearn_iir)
```

---

## 12) Notes for Thesis Summary

- The decisive factor was **toolchain alignment**: MicroPython’s ESP32 port and the M5Stack Core2 board definition rely on **ESP-IDF 4.4**; mixing in newer IDF broke component resolution.
- The **linker fix** required introducing a **minimal C stub** so the builtin module table is consistent with linked objects.
- A **two-environment approach** worked best: IDF-managed Python for the build, system Python (with `esptool`) for merging/flashing.
- Post-flash verification demonstrated the presence of the **7 required modules** and functionality of `ulab` via FFT.

---

_End of report._
