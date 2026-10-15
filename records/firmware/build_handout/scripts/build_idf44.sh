#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.."; pwd)"
export IDF_PATH="$ROOT/firmware/esp-idf"
export IDF_TOOLS_PATH="${IDF_TOOLS_PATH:-$HOME/.espressif-idf44}"
source "$IDF_PATH/export.sh"
cd "$ROOT/firmware/micropython/ports/esp32"
idf.py -DMICROPY_BOARD=CORE2 -DUSER_C_MODULES="$ROOT/usermods/micropython.cmake" build
