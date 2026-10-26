#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECK_ONLY=0
case "${1:-}" in
  --help|-h)
    echo "Usage: ./install.sh [--check]"
    echo "Create .venv, install Python dependencies, and install the mcu command."
    echo "--check checks prerequisites without changing files."
    echo "Set MCU_PYTHON to select Python and MCU_INSTALL_DIR to select the command directory."
    exit 0
    ;;
  --check) CHECK_ONLY=1 ;;
  "") ;;
  *) echo "Unknown argument: $1" >&2; exit 2 ;;
esac
if [[ $# -gt 1 ]]; then
  echo "Usage: ./install.sh [--check]" >&2
  exit 2
fi

if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  echo "MCU requires macOS on Apple Silicon. Run this script from a native arm64 shell." >&2
  exit 1
fi
PYTHON_BIN="${MCU_PYTHON:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python not found: $PYTHON_BIN. Install Python 3.11 or later." >&2
  exit 1
fi
"$PYTHON_BIN" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else "Python 3.11 or later is required.")'
if ! command -v xcrun >/dev/null 2>&1 || ! xcrun --find clang++ >/dev/null 2>&1; then
  echo "Install Xcode and select it with xcode-select before installing MCU." >&2
  exit 1
fi
if ! xcrun -sdk macosx metal --version; then
  echo "Metal compiler unavailable. Complete Xcode setup and run:" >&2
  echo "  xcodebuild -downloadComponent MetalToolchain" >&2
  exit 1
fi
if [[ "$CHECK_ONLY" == 1 ]]; then
  echo "Platform, Python, and compiler prerequisites are available."
  exit 0
fi

if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$ROOT/.venv"
fi
"$ROOT/.venv/bin/python" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else "Remove or replace the existing .venv with Python 3.11 or later.")'
"$ROOT/.venv/bin/python" -m pip install -r "$ROOT/requirements.txt"
(
  cd "$ROOT"
  "$ROOT/.venv/bin/python" -c 'from clang.cindex import Config, Index; from core.frontend.clang_probe import _configure; _configure(Config); Index.create(); print("libclang loaded.")'
)
bash "$ROOT/scripts/install_mcu.sh"
