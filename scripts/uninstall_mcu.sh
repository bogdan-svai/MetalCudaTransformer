#!/usr/bin/env bash
set -euo pipefail

INSTALL_DIR="${MCU_INSTALL_DIR:-$HOME/.local/bin}"
TARGET="$INSTALL_DIR/mcu"

if [[ -e "$TARGET" ]]; then
  rm "$TARGET"
  echo "Removed $TARGET"
else
  echo "No mcu executable found at $TARGET"
fi
