#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTALL_DIR="${MCU_INSTALL_DIR:-$HOME/.local/bin}"
TARGET="$INSTALL_DIR/mcu"
PYTHON_BIN="$REPO_ROOT/.venv/bin/python"

if [[ ! -x "$REPO_ROOT/mcu" ]]; then
  echo "mcu executable not found at: $REPO_ROOT/mcu" >&2
  exit 2
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3 || true)"
fi

if [[ -z "$PYTHON_BIN" ]]; then
  echo "python3 not found and project .venv is unavailable." >&2
  exit 2
fi

mkdir -p "$INSTALL_DIR"

cat > "$TARGET" <<EOF
#!/usr/bin/env bash
exec "$PYTHON_BIN" "$REPO_ROOT/mcu" "\$@"
EOF

chmod +x "$TARGET"

echo "Installed mcu -> $TARGET"
if [[ ":$PATH:" != *":$INSTALL_DIR:"* ]]; then
  echo "Add this to ~/.zshrc if mcu is not found:"
  echo "export PATH=\"$INSTALL_DIR:\$PATH\""
fi
