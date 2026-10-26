#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR="${MCU_SOURCE_DIR:-$HOME/.local/share/mcu}"
SOURCE_REF="${MCU_SOURCE_REF:-master}"

if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  echo "MCU requires macOS on Apple Silicon and a native arm64 shell." >&2
  exit 1
fi
for command in curl tar mktemp; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "Required command not found: $command" >&2
    exit 1
  fi
done
if [[ -e "$SOURCE_DIR" || -L "$SOURCE_DIR" ]]; then
  echo "Source directory already exists: $SOURCE_DIR" >&2
  echo "Run its install.sh again or choose a new MCU_SOURCE_DIR." >&2
  exit 1
fi

TEMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/mcu-download.XXXXXX")"
trap 'rm -rf "$TEMP_DIR"' EXIT
curl --fail --show-error --silent --location \
  "https://codeload.github.com/bogdan-svai/MetalCudaTransformer/tar.gz/$SOURCE_REF" \
  --output "$TEMP_DIR/source.tar.gz"
mkdir "$TEMP_DIR/source"
tar -xzf "$TEMP_DIR/source.tar.gz" -C "$TEMP_DIR/source" --strip-components=1
if [[ ! -f "$TEMP_DIR/source/install.sh" || ! -f "$TEMP_DIR/source/mcu" ]]; then
  echo "The downloaded archive does not contain the MCU installer." >&2
  exit 1
fi
(
  cd "$TEMP_DIR/source"
  bash ./install.sh --check
)
mkdir -p "$(dirname "$SOURCE_DIR")"
mkdir "$SOURCE_DIR"
cp -R "$TEMP_DIR/source/." "$SOURCE_DIR/"
bash "$SOURCE_DIR/install.sh"
