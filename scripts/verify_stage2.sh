#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "== pytest =="
python -m pytest -q

echo
echo "== stage1 check timing =="
/usr/bin/time -p ./mcu check examples/complex_app/app/main.cpp

echo
echo "== stage2 build timing =="
/usr/bin/time -p ./mcu build examples/complex_app/app/main.cpp

echo
echo "== artifact contents =="
unzip -l mcu_artifacts.zip

if command -v xcrun >/dev/null 2>&1; then
  echo
  echo "== metal syntax compile =="
  unzip -p mcu_artifacts.zip generated.metal > /tmp/mcu_generated.metal
  /usr/bin/time -p xcrun -sdk macosx metal /tmp/mcu_generated.metal -o /tmp/mcu_generated.air
else
  echo
  echo "xcrun not found; skipping native Metal compile."
fi
