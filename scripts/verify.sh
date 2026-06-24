#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "== pytest =="
python -m pytest -q

if command -v xcrun >/dev/null 2>&1; then
  echo
  echo "== stage5 runtime component syntax =="
  xcrun clang++ -std=c++17 -fobjc-arc -fsyntax-only \
    -Iruntime/include -Iruntime/src runtime/src/*.mm
fi

echo
echo "== stage1 check timing =="
/usr/bin/time -p ./mcu check examples/complex_app/app/main.cpp

echo
echo "== stage2 build timing =="
/usr/bin/time -p ./mcu build examples/complex_app/app/main.cpp

echo
echo "== stage2 artifact contents =="
unzip -l mcu_artifacts.zip

if command -v xcrun >/dev/null 2>&1; then
  echo
  echo "== stage2 metal syntax compile =="
  unzip -p mcu_artifacts.zip generated.metal > /tmp/mcu_generated.metal
  /usr/bin/time -p xcrun -sdk macosx metal /tmp/mcu_generated.metal -o /tmp/mcu_generated.air
else
  echo
  echo "xcrun not found; skipping native Metal compile."
fi

echo
echo "== stage3 coarsened build timing =="
/usr/bin/time -p ./mcu build examples/complex_app/app/main.cpp --coarsening 4

if command -v xcrun >/dev/null 2>&1; then
  echo
  echo "== stage3 metal syntax compile =="
  unzip -p mcu_artifacts.zip generated.metal > /tmp/mcu_coarsened.metal
  /usr/bin/time -p xcrun -sdk macosx metal /tmp/mcu_coarsened.metal -o /tmp/mcu_coarsened.air
fi

echo
echo "== stage3 automatic recovery =="
AUTO_DIR="$(mktemp -d)"
cat > "$AUTO_DIR/auto_coarsening.cu" <<'EOF'
__global__ void autoKernel(float* values, int count) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < count) {
        values[idx] = values[idx] + 1.0f;
    }
}

void host() {
    cudaLaunch(autoKernel, 1, 2048, 0, 2048);
}
EOF
./mcu build "$AUTO_DIR/auto_coarsening.cu"

if command -v xcrun >/dev/null 2>&1; then
  unzip -p mcu_artifacts.zip generated.metal > /tmp/mcu_auto_coarsening.metal
  /usr/bin/time -p xcrun -sdk macosx metal /tmp/mcu_auto_coarsening.metal -o /tmp/mcu_auto_coarsening.air
fi

if [[ "$(uname -s)" == "Darwin" ]] && command -v xcrun >/dev/null 2>&1; then
  echo
  echo "== stage5 native build and Metal validation =="
  /usr/bin/time -p ./mcu build examples/stage4_numeric/main.cu \
    --native --output /tmp/mcu_stage4_verify

  echo
  echo "== stage5 numeric regression runtime =="
  /usr/bin/time -p /tmp/mcu_stage4_verify/main

  echo
  echo "== stage6 multi-file SHA-256 native build =="
  /usr/bin/time -p ./mcu build examples/stage6_sha256/app/main.cpp \
    --native --output /tmp/mcu_stage6_verify

  echo
  echo "== stage6 SHA-256 Metal runtime =="
  /usr/bin/time -p /tmp/mcu_stage6_verify/main

  echo
  echo "== stage7 MCU object compile =="
  STAGE7_DIR="$(mktemp -d)"
  MCU_CACHE_DIR="$STAGE7_DIR/cache" /usr/bin/time -p ./mcu compile -dc \
    examples/stage4_numeric/main.cu -MMD -MF "$STAGE7_DIR/numeric.d" \
    -o "$STAGE7_DIR/numeric.o"

  echo
  echo "== stage7 object link and precompiled metallib =="
  MCU_CACHE_DIR="$STAGE7_DIR/cache" /usr/bin/time -p ./mcu compile \
    "$STAGE7_DIR/numeric.o" -o "$STAGE7_DIR/numeric"
  test -f "$STAGE7_DIR/numeric.mcu/generated.metallib"

  echo
  echo "== stage7 no-op build timing =="
  MCU_CACHE_DIR="$STAGE7_DIR/cache" /usr/bin/time -p ./mcu compile \
    "$STAGE7_DIR/numeric.o" -o "$STAGE7_DIR/numeric"

  echo
  echo "== stage7 native runtime without MSL JIT =="
  /usr/bin/time -p "$STAGE7_DIR/numeric"

  echo
  echo "== stage8 atomics, memory, and threadgroup integration =="
  /usr/bin/time -p python -m pytest -q tests/test_stage8_memory_atomics.py

  echo
  echo "== stage8 generic helper-to-shared atomic build =="
  MCU_CACHE_DIR="$STAGE7_DIR/cache" /usr/bin/time -p ./mcu build \
    examples/stage8_generic/main.cu --native --output /tmp/mcu_stage8_generic_verify

  echo
  echo "== stage8 signed/unsigned atomic metallib build =="
  MCU_CACHE_DIR="$STAGE7_DIR/cache" /usr/bin/time -p ./mcu build \
    examples/stage8_atomics/main.cu --native --output /tmp/mcu_stage8_atomics_verify

  echo
  echo "== stage8 generic helper-to-shared atomic runtime =="
  /usr/bin/time -p /tmp/mcu_stage8_generic_verify/main
fi
