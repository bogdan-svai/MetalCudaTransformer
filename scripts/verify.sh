#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ "$(uname -s)" == "Darwin" ]]; then
  echo "== Metal toolchain =="
  xcrun -sdk macosx metal --version
fi

echo "== pytest =="
python -m pytest -q

if command -v xcrun >/dev/null 2>&1; then
  echo
  echo "== runtime component syntax =="
  xcrun clang++ -std=c++17 -fobjc-arc -fsyntax-only \
    -Iruntime/include -Iruntime/src runtime/src/*.mm
fi

echo
echo "== check timing =="
/usr/bin/time -p ./mcu check examples/complex_app/app/main.cpp

echo
echo "== build timing =="
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

echo
echo "== coarsened build timing =="
/usr/bin/time -p ./mcu build examples/complex_app/app/main.cpp --coarsening 4

if command -v xcrun >/dev/null 2>&1; then
  echo
  echo "== metal syntax compile =="
  unzip -p mcu_artifacts.zip generated.metal > /tmp/mcu_coarsened.metal
  /usr/bin/time -p xcrun -sdk macosx metal /tmp/mcu_coarsened.metal -o /tmp/mcu_coarsened.air
fi

echo
echo "== automatic recovery =="
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
  echo "== native build and Metal validation =="
  /usr/bin/time -p ./mcu build examples/numeric/main.cu \
    --native --output /tmp/mcu_numeric_verify

  echo
  echo "== numeric regression runtime =="
  /usr/bin/time -p /tmp/mcu_numeric_verify/main

  echo
  echo "== multi-file SHA-256 native build =="
  /usr/bin/time -p ./mcu build examples/sha256/app/main.cpp \
    --native --output /tmp/mcu_sha256_verify

  echo
  echo "== SHA-256 Metal runtime =="
  /usr/bin/time -p /tmp/mcu_sha256_verify/main

  echo
  echo "== MCU object compile =="
  TOOLCHAIN_DIR="$(mktemp -d)"
  MCU_CACHE_DIR="$TOOLCHAIN_DIR/cache" /usr/bin/time -p ./mcu compile -dc \
    examples/numeric/main.cu -MMD -MF "$TOOLCHAIN_DIR/numeric.d" \
    -o "$TOOLCHAIN_DIR/numeric.o"

  echo
  echo "== object link and precompiled metallib =="
  MCU_CACHE_DIR="$TOOLCHAIN_DIR/cache" /usr/bin/time -p ./mcu compile \
    "$TOOLCHAIN_DIR/numeric.o" -o "$TOOLCHAIN_DIR/numeric"
  test -f "$TOOLCHAIN_DIR/numeric.mcu/generated.metallib"

  echo
  echo "== no-op build timing =="
  MCU_CACHE_DIR="$TOOLCHAIN_DIR/cache" /usr/bin/time -p ./mcu compile \
    "$TOOLCHAIN_DIR/numeric.o" -o "$TOOLCHAIN_DIR/numeric"

  echo
  echo "== native runtime without MSL JIT =="
  /usr/bin/time -p "$TOOLCHAIN_DIR/numeric"

  echo
  echo "== atomics, memory, and threadgroup integration =="
  /usr/bin/time -p python -m pytest -q tests/test_memory_atomics.py

  echo
  echo "== generic helper-to-shared atomic build =="
  MCU_CACHE_DIR="$TOOLCHAIN_DIR/cache" /usr/bin/time -p ./mcu build \
    examples/shared_reduction/main.cu --native --output /tmp/mcu_shared_reduction_verify

  echo
  echo "== signed/unsigned atomic metallib build =="
  MCU_CACHE_DIR="$TOOLCHAIN_DIR/cache" /usr/bin/time -p ./mcu build \
    examples/atomics/main.cu --native --output /tmp/mcu_atomics_verify

  echo
  echo "== generic helper-to-shared atomic runtime =="
  /usr/bin/time -p /tmp/mcu_shared_reduction_verify/main

  echo
  echo "== project compatibility and runtime metrics =="
  /usr/bin/time -p python scripts/verify_sha256.py --samples 5 \
    --output /tmp/mcu_sha256_report.json

  echo
  echo "== asynchronous scheduler tiers =="
  /usr/bin/time -p python scripts/verify_streams.py --samples 5 \
    --output /tmp/mcu_streams_report.json

  echo
  echo "== cuBLAS adapter =="
  /usr/bin/time -p python scripts/verify_cublas.py

  echo
  echo "== CUDA easy-to-stress checker (26 cases) =="
  /usr/bin/time -p python scripts/cuda_checker.py --tier all \
    --output /tmp/mcu_cuda_checker.json
fi
