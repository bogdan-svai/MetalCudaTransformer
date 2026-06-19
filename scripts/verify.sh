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
cat > /tmp/mcu_auto_coarsening.cu <<'EOF'
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
./mcu build /tmp/mcu_auto_coarsening.cu

if command -v xcrun >/dev/null 2>&1; then
  unzip -p mcu_artifacts.zip generated.metal > /tmp/mcu_auto_coarsening.metal
  /usr/bin/time -p xcrun -sdk macosx metal /tmp/mcu_auto_coarsening.metal -o /tmp/mcu_auto_coarsening.air
fi
