# CudaMetalTransformer

CudaMetalTransformer is a CUDA-to-Metal compiler prototype for running
educational GPU workloads on Apple Silicon. It is designed for universities
that need a small, inspectable alternative to a full CUDA lab environment.

The project is an AST/IR compiler, not a collection of source substitutions:

```text
CUDA → libclang AST → typed IR → validated optimization passes → Metal lowering
```

The semantic IR tracks device functions, calls, pointer aliases, memory
effects, address spaces and statement structure. Optimizations are conservative
and produce structured reports. `MCU_SEMANTIC_STRICT=1` disables semantic
fallback when complete libclang information is unavailable.

## Quick start

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
./mcu check examples/complex_app/app/main.cpp
```

On macOS with the Metal toolchain installed:

```bash
TOOLCHAINS=com.apple.dt.toolchain.Metal.32023 make verify
```

## Benchmarks

The product suite covers STREAM, HotSpot, reduction and histogram kernels.
Run a reproducible optimizer matrix with correctness and regression gates:

```bash
python scripts/benchmark_optimization_suite.py \
  --only hotspot --samples 5 --warmups 3 \
  --output optimizer-quality-gate.json
python scripts/render_benchmark_report.py \
  optimizer-quality-gate.json --output optimizer-quality-gate.md
```

The matrix compares baseline, constant promotion, loop unrolling and the
combined safe configuration. The quality gate fails when a variant fails
correctness or regresses beyond the configured threshold (`5%` locally;
CI uses `20%` to account for shared runner variance).

Benchmark numbers are machine- and workload-specific. They are not presented
as CUDA/NVIDIA parity claims; the suite reports the actual Metal runtime and
correctness result on the host where it runs.

## Development quality gates

```bash
python -m pytest -q
make verify
```

CI runs the full verification suite, a reduced optimizer benchmark, and stores
the JSON/Markdown reports as build artifacts.

## Current limitations

Some compatibility parsing and legacy lowering paths still use textual
scanning. They are isolated behind the semantic frontend and cannot make
optimization or memory-safety decisions. The migration target is complete
Statement IR lowering, followed by removal of the compatibility parser from
device compilation.

See [planning/Architecture.md](planning/Architecture.md) for the compiler
contracts and migration order.

