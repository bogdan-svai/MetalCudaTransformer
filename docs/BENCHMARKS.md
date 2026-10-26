# Performance measurements

Frontend latency, native compilation, process startup and GPU execution are
separate measurements. MCU's cache optimization affects compiler work; it does
not establish faster GPU computation or a comparison with NVIDIA CUDA.

The cache implementation is in [semantic.py](../core/frontend/semantic.py) and
[cache/store.py](../core/cache/store.py). Tests in
[test_compiler_correctness.py](../tests/test_compiler_correctness.py) cover cache
hits, content invalidation and reparsing changed translation units.

## Reproduce frontend cache measurements

```bash
./.venv/bin/python scripts/benchmark_frontend.py --samples 5 --warmups 1 \
  --output .mcu/reports/frontend.json
```

The default workload is `examples/sha256/app/main.cpp`, which discovers four files:
the host entry, a shared type header, the kernel and its device helper source.
Each sample uses an empty temporary semantic cache for one `mcu check --json`
process, then reuses that cache in a second process. The script verifies the
reported cache miss and cache hit, and enables strict semantic checking. It
measures `timings.phases_seconds.frontend`, excluding Python startup, linking,
Metal compilation and GPU execution. It does not flush operating-system caches.
One complete pair is discarded before five measured pairs.

The report retains samples, medians, cache conditions, source content hashes,
revision, working-tree status and machine metadata. Both measurements use the
same compiler and source files; the semantic cache state differs.

## Sample measurements

Measured on 2026-10-26: Apple M4 Max (Mac16,5), arm64,
macOS 26.6.2, Python 3.14.4, Xcode 27.0 (27A266a), Python clang 21.1.7.
The run used five sample pairs after one discarded pair:

| Sample | Empty semantic cache, ms | Populated semantic cache, ms |
| --- | ---: | ---: |
| 1 | 790.452 | 20.209 |
| 2 | 782.788 | 20.332 |
| 3 | 792.829 | 20.185 |
| 4 | 781.624 | 20.184 |
| 5 | 804.524 | 21.965 |
| Median | 790.452 | 20.209 |

These measurements cover the frontend phase for this workload. Source discovery,
headers, toolchain versions and OS cache state affect results. The command writes
samples and environment details to JSON. GPU execution requires separate timing.

## Workloads with CPU correctness checks

For the optimizer workload matrix, with Metal available:

```bash
python scripts/verify_optimization_matrix.py --only all \
  --tiers small medium large --samples 5 --warmups 3 \
  --output .mcu/reports/optimization-matrix.json
python scripts/verify_performance.py --samples 5 \
  --output .mcu/reports/performance.json
```

The matrix covers STREAM, HotSpot, reduction and histogram variants. The
performance collector covers SHA-256, streams, the limited cuBLAS adapter and
compatibility checks.

Record machine, SDK, command, inputs, warmups, sample count, cache state and
correctness results alongside timings. A batching comparison against synchronous
submission is not a CUDA-versus-Metal comparison. Adapted benchmark directories
retain their upstream attribution and are not official upstream benchmark runs.
