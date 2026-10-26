# Supported features

MCU supports a subset of CUDA C++ source and runtime operations on Apple Silicon.
Use the capability inventory for supported signatures and restrictions:

```bash
mcu capabilities --json
```

## Compiler

The frontend combines source discovery, textual extraction and libclang semantic
analysis. Typed IR guides device linking, validation and source rewrites. Some
lowering and optimizer paths still inspect text. `MCU_SEMANTIC_STRICT=1` requires
the semantic information needed by a build; it does not remove all textual paths.

Device functions, kernel launches, multi-file linking, shared memory and selected
atomics are covered by regression tests. General CUDA C++, PTX and NVIDIA-specific
hardware behavior are outside the supported scope. A successful source check does
not establish successful native compilation or correct execution.

## Runtime

The Objective-C++ runtime provides tested subsets of allocation, copies,
synchronization, streams, events, warp intrinsics and CUDA graphs through Metal.
The cuBLAS adapter supports selected SGEMM operations through Metal Performance
Shaders; it does not provide general CUDA library compatibility.

## Automatic host-loop offload

Offload is disabled by default. The opt-in mode supports selected independent
loops over local unsigned arrays and immutable pointers derived from them.
Unknown external or mutable pointers, reductions, general calls, searches and
loops in headers are not offloaded. Eligible loops have no measured CPU/GPU
profitability threshold. The option is supported by native `build`, with the
NVCC-compatible driver and Makefile integration outside its current scope.

See [usage](USAGE.md#experimental-automatic-host-loop-offload) for guards and
fallback behavior.

## Verification and performance

[Testing](TESTING.md) describes source-only checks, native tests and CI.
[Benchmark methodology](BENCHMARKS.md) separates frontend, compilation and GPU
measurements. Native verification needs macOS on Apple Silicon, Xcode, the Metal
Toolchain and a usable GPU. Performance depends on workload size, hardware,
toolchain and cache state.
