# MCU: CUDA-to-Metal Compiler and Runtime

MCU is an experimental compiler and runtime that lets a tested subset of CUDA
C++ run on Apple Silicon GPUs through Metal. Its purpose is to make small and
medium CUDA-style compute projects usable on Macs without requiring an NVIDIA
GPU or a complete rewrite in Metal Shading Language (MSL).

The project accepts CUDA/C++ source files, analyzes them with Clang, builds a
semantic intermediate representation (IR), applies validation and optimization
passes, emits Metal source, and connects the generated kernels to a native
Objective-C++ runtime. The runtime handles buffers, kernel dispatch, streams,
synchronization, atomics, events and a limited cuBLAS-compatible SGEMM adapter.
Multi-file projects are parsed per translation unit and linked through Clang
symbol identities, which allows device helper calls to be resolved across source
files while preserving original source locations for diagnostics and rewrites.

The command-line tool is called `mcu`. It can inspect a source tree, report
compatibility diagnostics, generate Metal artifacts, or build a native macOS
executable. The compiler can be installed once as a user-level command and then
run from any CUDA project directory:

```bash
mcu check .
mcu analyze . --json
mcu build .
mcu build . --native --output .mcu/my-project
```

Native execution requires macOS on Apple Silicon, Xcode, the Metal Toolchain,
Python and native libclang. The project is source-level and does not execute
PTX or provide a drop-in replacement for `nvcc`. Supported behavior is defined
by the implemented and tested CUDA subset rather than by the full CUDA language
or runtime specification.

The tested surface includes device functions, kernel launches, shared memory,
atomics, streams, events, warp intrinsics, CUDA graphs, selected memory
operations and limited cuBLAS integration. The repository includes examples for
numeric kernels, reductions, histograms, stencil workloads, SHA-256 and runtime
API behavior. Verification combines unit tests, native Metal builds, runtime
correctness checks, a 26-case CUDA compatibility suite and repeatable benchmark
tiers with correctness oracles.

MCU is best understood as a practical research tool and a compiler prototype,
not as general CUDA compatibility. Programs using unsupported language features,
libraries or assumptions about NVIDIA-specific hardware may be rejected or
require adaptation. Performance results depend on the Apple Silicon model,
Metal version and workload size. The architecture, usage instructions, support
scope and current verification results are documented in the other files under
`docs/`.

The implementation also includes persistent semantic caching and measured
optimizer passes, so repeated builds can reuse unchanged translation units while
preserving diagnostics and generated output.
