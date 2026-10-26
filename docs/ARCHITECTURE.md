# Architecture

MCU translates a tested subset of CUDA C++ source to Metal and builds a native
host executable. It does not execute PTX or supply the full NVIDIA runtime.

## Frontend and IR

`core/frontend/project_parser.py` discovers sources and constructs project
metadata. Its Clang probe and textual extraction are both active.
`core/frontend/semantic.py` builds the typed semantic project, caches translation
units and relinks symbols. `source_calls.py` collects source identities and ranges
from original units for safe rewrites. Some analysis still parses compatibility
source with CUDA declarations; this is not complete CUDA legality checking.

`core/ir/models.py` contains project and routing metadata. `core/ir/symbols.py`
contains typed functions, statements, calls, effects and source ranges. These
representations coexist. Host-loop analysis has a separate opt-in IR.

## Validation and generation

`core/application/pipeline.py` coordinates checking, semantic parsing, device
linking, validation, optimization and generation. `core/passes/device_linker.py`
resolves device calls and address-space requirements. `core/backends/metal.py`
emits kernels and binding metadata; `device.py` lowers device helpers and calls;
`host.py` rewrites host launches and runtime operations.

Semantic IR guides supported transformations and validates source edits. Some
lowering and optimizer operations still inspect emitted text. Setting
`MCU_SEMANTIC_STRICT=1` rejects missing required semantic information; it does not
eliminate all textual paths.

## Build artifacts

A source-only build writes `mcu_artifacts.zip`: generated Metal, routing metadata,
symbol information and analysis output. A native build additionally writes host
copies and invokes Apple's Metal compiler and `metallib` linker, then compiles
Objective-C++ host/runtime sources and links Metal and required Apple frameworks.
See `core/build/native.py`. The generated executable loads the Metal library.
The compilation driver also supports precompiled libraries and runtime fallback
paths; this is source translation, not PTX JIT.

## Runtime

`runtime/src/buffer_registry.mm` tracks allocations and ranges.
`kernel_dispatcher.mm` binds arguments and launches kernels.
`stream_scheduler.mm` handles queues, pending work, synchronization and staging
buffer reuse. Events, atomics and graph operations implement tested subsets.
The limited cuBLAS adapter uses Metal Performance Shaders for SGEMM.

Explicit-stream batching preserves stream ordering and flushes at required
synchronization boundaries. It does not imply arbitrary CUDA stream equivalence.
GPU and command-buffer metrics are distinct from compiler frontend timings.

## Caching

`core/cache/store.py` verifies checksums and publishes cached files atomically.
Semantic keys include schema/version information, compilation context and source
content; translation units are cached independently. Header changes invalidate
relevant inputs, and affected device symbols are relinked. Native caches also
include generated content and build context. Cache hits are reported in tooling
JSON; tests exercise content changes and cache invalidation.

## Scope

The CLI capability inventory and regression tests define support. General C++,
CUDA libraries and NVIDIA hardware-specific assumptions may be rejected or need
adaptation. Unsupported cases must be diagnosed rather than inferred from a
successful source scan. See [testing](TESTING.md) and [usage](USAGE.md).
