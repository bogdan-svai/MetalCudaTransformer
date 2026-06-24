import os
import tempfile
from pathlib import Path

from core.backends.host import rewrite_triple_chevrons


CUDA_STUBS = "\n".join([
    "#define __global__",
    "#define __device__",
    "#define __host__",
    '#define __shared__ __attribute__((annotate("mcu_shared")))',
    "struct dim3 { unsigned int x; unsigned int y; unsigned int z; dim3(unsigned int vx=1, unsigned int vy=1, unsigned int vz=1): x(vx), y(vy), z(vz) {} };",
    "extern dim3 threadIdx;",
    "extern dim3 blockIdx;",
    "extern dim3 blockDim;",
    "extern dim3 gridDim;",
    "enum cudaError_t { cudaSuccess = 0 };",
    "enum cudaMemcpyKind { cudaMemcpyHostToHost, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice, cudaMemcpyDefault };",
    "template <typename Kernel, typename... Args> cudaError_t cudaLaunch(Kernel kernel, dim3 blocks, dim3 threads, Args... args);",
    "template <typename Kernel, typename... Args> cudaError_t cudaLaunch(Kernel kernel, int blocks, int threads, Args... args);",
    "template <typename Kernel, typename... Args> cudaError_t cudaLaunchShared(Kernel kernel, dim3 blocks, dim3 threads, unsigned long shared, Args... args);",
    "template <typename Kernel, typename... Args> cudaError_t cudaLaunchShared(Kernel kernel, int blocks, int threads, unsigned long shared, Args... args);",
    "cudaError_t cudaMalloc(void**, unsigned long);",
    "template <typename T> cudaError_t cudaMalloc(T** pointer, unsigned long size);",
    "cudaError_t cudaFree(void*);",
    "cudaError_t cudaMemcpy(void*, const void*, unsigned long, cudaMemcpyKind);",
    "cudaError_t cudaMemset(void*, int, unsigned long);",
    "cudaError_t cudaDeviceSynchronize();",
    "cudaError_t cudaGetLastError();",
    "void __syncthreads();",
    "template <typename T> T atomicAdd(T*, T);",
    "template <typename T> T atomicCAS(T*, T, T);",
    "template <typename T> T atomicMin(T*, T);",
    "template <typename T> T atomicMax(T*, T);",
    "template <typename T> T atomicExch(T*, T);",
    "",
])


def probe_with_clang(sources, read_text, compile_context=None):
    try:
        from clang.cindex import Config, Index
    except Exception as exc:
        return "regex_fallback", [{
            "severity": "warning",
            "message": f"libclang Python bindings unavailable; using regex fallback ({exc}).",
            "file": None,
            "line": None,
        }]

    configured = _configure(Config)
    if not configured:
        return "regex_fallback", [{
            "severity": "warning",
            "message": "Native libclang library not found; using regex fallback.",
            "file": None,
            "line": None,
        }]

    diagnostics = []
    try:
        index = Index.create()
        with tempfile.NamedTemporaryFile("w", suffix=".cpp", delete=False, encoding="utf-8") as handle:
            virtual_path = handle.name
            handle.write(CUDA_STUBS)
            ordered_sources = sorted(
                sources,
                key=lambda path: (Path(path).suffix not in {".h", ".hpp"}, Path(path).as_posix()),
            )
            for source in ordered_sources:
                handle.write(f'\n#line 1 "{Path(source).as_posix()}"\n')
                probe_source = rewrite_triple_chevrons(read_text(source))
                for line in probe_source.splitlines():
                    stripped = line.lstrip()
                    is_local_include = stripped.startswith('#include "')
                    is_cuda_include = stripped.startswith("#include") and (
                        "cuda_runtime.h" in line or "cuda.h" in line
                    )
                    handle.write("\n" if is_local_include or is_cuda_include else line + "\n")
        try:
            args = ["-x", "c++", "-std=c++17"]
            context = compile_context or {}
            for path in context.get("include_paths", []):
                args.extend(["-I", path])
            args.extend(f"-D{definition}" for definition in context.get("definitions", []))
            args.extend(f"-U{name}" for name in context.get("undefinitions", []))
            sdk = "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk"
            resource_dir = "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib/clang/17"
            if os.path.isdir(sdk):
                args.extend(["-isysroot", sdk])
            if os.path.isdir(resource_dir):
                args.extend(["-resource-dir", resource_dir])
            tu = index.parse(virtual_path, args=args)
            for diagnostic in tu.diagnostics:
                if diagnostic.severity >= diagnostic.Warning:
                    diagnostics.append({
                        "severity": "error" if diagnostic.severity >= diagnostic.Error else "warning",
                        "message": str(diagnostic),
                        "file": str(diagnostic.location.file) if diagnostic.location.file else None,
                        "line": diagnostic.location.line or None,
                    })
        finally:
            try:
                os.remove(virtual_path)
            except OSError:
                pass
    except Exception as exc:
        diagnostics.append({
            "severity": "warning",
            "message": f"libclang parse failed; regex fallback metadata was used ({exc}).",
            "file": None,
            "line": None,
        })
        return "regex_fallback", diagnostics

    return "libclang_probe+regex_fallback", diagnostics


def _configure(config):
    candidates = [
        "/Library/Developer/CommandLineTools/usr/lib/libclang.dylib",
        "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib/libclang.dylib",
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            try:
                config.set_library_file(candidate)
                config.set_compatibility_check(False)
            except Exception:
                pass
            return True
    return False
