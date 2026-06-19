import os
import tempfile
from pathlib import Path


CUDA_STUBS = "\n".join([
    "#define __global__",
    "#define __device__",
    "#define __host__",
    "struct dim3 { unsigned int x; unsigned int y; unsigned int z; dim3(unsigned int vx=1, unsigned int vy=1, unsigned int vz=1): x(vx), y(vy), z(vz) {} };",
    "extern dim3 threadIdx;",
    "extern dim3 blockIdx;",
    "extern dim3 blockDim;",
    "extern dim3 gridDim;",
    "template <typename Kernel, typename... Args> void cudaLaunch(Kernel kernel, dim3 blocks, dim3 threads, Args... args);",
    "template <typename Kernel, typename... Args> void cudaLaunch(Kernel kernel, int blocks, int threads, Args... args);",
    "",
])


def probe_with_clang(sources, read_text):
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
            for source in sources:
                handle.write(f'\n#line 1 "{Path(source).as_posix()}"\n')
                for line in read_text(source).splitlines():
                    handle.write("\n" if line.lstrip().startswith("#include") else line + "\n")
        try:
            tu = index.parse(virtual_path, args=["-x", "c++", "-std=c++17"])
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
