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
    "struct cudaDeviceProp { char name[256]; unsigned long totalGlobalMem; unsigned long sharedMemPerBlock; int regsPerBlock; int warpSize; unsigned long memPitch; int maxThreadsPerBlock; int maxThreadsDim[3]; int maxGridSize[3]; int clockRate; unsigned long totalConstMem; int major; int minor; int textureAlignment; int deviceOverlap; int multiProcessorCount; int kernelExecTimeoutEnabled; int integrated; int canMapHostMemory; int computeMode; int concurrentKernels; int ECCEnabled; int pciBusID; int pciDeviceID; int tccDriver; int memoryClockRate; int memoryBusWidth; int l2CacheSize; int maxThreadsPerMultiProcessor; int unifiedAddressing; int managedMemory; int asyncEngineCount; };",
    "extern dim3 threadIdx;",
    "extern dim3 blockIdx;",
    "extern dim3 blockDim;",
    "extern dim3 gridDim;",
    "enum cudaError_t { cudaSuccess = 0, cudaErrorInvalidValue = 1, cudaErrorMemoryAllocation = 2, cudaErrorInitializationError = 3, cudaErrorLaunchFailure = 4, cudaErrorNotSupported = 5, cudaErrorInvalidDevice = 6, cudaErrorInvalidResourceHandle = 7 };",
    "enum cudaMemcpyKind { cudaMemcpyHostToHost, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice, cudaMemcpyDefault };",
    "enum cudaStreamCaptureMode { cudaStreamCaptureModeGlobal = 0, cudaStreamCaptureModeThreadLocal = 1, cudaStreamCaptureModeRelaxed = 2 };",
    "struct cudaExtent { unsigned long width; unsigned long height; unsigned long depth; };",
    "struct cudaPitchedPtr { void* ptr; unsigned long pitch; unsigned long xsize; unsigned long ysize; };",
    "inline cudaExtent make_cudaExtent(unsigned long width, unsigned long height, unsigned long depth) { return {width, height, depth}; }",
    "inline cudaPitchedPtr make_cudaPitchedPtr(void* ptr, unsigned long pitch, unsigned long xsize, unsigned long ysize) { return {ptr, pitch, xsize, ysize}; }",
    "struct cudaMemcpy3DParms { cudaPitchedPtr srcPtr; cudaPitchedPtr dstPtr; cudaExtent extent; cudaMemcpyKind kind; };",
    "struct cudaMemsetParams { void* dst; unsigned long pitch; unsigned int value; unsigned int elementSize; unsigned long width; unsigned long height; };",
    "struct mcuStream; using cudaStream_t = mcuStream*;",
    "struct mcuEvent; using cudaEvent_t = mcuEvent*;",
    "struct mcuGraph; using cudaGraph_t = mcuGraph*;",
    "struct mcuGraphExec; using cudaGraphExec_t = mcuGraphExec*;",
    "struct mcuGraphNode; using cudaGraphNode_t = mcuGraphNode*;",
    "enum cudaEventFlags { cudaEventDefault = 0, cudaEventBlockingSync = 1, cudaEventDisableTiming = 2, cudaEventInterprocess = 4 };",
    "struct cudaKernelNodeParams { void* func; dim3 gridDim; dim3 blockDim; unsigned int sharedMemBytes; void** kernelParams; void** extra; };",
    "struct mcuCublasContext; using cublasHandle_t = mcuCublasContext*;",
    "enum cublasStatus_t { CUBLAS_STATUS_SUCCESS = 0, CUBLAS_STATUS_NOT_INITIALIZED = 1, CUBLAS_STATUS_INVALID_VALUE = 7, CUBLAS_STATUS_EXECUTION_FAILED = 13, CUBLAS_STATUS_NOT_SUPPORTED = 15 };",
    "enum cublasOperation_t { CUBLAS_OP_N = 0, CUBLAS_OP_T = 1, CUBLAS_OP_C = 2 };",
    "enum cublasPointerMode_t { CUBLAS_POINTER_MODE_HOST = 0, CUBLAS_POINTER_MODE_DEVICE = 1 };",
    "template <typename Kernel, typename... Args> cudaError_t cudaLaunch(Kernel kernel, dim3 blocks, dim3 threads, Args... args);",
    "template <typename Kernel, typename... Args> cudaError_t cudaLaunch(Kernel kernel, int blocks, int threads, Args... args);",
    "template <typename Kernel, typename... Args> cudaError_t cudaLaunchShared(Kernel kernel, dim3 blocks, dim3 threads, unsigned long shared, Args... args);",
    "template <typename Kernel, typename... Args> cudaError_t cudaLaunchShared(Kernel kernel, int blocks, int threads, unsigned long shared, Args... args);",
    "template <typename Kernel, typename... Args> cudaError_t cudaLaunchSharedOnStream(Kernel kernel, dim3 blocks, dim3 threads, unsigned long shared, cudaStream_t stream, Args... args);",
    "template <typename Kernel, typename... Args> cudaError_t cudaLaunchSharedOnStream(Kernel kernel, int blocks, int threads, unsigned long shared, cudaStream_t stream, Args... args);",
    "cudaError_t cudaMalloc(void**, unsigned long);",
    "template <typename T> cudaError_t cudaMalloc(T** pointer, unsigned long size);",
    "cudaError_t cudaFree(void*);",
    "cudaError_t cudaMemcpy(void*, const void*, unsigned long, cudaMemcpyKind);",
    "cudaError_t cudaMemcpyAsync(void*, const void*, unsigned long, cudaMemcpyKind, cudaStream_t);",
    "cudaError_t cudaMemset(void*, int, unsigned long);",
    "cudaError_t cudaMemsetAsync(void*, int, unsigned long, cudaStream_t);",
    "cudaError_t cudaDeviceSynchronize();",
    "cudaError_t cudaStreamCreate(cudaStream_t*);",
    "cudaError_t cudaStreamDestroy(cudaStream_t);",
    "cudaError_t cudaStreamSynchronize(cudaStream_t);",
    "cudaError_t cudaGetLastError();",
    "cudaError_t cudaPeekAtLastError();",
    "cudaError_t cudaGetDeviceCount(int*);",
    "cudaError_t cudaGetDevice(int*);",
    "cudaError_t cudaSetDevice(int);",
    "cudaError_t cudaGetDeviceProperties(cudaDeviceProp*, int);",
    "cudaError_t cudaDeviceReset();",
    "cudaError_t cudaEventCreate(cudaEvent_t*);",
    "cudaError_t cudaEventCreateWithFlags(cudaEvent_t*, unsigned int);",
    "cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t);",
    "cudaError_t cudaEventSynchronize(cudaEvent_t);",
    "cudaError_t cudaEventElapsedTime(float*, cudaEvent_t, cudaEvent_t);",
    "cudaError_t cudaEventDestroy(cudaEvent_t);",
    "cudaError_t cudaGraphCreate(cudaGraph_t*, unsigned int);",
    "cudaError_t cudaGraphDestroy(cudaGraph_t);",
    "cudaError_t cudaGraphInstantiate(cudaGraphExec_t*, cudaGraph_t, cudaGraphNode_t*, char*, unsigned long);",
    "cudaError_t cudaGraphExecDestroy(cudaGraphExec_t);",
    "cudaError_t cudaGraphLaunch(cudaGraphExec_t, cudaStream_t);",
    "cudaError_t cudaGraphAddKernelNode(cudaGraphNode_t*, cudaGraph_t, const cudaGraphNode_t*, unsigned long, const cudaKernelNodeParams*);",
    "cudaError_t cudaGraphAddDependencies(cudaGraph_t, const cudaGraphNode_t*, const cudaGraphNode_t*, unsigned long);",
    "cudaError_t cudaGraphGetNodes(cudaGraph_t, cudaGraphNode_t*, unsigned long*);",
    "cudaError_t cudaGraphGetRootNodes(cudaGraph_t, cudaGraphNode_t*, unsigned long*);",
    "cudaError_t cudaGraphAddMemcpyNode(cudaGraphNode_t*, cudaGraph_t, const cudaGraphNode_t*, unsigned long, const cudaMemcpy3DParms*);",
    "cudaError_t cudaGraphAddMemsetNode(cudaGraphNode_t*, cudaGraph_t, const cudaGraphNode_t*, unsigned long, const cudaMemsetParams*);",
    "cudaError_t cudaStreamBeginCapture(cudaStream_t, unsigned int);",
    "cudaError_t cudaStreamEndCapture(cudaStream_t, cudaGraph_t*);",
    "void* mcuKernel(const char*);",
    "template <typename Kernel> cudaError_t cudaOccupancyMaxPotentialBlockSize(int*, int*, Kernel, unsigned long = 0, int = 0);",
    "cublasStatus_t cublasCreate(cublasHandle_t*);",
    "cublasStatus_t cublasDestroy(cublasHandle_t);",
    "cublasStatus_t cublasSetStream(cublasHandle_t, cudaStream_t);",
    "cublasStatus_t cublasGetStream(cublasHandle_t, cudaStream_t*);",
    "cublasStatus_t cublasSetPointerMode(cublasHandle_t, cublasPointerMode_t);",
    "cublasStatus_t cublasGetPointerMode(cublasHandle_t, cublasPointerMode_t*);",
    "cublasStatus_t cublasSgemm(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const float*, const float*, int, const float*, int, const float*, float*, int);",
    "void __syncthreads();",
    "void __syncwarp(unsigned int = 0xffffffffu);",
    "unsigned int __activemask();",
    "template <typename T> T __shfl_sync(unsigned int, T, int, int = 32);",
    "template <typename T> T __shfl_down_sync(unsigned int, T, unsigned int, int = 32);",
    "unsigned int __ballot_sync(unsigned int, int);",
    "int __any_sync(unsigned int, int);",
    "int __all_sync(unsigned int, int);",
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
                        "cuda_runtime.h" in line or "cuda.h" in line or "cublas_v2.h" in line
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
