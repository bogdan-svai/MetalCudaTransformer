from core.registry.catalog import default_registry
from core.registry.library_adapters import default_library_registry
from core.tooling.contract import CONTRACT_SCHEMA_VERSION, JSON_CONTRACT
from core.version import MCU_VERSION, RUNTIME_ABI


def capability_payload():
    registry = default_registry()
    libraries = default_library_registry()
    return {
        "ok": True,
        "command": "capabilities",
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "mcu_version": MCU_VERSION,
        "contract": JSON_CONTRACT,
        "runtime_abi": RUNTIME_ABI,
        "target": {
            "backend": "metal",
            "platform": "macos",
            "thread_execution_width": 32,
            "max_threads_per_threadgroup": 1024,
            "address_spaces": ["device", "threadgroup", "thread", "constant"],
        },
        "syntax": _syntax_capabilities(),
        "runtime_apis": _runtime_apis(),
        "intrinsics": _intrinsics(registry),
        "nvcc_flags": _nvcc_flags(),
        "library_adapters": libraries.to_manifest(),
        "diagnostics": [],
    }


def _syntax_capabilities():
    return [
        _entry("cuda_triple_chevron_launch", "supported", "1", signatures=("kernel<<<grid, block[, shared[, stream]]>>>(args...)",)),
        _entry("cudaLaunch_macro", "supported", "1", signatures=("cudaLaunch(kernel, grid, block, args...)",)),
        _entry("__global__", "supported", "1", category="qualifier"),
        _entry("__device__", "supported", "6", category="qualifier"),
        _entry("__host__ __device__", "partial", "6", category="qualifier", notes="Supported for device-callable lowering; host overload parity is limited."),
        _entry("__shared__", "supported", "8", category="memory"),
        _entry("dynamic_shared_memory", "supported", "8", category="memory"),
        _entry("dim3", "supported", "4", category="type"),
        _entry("multi_file_project", "supported", "6", category="project"),
        _entry("separate_compile_and_link", "supported", "7", category="driver"),
        _entry("make_nvcc_shim", "supported", "9", category="driver"),
        _entry("cuda_graphs", "planned", "14", category="runtime"),
        _entry("warp_intrinsics", "planned", "14", category="device_intrinsic"),
    ]


def _runtime_apis():
    entries = [
        ("cudaMalloc", "cudaError_t(void**, size_t)", "4"),
        ("cudaFree", "cudaError_t(void*)", "4"),
        ("cudaMemcpy", "cudaError_t(void*, const void*, size_t, cudaMemcpyKind)", "4"),
        ("cudaMemcpyAsync", "cudaError_t(void*, const void*, size_t, cudaMemcpyKind, cudaStream_t)", "10"),
        ("cudaMemset", "cudaError_t(void*, int, size_t)", "8"),
        ("cudaDeviceSynchronize", "cudaError_t()", "4"),
        ("cudaStreamCreate", "cudaError_t(cudaStream_t*)", "10"),
        ("cudaStreamDestroy", "cudaError_t(cudaStream_t)", "10"),
        ("cudaStreamSynchronize", "cudaError_t(cudaStream_t)", "10"),
        ("cudaGetLastError", "cudaError_t()", "4"),
        ("cudaPeekAtLastError", "cudaError_t()", "12"),
        ("cudaGetErrorString", "const char*(cudaError_t)", "12"),
        ("cudaGetDeviceCount", "cudaError_t(int*)", "12"),
        ("cudaGetDevice", "cudaError_t(int*)", "12"),
        ("cudaSetDevice", "cudaError_t(int)", "12"),
        ("cudaGetDeviceProperties", "cudaError_t(cudaDeviceProp*, int)", "12"),
        ("cudaDeviceReset", "cudaError_t()", "12"),
        ("cudaEventCreate", "cudaError_t(cudaEvent_t*)", "12"),
        ("cudaEventCreateWithFlags", "cudaError_t(cudaEvent_t*, unsigned int)", "12"),
        ("cudaEventRecord", "cudaError_t(cudaEvent_t, cudaStream_t)", "12"),
        ("cudaEventSynchronize", "cudaError_t(cudaEvent_t)", "12"),
        ("cudaEventElapsedTime", "cudaError_t(float*, cudaEvent_t, cudaEvent_t)", "12"),
        ("cudaEventDestroy", "cudaError_t(cudaEvent_t)", "12"),
        ("cudaOccupancyMaxPotentialBlockSize", "cudaError_t(int*, int*, Kernel, size_t, int)", "12"),
    ]
    return [
        _entry(name, "supported", stage, category="runtime_api", signatures=(signature,))
        for name, signature, stage in entries
    ]


def _intrinsics(registry):
    return [
        {
            "name": name,
            "category": rule["category"],
            "status": "supported",
            "stage": _intrinsic_stage(name),
            "capability": rule["capability"],
            "signatures": [
                f"{name}/{arity}" for arity in rule.get("supported_arities", [])
            ],
            "target": rule.get("target_symbol"),
            "notes": "",
        }
        for name, rule in registry.to_dict().items()
    ]


def _nvcc_flags():
    return [
        _entry("-c", "supported", "7", category="nvcc_flag"),
        _entry("-dc", "supported", "7", category="nvcc_flag"),
        _entry("-o", "supported", "7", category="nvcc_flag"),
        _entry("-I", "supported", "7", category="nvcc_flag"),
        _entry("-D", "supported", "7", category="nvcc_flag"),
        _entry("-U", "supported", "7", category="nvcc_flag"),
        _entry("-O0/-O1/-O2/-O3", "supported", "7", category="nvcc_flag"),
        _entry("-M/-MM/-MD/-MMD/-MF/-MT", "supported", "7", category="nvcc_flag"),
        _entry("-L/-l", "supported", "11", category="nvcc_flag", notes="Supported for native link invocations."),
        _entry("-Xcompiler", "partial", "7", category="nvcc_flag", notes="Only a conservative host flag allowlist is accepted."),
        _entry("--generate-code/-gencode", "planned", "future", category="nvcc_flag"),
    ]


def _entry(name, status, stage, category="syntax", signatures=(), notes=""):
    return {
        "name": name,
        "category": category,
        "status": status,
        "stage": str(stage),
        "signatures": list(signatures),
        "target_requirements": ["macos", "metal"] if status in {"supported", "partial"} else [],
        "notes": notes,
    }


def _intrinsic_stage(name):
    if name.startswith("atomic"):
        return "8"
    if name == "memset":
        return "8"
    if name == "__syncthreads":
        return "1"
    return "4"

