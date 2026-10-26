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
        "warp_intrinsics": _warp_intrinsics(),
        "cuda_graphs": _graph_capabilities(),
        "nvcc_flags": _nvcc_flags(),
        "library_adapters": libraries.to_manifest(),
        "diagnostics": [],
    }


def _syntax_capabilities():
    return [
        _entry("cuda_triple_chevron_launch", "supported", signatures=("kernel<<<grid, block[, shared[, stream]]>>>(args...)",)),
        _entry("cudaLaunch_macro", "supported", signatures=("cudaLaunch(kernel, grid, block, args...)",)),
        _entry("__global__", "supported", category="qualifier"),
        _entry("__device__", "supported", category="qualifier"),
        _entry("__host__ __device__", "partial", category="qualifier", notes="Supported for device-callable lowering; host overload parity is limited."),
        _entry("__shared__", "supported", category="memory"),
        _entry("dynamic_shared_memory", "supported", category="memory"),
        _entry("dim3", "supported", category="type"),
        _entry("multi_file_project", "supported", category="project"),
        _entry("separate_compile_and_link", "supported", category="driver"),
        _entry("make_nvcc_shim", "supported", category="driver"),
        _entry("cuda_graphs", "partial", category="runtime", notes="Explicit kernel/copy/memset graph nodes and single-stream capture are supported for the documented subset; broad update remains planned."),
        _entry("warp_intrinsics", "partial", category="device_intrinsic", notes="supports full-mask shuffle, shuffle_down, any/all, activemask, and syncwarp; ballot and dynamic masks remain unsupported."),
    ]


def _runtime_apis():
    entries = [
        ("cudaMalloc", "cudaError_t(void**, size_t)"),
        ("cudaFree", "cudaError_t(void*)"),
        ("cudaMemcpy", "cudaError_t(void*, const void*, size_t, cudaMemcpyKind)"),
        ("cudaMemcpyAsync", "cudaError_t(void*, const void*, size_t, cudaMemcpyKind, cudaStream_t)"),
        ("cudaMemset", "cudaError_t(void*, int, size_t)"),
        ("cudaMemsetAsync", "cudaError_t(void*, int, size_t, cudaStream_t)"),
        ("cudaDeviceSynchronize", "cudaError_t()"),
        ("cudaStreamCreate", "cudaError_t(cudaStream_t*)"),
        ("cudaStreamDestroy", "cudaError_t(cudaStream_t)"),
        ("cudaStreamSynchronize", "cudaError_t(cudaStream_t)"),
        ("cudaGetLastError", "cudaError_t()"),
        ("cudaPeekAtLastError", "cudaError_t()"),
        ("cudaGetErrorString", "const char*(cudaError_t)"),
        ("cudaGetDeviceCount", "cudaError_t(int*)"),
        ("cudaGetDevice", "cudaError_t(int*)"),
        ("cudaSetDevice", "cudaError_t(int)"),
        ("cudaGetDeviceProperties", "cudaError_t(cudaDeviceProp*, int)"),
        ("cudaDeviceReset", "cudaError_t()"),
        ("cudaEventCreate", "cudaError_t(cudaEvent_t*)"),
        ("cudaEventCreateWithFlags", "cudaError_t(cudaEvent_t*, unsigned int)"),
        ("cudaEventRecord", "cudaError_t(cudaEvent_t, cudaStream_t)"),
        ("cudaEventSynchronize", "cudaError_t(cudaEvent_t)"),
        ("cudaEventElapsedTime", "cudaError_t(float*, cudaEvent_t, cudaEvent_t)"),
        ("cudaEventDestroy", "cudaError_t(cudaEvent_t)"),
        ("cudaOccupancyMaxPotentialBlockSize", "cudaError_t(int*, int*, Kernel, size_t, int)"),
        ("cudaGraphCreate", "cudaError_t(cudaGraph_t*, unsigned int)"),
        ("cudaGraphDestroy", "cudaError_t(cudaGraph_t)"),
        ("cudaGraphInstantiate", "cudaError_t(cudaGraphExec_t*, cudaGraph_t, cudaGraphNode_t*, char*, size_t)"),
        ("cudaGraphExecDestroy", "cudaError_t(cudaGraphExec_t)"),
        ("cudaGraphLaunch", "cudaError_t(cudaGraphExec_t, cudaStream_t)"),
        ("cudaGraphAddKernelNode", "cudaError_t(cudaGraphNode_t*, cudaGraph_t, const cudaGraphNode_t*, size_t, const cudaKernelNodeParams*)"),
        ("cudaGraphAddMemcpyNode", "cudaError_t(cudaGraphNode_t*, cudaGraph_t, const cudaGraphNode_t*, size_t, const cudaMemcpy3DParms*)"),
        ("cudaGraphAddMemsetNode", "cudaError_t(cudaGraphNode_t*, cudaGraph_t, const cudaGraphNode_t*, size_t, const cudaMemsetParams*)"),
        ("cudaGraphAddDependencies", "cudaError_t(cudaGraph_t, const cudaGraphNode_t*, const cudaGraphNode_t*, size_t)"),
        ("cudaGraphGetNodes", "cudaError_t(cudaGraph_t, cudaGraphNode_t*, size_t*)"),
        ("cudaGraphGetRootNodes", "cudaError_t(cudaGraph_t, cudaGraphNode_t*, size_t*)"),
        ("cudaStreamBeginCapture", "cudaError_t(cudaStream_t, unsigned int)"),
        ("cudaStreamEndCapture", "cudaError_t(cudaStream_t, cudaGraph_t*)"),
    ]
    return [
        _entry(name, _runtime_status(name), category="runtime_api", signatures=(signature,))
        for name, signature in entries
    ]


def _runtime_status(name):
    if name in {"cudaStreamBeginCapture", "cudaStreamEndCapture"}:
        return "partial"
    return "partial" if name.startswith("cudaGraph") else "supported"


def _warp_intrinsics():
    supported = {
        "__syncwarp": ("supported", [], ["0xffffffff"], [32], "Lowers to Metal simdgroup_barrier for the supported full-mask subset."),
        "__activemask": ("supported", ["unsigned int"], ["0xffffffff"], [32], "Returns the supported full-warp active mask."),
        "__shfl_sync": ("partial", ["int", "unsigned int", "float"], ["0xffffffff"], [32], "Lowers to Metal simd_shuffle for full-mask width-32 scalar values."),
        "__shfl_down_sync": ("partial", ["int", "unsigned int", "float"], ["0xffffffff"], [32], "Lowers to Metal simd_shuffle_down for full-mask width-32 scalar values."),
        "__any_sync": ("partial", ["int", "bool"], ["0xffffffff"], [32], "Lowers to Metal simd_any for full-mask predicates."),
        "__all_sync": ("partial", ["int", "bool"], ["0xffffffff"], [32], "Lowers to Metal simd_all for full-mask predicates."),
    }
    return [
        {
            "name": name,
            "category": "warp_intrinsic",
            "status": supported.get(name, ("planned", [], [], [], ""))[0],
            "supported_types": supported.get(name, ("planned", [], [], [], ""))[1],
            "supported_masks": supported.get(name, ("planned", [], [], [], ""))[2],
            "supported_widths": supported.get(name, ("planned", [], [], [], ""))[3],
            "target_requirements": ["macos", "metal"],
            "notes": supported.get(
                name,
                ("planned", [], [], [], "Rejected with UnsupportedWarpIntrinsic until a safe Metal mapping or emulation exists."),
            )[4],
        }
        for name in [
            "__syncwarp",
            "__activemask",
            "__shfl_sync",
            "__shfl_down_sync",
            "__ballot_sync",
            "__any_sync",
            "__all_sync",
        ]
    ]


def _graph_capabilities():
    return [
        {
            "name": "explicit_kernel_graph_nodes",
            "category": "cuda_graph",
            "status": "partial",
            "supported_node_kinds": ["kernel", "memcpy_1d", "memset_1d"],
            "supported_capture_modes": [],
            "unsupported": ["3d_memcpy_nodes", "2d_memset_nodes", "host_nodes", "broad_update"],
            "notes": "Kernel, one-dimensional memcpy, and byte memset nodes can be instantiated and replayed.",
        },
        {
            "name": "stream_capture",
            "category": "cuda_graph",
            "status": "partial",
            "supported_node_kinds": ["kernel", "memcpy_1d", "memset_1d"],
            "supported_capture_modes": ["global", "thread_local", "relaxed"],
            "unsupported": ["nested_capture", "cross_stream_capture_dependencies", "host_nodes"],
            "notes": "Single-stream capture records MCU kernel dispatch, cudaMemcpyAsync, and cudaMemsetAsync into a replayable graph.",
        },
    ]


def _intrinsics(registry):
    return [
        {
            "name": name,
            "category": rule["category"],
            "status": "supported",
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
        _entry("-c", "supported", category="nvcc_flag"),
        _entry("-dc", "supported", category="nvcc_flag"),
        _entry("-o", "supported", category="nvcc_flag"),
        _entry("-I", "supported", category="nvcc_flag"),
        _entry("-D", "supported", category="nvcc_flag"),
        _entry("-U", "supported", category="nvcc_flag"),
        _entry("-O0/-O1/-O2/-O3", "supported", category="nvcc_flag"),
        _entry("-M/-MM/-MD/-MMD/-MF/-MT", "supported", category="nvcc_flag"),
        _entry("-L/-l", "supported", category="nvcc_flag", notes="Supported for native link invocations."),
        _entry("-Xcompiler", "partial", category="nvcc_flag", notes="Only a conservative host flag allowlist is accepted."),
        _entry("--generate-code/-gencode", "planned", category="nvcc_flag"),
    ]


def _entry(name, status, category="syntax", signatures=(), notes=""):
    return {
        "name": name,
        "category": category,
        "status": status,
        "signatures": list(signatures),
        "target_requirements": ["macos", "metal"] if status in {"supported", "partial"} else [],
        "notes": notes,
    }
