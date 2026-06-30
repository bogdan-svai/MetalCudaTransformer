#pragma once

#include <array>
#include <cstddef>
#include <cstring>
#include <type_traits>
#include <utility>

struct dim3 {
    unsigned int x;
    unsigned int y;
    unsigned int z;

    constexpr dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1)
        : x(vx), y(vy), z(vz) {}
};

enum cudaError_t {
    cudaSuccess = 0,
    cudaErrorInvalidValue = 1,
    cudaErrorMemoryAllocation = 2,
    cudaErrorInitializationError = 3,
    cudaErrorLaunchFailure = 4,
    cudaErrorNotSupported = 5,
    cudaErrorInvalidDevice = 6,
    cudaErrorInvalidResourceHandle = 7,
};

enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4,
};

enum cudaStreamCaptureMode {
    cudaStreamCaptureModeGlobal = 0,
    cudaStreamCaptureModeThreadLocal = 1,
    cudaStreamCaptureModeRelaxed = 2,
};

struct cudaPos {
    std::size_t x;
    std::size_t y;
    std::size_t z;
};

struct cudaExtent {
    std::size_t width;
    std::size_t height;
    std::size_t depth;
};

struct cudaPitchedPtr {
    void* ptr;
    std::size_t pitch;
    std::size_t xsize;
    std::size_t ysize;
};

inline cudaPos make_cudaPos(std::size_t x, std::size_t y, std::size_t z) {
    return {x, y, z};
}

inline cudaExtent make_cudaExtent(std::size_t width, std::size_t height, std::size_t depth) {
    return {width, height, depth};
}

inline cudaPitchedPtr make_cudaPitchedPtr(void* ptr, std::size_t pitch, std::size_t xsize, std::size_t ysize) {
    return {ptr, pitch, xsize, ysize};
}

struct cudaMemcpy3DParms {
    cudaPitchedPtr srcPtr{};
    cudaPitchedPtr dstPtr{};
    cudaExtent extent{};
    cudaMemcpyKind kind = cudaMemcpyDefault;
};

struct cudaMemsetParams {
    void* dst;
    std::size_t pitch;
    unsigned int value;
    unsigned int elementSize;
    std::size_t width;
    std::size_t height;
};

struct mcuStream;
using cudaStream_t = mcuStream*;
struct mcuEvent;
using cudaEvent_t = mcuEvent*;
struct mcuGraph;
using cudaGraph_t = mcuGraph*;
struct mcuGraphExec;
using cudaGraphExec_t = mcuGraphExec*;
struct mcuGraphNode;
using cudaGraphNode_t = mcuGraphNode*;

enum cudaEventFlags {
    cudaEventDefault = 0,
    cudaEventBlockingSync = 1,
    cudaEventDisableTiming = 2,
    cudaEventInterprocess = 4,
};

struct cudaDeviceProp {
    char name[256];
    std::size_t totalGlobalMem;
    std::size_t sharedMemPerBlock;
    int regsPerBlock;
    int warpSize;
    std::size_t memPitch;
    int maxThreadsPerBlock;
    int maxThreadsDim[3];
    int maxGridSize[3];
    int clockRate;
    std::size_t totalConstMem;
    int major;
    int minor;
    int textureAlignment;
    int deviceOverlap;
    int multiProcessorCount;
    int kernelExecTimeoutEnabled;
    int integrated;
    int canMapHostMemory;
    int computeMode;
    int concurrentKernels;
    int ECCEnabled;
    int pciBusID;
    int pciDeviceID;
    int tccDriver;
    int memoryClockRate;
    int memoryBusWidth;
    int l2CacheSize;
    int maxThreadsPerMultiProcessor;
    int unifiedAddressing;
    int managedMemory;
    int asyncEngineCount;
};

struct cudaKernelNodeParams {
    void* func;
    dim3 gridDim;
    dim3 blockDim;
    unsigned int sharedMemBytes;
    void** kernelParams;
    void** extra;
};

cudaError_t cudaMalloc(void** device_ptr, std::size_t size);
cudaError_t cudaFree(void* device_ptr);
cudaError_t cudaMemcpy(void* dst, const void* src, std::size_t count, cudaMemcpyKind kind);
cudaError_t cudaMemcpyAsync(
    void* dst,
    const void* src,
    std::size_t count,
    cudaMemcpyKind kind,
    cudaStream_t stream = nullptr
);
cudaError_t cudaMemset(void* dst, int value, std::size_t count);
cudaError_t cudaMemsetAsync(void* dst, int value, std::size_t count, cudaStream_t stream = nullptr);
cudaError_t cudaDeviceSynchronize();
cudaError_t cudaStreamCreate(cudaStream_t* stream);
cudaError_t cudaStreamDestroy(cudaStream_t stream);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cudaGetLastError();
cudaError_t cudaPeekAtLastError();
const char* cudaGetErrorString(cudaError_t error);
cudaError_t cudaGetDeviceCount(int* count);
cudaError_t cudaGetDevice(int* device);
cudaError_t cudaSetDevice(int device);
cudaError_t cudaGetDeviceProperties(cudaDeviceProp* properties, int device);
cudaError_t cudaDeviceReset();
cudaError_t cudaEventCreate(cudaEvent_t* event);
cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags);
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream = nullptr);
cudaError_t cudaEventSynchronize(cudaEvent_t event);
cudaError_t cudaEventElapsedTime(float* milliseconds, cudaEvent_t start, cudaEvent_t end);
cudaError_t cudaEventDestroy(cudaEvent_t event);
cudaError_t cudaGraphCreate(cudaGraph_t* graph, unsigned int flags);
cudaError_t cudaGraphDestroy(cudaGraph_t graph);
cudaError_t cudaGraphInstantiate(
    cudaGraphExec_t* graph_exec,
    cudaGraph_t graph,
    cudaGraphNode_t* error_node,
    char* log_buffer,
    std::size_t buffer_size
);
cudaError_t cudaGraphExecDestroy(cudaGraphExec_t graph_exec);
cudaError_t cudaGraphLaunch(cudaGraphExec_t graph_exec, cudaStream_t stream);
cudaError_t cudaGraphAddKernelNode(
    cudaGraphNode_t* node,
    cudaGraph_t graph,
    const cudaGraphNode_t* dependencies,
    std::size_t dependency_count,
    const cudaKernelNodeParams* parameters
);
cudaError_t cudaGraphAddDependencies(
    cudaGraph_t graph,
    const cudaGraphNode_t* from,
    const cudaGraphNode_t* to,
    std::size_t dependency_count
);
cudaError_t cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t* nodes, std::size_t* count);
cudaError_t cudaGraphGetRootNodes(cudaGraph_t graph, cudaGraphNode_t* nodes, std::size_t* count);
cudaError_t cudaGraphAddMemcpyNode(
    cudaGraphNode_t* node,
    cudaGraph_t graph,
    const cudaGraphNode_t* dependencies,
    std::size_t dependency_count,
    const cudaMemcpy3DParms* parameters
);
cudaError_t cudaGraphAddMemsetNode(
    cudaGraphNode_t* node,
    cudaGraph_t graph,
    const cudaGraphNode_t* dependencies,
    std::size_t dependency_count,
    const cudaMemsetParams* parameters
);
cudaError_t cudaStreamBeginCapture(cudaStream_t stream, unsigned int mode);
cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t* graph);
cudaError_t mcuSetArtifactDirectory(const char* path);
void* mcuKernel(const char* kernel_name);

template <typename T>
cudaError_t cudaMalloc(T** device_ptr, std::size_t size) {
    return cudaMalloc(reinterpret_cast<void**>(device_ptr), size);
}

template <typename Kernel>
cudaError_t cudaOccupancyMaxPotentialBlockSize(
    int* min_grid_size,
    int* block_size,
    Kernel,
    std::size_t dynamic_shared_memory_size = 0,
    int block_size_limit = 0
) {
    if (!min_grid_size || !block_size || dynamic_shared_memory_size > 48 * 1024) {
        return cudaErrorInvalidValue;
    }
    int limit = block_size_limit > 0 && block_size_limit < 1024 ? block_size_limit : 1024;
    *block_size = limit;
    *min_grid_size = 1;
    return cudaSuccess;
}

namespace mcu {

enum class ArgumentKind { Buffer, Scalar };

struct KernelArgument {
    ArgumentKind kind;
    const void* scalar_data;
    std::size_t scalar_size;
    void* buffer_pointer;
};

cudaError_t dispatch(
    const char* kernel_name,
    dim3 grid,
    dim3 block,
    const KernelArgument* arguments,
    std::size_t argument_count,
    std::size_t shared_bytes = 0,
    cudaStream_t stream = nullptr
);

namespace detail {

template <typename T>
KernelArgument pack_argument(T&& value) {
    using Value = std::remove_reference_t<T>;
    if constexpr (std::is_pointer_v<Value>) {
        return {ArgumentKind::Buffer, nullptr, 0, const_cast<void*>(reinterpret_cast<const void*>(value))};
    } else {
        return {ArgumentKind::Scalar, &value, sizeof(Value), nullptr};
    }
}

}  // namespace detail

template <typename... Args>
cudaError_t launch(const char* kernel_name, dim3 grid, dim3 block, Args&&... args) {
    std::array<KernelArgument, sizeof...(Args)> packed = {
        detail::pack_argument(std::forward<Args>(args))...
    };
    return dispatch(kernel_name, grid, block, packed.data(), packed.size());
}

template <typename... Args>
cudaError_t launch_shared(
    const char* kernel_name,
    dim3 grid,
    dim3 block,
    std::size_t shared_bytes,
    Args&&... args
) {
    std::array<KernelArgument, sizeof...(Args)> packed = {
        detail::pack_argument(std::forward<Args>(args))...
    };
    return dispatch(kernel_name, grid, block, packed.data(), packed.size(), shared_bytes);
}

template <typename... Args>
cudaError_t launch_on_stream(
    const char* kernel_name,
    dim3 grid,
    dim3 block,
    cudaStream_t stream,
    Args&&... args
) {
    std::array<KernelArgument, sizeof...(Args)> packed = {
        detail::pack_argument(std::forward<Args>(args))...
    };
    return dispatch(kernel_name, grid, block, packed.data(), packed.size(), 0, stream);
}

template <typename... Args>
cudaError_t launch_shared_on_stream(
    const char* kernel_name,
    dim3 grid,
    dim3 block,
    std::size_t shared_bytes,
    cudaStream_t stream,
    Args&&... args
) {
    std::array<KernelArgument, sizeof...(Args)> packed = {
        detail::pack_argument(std::forward<Args>(args))...
    };
    return dispatch(
        kernel_name, grid, block, packed.data(), packed.size(), shared_bytes, stream
    );
}

}  // namespace mcu

#define cudaLaunch(kernel, grid, block, ...) \
    ::mcu::launch(#kernel, (grid), (block), ##__VA_ARGS__)
#define cudaLaunchShared(kernel, grid, block, shared, ...) \
    ::mcu::launch_shared(#kernel, (grid), (block), (shared), ##__VA_ARGS__)
#define cudaLaunchOnStream(kernel, grid, block, stream, ...) \
    ::mcu::launch_on_stream(#kernel, (grid), (block), (stream), ##__VA_ARGS__)
#define cudaLaunchSharedOnStream(kernel, grid, block, shared, stream, ...) \
    ::mcu::launch_shared_on_stream( \
        #kernel, (grid), (block), (shared), (stream), ##__VA_ARGS__ \
    )
