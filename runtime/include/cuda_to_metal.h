#pragma once

#include <array>
#include <cstddef>
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
};

enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4,
};

struct mcuStream;
using cudaStream_t = mcuStream*;

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
cudaError_t cudaDeviceSynchronize();
cudaError_t cudaStreamCreate(cudaStream_t* stream);
cudaError_t cudaStreamDestroy(cudaStream_t stream);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cudaGetLastError();
const char* cudaGetErrorString(cudaError_t error);
cudaError_t mcuSetArtifactDirectory(const char* path);

template <typename T>
cudaError_t cudaMalloc(T** device_ptr, std::size_t size) {
    return cudaMalloc(reinterpret_cast<void**>(device_ptr), size);
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
