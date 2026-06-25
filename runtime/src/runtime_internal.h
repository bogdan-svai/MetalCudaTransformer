#pragma once

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "cublas_v2.h"
#include "cuda_to_metal.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

struct mcuStream {
    std::uint64_t identifier;
};

struct mcuCublasContext {
    cudaStream_t stream = nullptr;
    cublasPointerMode_t pointer_mode = CUBLAS_POINTER_MODE_HOST;
    bool alive = true;
};

namespace mcu::runtime {

void recordRuntimeMetric(const char* phase, double seconds);
void incrementRuntimeCounter(const char* name, std::uint64_t amount = 1);
void setRuntimeDeviceName(const char* name);

class DeviceContext {
public:
    bool ensure(std::string& error);
    id<MTLDevice> device() const { return device_; }
    id<MTLCommandQueue> queue() const { return queue_; }

private:
    id<MTLDevice> device_ = nil;
    id<MTLCommandQueue> queue_ = nil;
};

class BufferRegistry {
public:
    bool allocate(id<MTLDevice> device, void** output, std::size_t size, std::string& error);
    bool release(void* pointer, std::string& error);
    bool copy(void* dst, const void* src, std::size_t count, cudaMemcpyKind kind, std::string& error);
    bool fill(void* dst, int value, std::size_t count, std::string& error);
    bool find(
        const void* pointer,
        std::size_t count,
        id<MTLBuffer> __strong& buffer,
        NSUInteger& offset
    ) const;
    bool resolveCopyKind(
        const void* dst,
        const void* src,
        std::size_t count,
        cudaMemcpyKind requested,
        cudaMemcpyKind& resolved,
        std::string& error
    ) const;

private:
    struct Allocation {
        id<MTLBuffer> buffer;
        std::size_t size;
    };

    bool validateCopyRange(
        const void* pointer,
        std::size_t count,
        cudaMemcpyKind kind,
        bool destination,
        std::string& error
    ) const;
    bool contains(const void* pointer, std::size_t count) const;

    std::unordered_map<void*, Allocation> allocations_;
};

struct PendingCommand {
    id<MTLCommandBuffer> command = nil;
    id<MTLBuffer> staging = nil;
    void* host_destination = nullptr;
    std::size_t copy_size = 0;
    std::string operation;
};

struct StreamState {
    id<MTLCommandQueue> queue = nil;
    std::vector<PendingCommand> pending;
};

class StreamRegistry {
public:
    ~StreamRegistry();
    bool ensureDefault(id<MTLCommandQueue> queue, std::string& error);
    bool create(id<MTLDevice> device, cudaStream_t* output, std::string& error);
    bool destroy(cudaStream_t stream, std::string& error);
    bool prepare(cudaStream_t stream, std::string& error);
    bool enqueue(cudaStream_t stream, PendingCommand command, std::string& error);
    bool synchronize(cudaStream_t stream, std::string& error);
    bool synchronizeAll(std::string& error);
    StreamState* get(cudaStream_t stream, std::string& error);

private:
    bool synchronizeState(StreamState& state, std::string& error);
    bool synchronizeDefault(std::string& error);
    bool synchronizeExplicit(std::string& error);

    std::unordered_map<cudaStream_t, StreamState> streams_;
    std::uint64_t next_identifier_ = 1;
};

class ArtifactCatalog {
public:
    bool setDirectory(const char* path, std::string& error);
    bool ensure(id<MTLDevice> device, std::string& error);
    NSDictionary* route(const char* kernel_name) const;
    id<MTLLibrary> library() const { return library_; }

private:
    static std::string executableDirectory();

    std::string directory_;
    id<MTLLibrary> library_ = nil;
    NSDictionary* routing_ = nil;
};

class PipelineCache {
public:
    void reset();
    id<MTLComputePipelineState> get(
        id<MTLDevice> device,
        id<MTLLibrary> library,
        NSString* entry,
        std::string& error
    );

private:
    std::unordered_map<std::string, id<MTLComputePipelineState>> pipelines_;
};

bool resolveDispatchGeometry(
    NSDictionary* route,
    id<MTLComputePipelineState> pipeline,
    dim3 block,
    MTLSize& physical,
    std::string& error
);

class KernelDispatcher {
public:
    static cudaError_t dispatch(
        id<MTLDevice> device,
        id<MTLCommandQueue> queue,
        BufferRegistry& buffers,
        ArtifactCatalog& artifacts,
        PipelineCache& pipelines,
        const char* kernel_name,
        dim3 grid,
        dim3 block,
        const KernelArgument* arguments,
        std::size_t argument_count,
        std::size_t shared_bytes,
        id<MTLCommandBuffer> __strong& submitted,
        std::string& error
    );
};

class AsyncCopyDispatcher {
public:
    static cudaError_t submit(
        id<MTLDevice> device,
        id<MTLCommandQueue> queue,
        BufferRegistry& buffers,
        void* dst,
        const void* src,
        std::size_t count,
        cudaMemcpyKind kind,
        PendingCommand& pending,
        std::string& error
    );
};

}  // namespace mcu::runtime
