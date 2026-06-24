#pragma once

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "cuda_to_metal.h"

#include <cstddef>
#include <string>
#include <unordered_map>

namespace mcu::runtime {

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
        DeviceContext& context,
        BufferRegistry& buffers,
        ArtifactCatalog& artifacts,
        PipelineCache& pipelines,
        const char* kernel_name,
        dim3 grid,
        dim3 block,
        const KernelArgument* arguments,
        std::size_t argument_count,
        std::size_t shared_bytes,
        std::string& error
    );
};

}  // namespace mcu::runtime
