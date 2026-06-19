#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mach-o/dyld.h>

#include "cuda_to_metal.h"

#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct Allocation {
    id<MTLBuffer> buffer;
    std::size_t size;
};

class Runtime {
public:
    cudaError_t setArtifactDirectory(const char* path) {
        if (!path || !*path) {
            return fail(cudaErrorInvalidValue, "Artifact directory is empty");
        }
        std::lock_guard<std::mutex> lock(mutex_);
        artifact_directory_ = path;
        library_ = nil;
        routing_ = nil;
        pipelines_.clear();
        return succeed();
    }

    cudaError_t allocate(void** output, std::size_t size) {
        if (!output || size == 0) {
            return fail(cudaErrorInvalidValue, "cudaMalloc requires a non-null output and non-zero size");
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (!ensureDevice()) {
            return last_error_;
        }
        id<MTLBuffer> buffer = [device_ newBufferWithLength:size options:MTLResourceStorageModeShared];
        if (!buffer || ![buffer contents]) {
            return fail(cudaErrorMemoryAllocation, "Metal shared buffer allocation failed");
        }
        void* pointer = [buffer contents];
        allocations_[pointer] = {buffer, size};
        *output = pointer;
        return succeed();
    }

    cudaError_t release(void* pointer) {
        if (!pointer) {
            return succeed();
        }
        std::lock_guard<std::mutex> lock(mutex_);
        auto found = allocations_.find(pointer);
        if (found == allocations_.end()) {
            return fail(cudaErrorInvalidValue, "cudaFree received an unknown device pointer");
        }
        allocations_.erase(found);
        return succeed();
    }

    cudaError_t copy(void* dst, const void* src, std::size_t count, cudaMemcpyKind kind) {
        if ((!dst || !src) && count != 0) {
            return fail(cudaErrorInvalidValue, "cudaMemcpy received a null pointer");
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (!validateCopyRange(dst, count, kind, true) || !validateCopyRange(src, count, kind, false)) {
            return last_error_;
        }
        std::memmove(dst, src, count);
        return succeed();
    }

    cudaError_t dispatchKernel(
        const char* kernel_name,
        dim3 grid,
        dim3 block,
        const mcu::KernelArgument* arguments,
        std::size_t argument_count
    ) {
        if (!kernel_name || !*kernel_name || grid.x == 0 || grid.y == 0 || grid.z == 0 ||
            block.x == 0 || block.y == 0 || block.z == 0) {
            return fail(cudaErrorInvalidValue, "Kernel name and launch dimensions must be non-zero");
        }

        std::lock_guard<std::mutex> lock(mutex_);
        if (!ensureArtifacts()) {
            return last_error_;
        }

        NSDictionary* kernels = routing_[@"translated_kernels"];
        NSDictionary* route = kernels[[NSString stringWithUTF8String:kernel_name]];
        if (!route) {
            return fail(cudaErrorInvalidValue, std::string("No routing entry for kernel '") + kernel_name + "'");
        }
        NSArray* bindings = route[@"bindings"];
        if ([bindings count] != argument_count) {
            return fail(cudaErrorInvalidValue, "Kernel argument count does not match routing metadata");
        }

        NSString* entry = route[@"msl_entry_point"];
        id<MTLComputePipelineState> pipeline = pipelineForEntry(entry);
        if (!pipeline) {
            return last_error_;
        }

        MTLSize physical = MTLSizeMake(block.x, block.y, block.z);
        NSDictionary* metrics = route[@"optimization_metrics"];
        NSUInteger factor = [metrics[@"coarsening_factor"] unsignedIntegerValue];
        factor = factor == 0 ? 1 : factor;
        if (factor > 1) {
            NSArray* original = metrics[@"original_block"];
            if ([original count] == 3 &&
                ([original[0] unsignedIntValue] != block.x ||
                 [original[1] unsignedIntValue] != block.y ||
                 [original[2] unsignedIntValue] != block.z)) {
                return fail(cudaErrorInvalidValue, "Runtime block does not match the coarsening plan");
            }
            if (block.x % factor != 0) {
                return fail(cudaErrorInvalidValue, "Runtime block.x is not divisible by coarsening factor");
            }
            physical.width = block.x / factor;
        }
        NSArray* recordedPhysical = metrics[@"physical_block"];
        if ([recordedPhysical count] == 3 &&
            ([recordedPhysical[0] unsignedIntegerValue] != physical.width ||
             [recordedPhysical[1] unsignedIntegerValue] != physical.height ||
             [recordedPhysical[2] unsignedIntegerValue] != physical.depth)) {
            return fail(cudaErrorInvalidValue, "Physical block does not match routing metadata");
        }
        NSUInteger volume = physical.width * physical.height * physical.depth;
        if (volume > [pipeline maxTotalThreadsPerThreadgroup]) {
            return fail(cudaErrorInvalidValue, "Physical block exceeds Metal pipeline threadgroup limit");
        }

        id<MTLCommandBuffer> command = [queue_ commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        for (NSUInteger index = 0; index < [bindings count]; ++index) {
            NSDictionary* binding = bindings[index];
            NSUInteger slot = [binding[@"buffer_index"] unsignedIntegerValue];
            NSString* kind = binding[@"kind"];
            const mcu::KernelArgument& argument = arguments[index];
            if ([kind isEqualToString:@"buffer"]) {
                if (argument.kind != mcu::ArgumentKind::Buffer) {
                    return fail(cudaErrorInvalidValue, "Expected a device-buffer kernel argument");
                }
                id<MTLBuffer> buffer = nil;
                NSUInteger offset = 0;
                if (!findAllocation(argument.buffer_pointer, 1, buffer, offset)) {
                    return fail(cudaErrorInvalidValue, "Kernel argument is not a registered device pointer");
                }
                [encoder setBuffer:buffer offset:offset atIndex:slot];
            } else {
                if (argument.kind != mcu::ArgumentKind::Scalar || !argument.scalar_data) {
                    return fail(cudaErrorInvalidValue, "Expected a scalar kernel argument");
                }
                [encoder setBytes:argument.scalar_data length:argument.scalar_size atIndex:slot];
            }
        }

        [encoder dispatchThreadgroups:MTLSizeMake(grid.x, grid.y, grid.z)
                 threadsPerThreadgroup:physical];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if ([command status] == MTLCommandBufferStatusError) {
            NSString* description = [[command error] localizedDescription];
            return fail(cudaErrorLaunchFailure, description ? [description UTF8String] : "Metal command failed");
        }
        return succeed();
    }

    cudaError_t synchronize() {
        return last_error_;
    }

    cudaError_t lastError() const {
        return last_error_;
    }

private:
    bool ensureDevice() {
        if (device_) {
            return true;
        }
        device_ = MTLCreateSystemDefaultDevice();
        if (!device_) {
            fail(cudaErrorInitializationError, "No Metal device is available");
            return false;
        }
        queue_ = [device_ newCommandQueue];
        if (!queue_) {
            fail(cudaErrorInitializationError, "Metal command queue creation failed");
            return false;
        }
        return true;
    }

    bool ensureArtifacts() {
        if (library_ && routing_) {
            return true;
        }
        if (!ensureDevice()) {
            return false;
        }
        if (artifact_directory_.empty()) {
            artifact_directory_ = executableDirectory();
        }
        NSString* directory = [NSString stringWithUTF8String:artifact_directory_.c_str()];
        NSString* metalPath = [directory stringByAppendingPathComponent:@"generated.metal"];
        NSString* routingPath = [directory stringByAppendingPathComponent:@"routing_table.txt"];
        NSError* error = nil;
        NSString* source = [NSString stringWithContentsOfFile:metalPath encoding:NSUTF8StringEncoding error:&error];
        if (!source) {
            fail(cudaErrorInitializationError, error ? [[error localizedDescription] UTF8String] : "Cannot load generated.metal");
            return false;
        }
        library_ = [device_ newLibraryWithSource:source options:nil error:&error];
        if (!library_) {
            fail(cudaErrorInitializationError, error ? [[error localizedDescription] UTF8String] : "MSL JIT compilation failed");
            return false;
        }
        NSData* data = [NSData dataWithContentsOfFile:routingPath options:0 error:&error];
        NSDictionary* root = data ? [NSJSONSerialization JSONObjectWithData:data options:0 error:&error] : nil;
        if (!root || [root[@"schema_version"] integerValue] != 1) {
            fail(cudaErrorInitializationError, error ? [[error localizedDescription] UTF8String] : "Unsupported routing schema");
            return false;
        }
        routing_ = root;
        return true;
    }

    id<MTLComputePipelineState> pipelineForEntry(NSString* entry) {
        std::string key = [entry UTF8String];
        auto cached = pipelines_.find(key);
        if (cached != pipelines_.end()) {
            return cached->second;
        }
        id<MTLFunction> function = [library_ newFunctionWithName:entry];
        if (!function) {
            fail(cudaErrorInitializationError, "Metal entry point was not found");
            return nil;
        }
        NSError* error = nil;
        id<MTLComputePipelineState> pipeline = [device_ newComputePipelineStateWithFunction:function error:&error];
        if (!pipeline) {
            fail(cudaErrorInitializationError, error ? [[error localizedDescription] UTF8String] : "Pipeline creation failed");
            return nil;
        }
        pipelines_[key] = pipeline;
        return pipeline;
    }

    bool validateCopyRange(const void* pointer, std::size_t count, cudaMemcpyKind kind, bool destination) {
        bool requiresDevice = destination
            ? kind == cudaMemcpyHostToDevice || kind == cudaMemcpyDeviceToDevice
            : kind == cudaMemcpyDeviceToHost || kind == cudaMemcpyDeviceToDevice;
        if (!requiresDevice || kind == cudaMemcpyDefault) {
            return true;
        }
        id<MTLBuffer> buffer = nil;
        NSUInteger offset = 0;
        if (!findAllocation(pointer, count, buffer, offset)) {
            fail(cudaErrorInvalidValue, "cudaMemcpy device range is outside registered allocation");
            return false;
        }
        return true;
    }

    bool findAllocation(
        const void* pointer,
        std::size_t count,
        id<MTLBuffer> __strong& buffer,
        NSUInteger& offset
    ) {
        auto address = reinterpret_cast<std::uintptr_t>(pointer);
        for (const auto& item : allocations_) {
            auto base = reinterpret_cast<std::uintptr_t>(item.first);
            if (address >= base && address + count <= base + item.second.size) {
                buffer = item.second.buffer;
                offset = address - base;
                return true;
            }
        }
        return false;
    }

    cudaError_t fail(cudaError_t error, const std::string& message) {
        last_error_ = error;
        last_message_ = message;
        return error;
    }

    cudaError_t succeed() {
        last_error_ = cudaSuccess;
        last_message_.clear();
        return cudaSuccess;
    }

    static std::string executableDirectory() {
        uint32_t size = 0;
        _NSGetExecutablePath(nullptr, &size);
        std::vector<char> path(size);
        if (_NSGetExecutablePath(path.data(), &size) != 0) {
            return ".";
        }
        NSString* executable = [NSString stringWithUTF8String:path.data()];
        return [[[executable stringByStandardizingPath] stringByDeletingLastPathComponent] UTF8String];
    }

    std::mutex mutex_;
    id<MTLDevice> device_ = nil;
    id<MTLCommandQueue> queue_ = nil;
    id<MTLLibrary> library_ = nil;
    NSDictionary* routing_ = nil;
    std::unordered_map<void*, Allocation> allocations_;
    std::unordered_map<std::string, id<MTLComputePipelineState>> pipelines_;
    std::string artifact_directory_;
    cudaError_t last_error_ = cudaSuccess;
    std::string last_message_;
};

Runtime& runtime() {
    static Runtime instance;
    return instance;
}

}  // namespace

cudaError_t cudaMalloc(void** device_ptr, std::size_t size) {
    @autoreleasepool { return runtime().allocate(device_ptr, size); }
}

cudaError_t cudaFree(void* device_ptr) {
    @autoreleasepool { return runtime().release(device_ptr); }
}

cudaError_t cudaMemcpy(void* dst, const void* src, std::size_t count, cudaMemcpyKind kind) {
    @autoreleasepool { return runtime().copy(dst, src, count, kind); }
}

cudaError_t cudaDeviceSynchronize() {
    return runtime().synchronize();
}

cudaError_t cudaGetLastError() {
    return runtime().lastError();
}

const char* cudaGetErrorString(cudaError_t error) {
    switch (error) {
        case cudaSuccess: return "success";
        case cudaErrorInvalidValue: return "invalid value";
        case cudaErrorMemoryAllocation: return "memory allocation failed";
        case cudaErrorInitializationError: return "runtime initialization failed";
        case cudaErrorLaunchFailure: return "kernel launch failed";
    }
    return "unknown MCU runtime error";
}

cudaError_t mcuSetArtifactDirectory(const char* path) {
    @autoreleasepool { return runtime().setArtifactDirectory(path); }
}

cudaError_t mcu::dispatch(
    const char* kernel_name,
    dim3 grid,
    dim3 block,
    const KernelArgument* arguments,
    std::size_t argument_count
) {
    @autoreleasepool {
        return runtime().dispatchKernel(kernel_name, grid, block, arguments, argument_count);
    }
}
