#include "runtime_internal.h"

#include <chrono>

namespace mcu::runtime {

cudaError_t KernelDispatcher::dispatch(
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
) {
    NSDictionary* route = artifacts.route(kernel_name);
    if (!route) {
        error = std::string("No routing entry for kernel '") + kernel_name + "'";
        return cudaErrorInvalidValue;
    }
    NSArray* bindings = route[@"bindings"];
    if ([bindings count] != argument_count) {
        error = "Kernel argument count does not match routing metadata";
        return cudaErrorInvalidValue;
    }
    id<MTLComputePipelineState> pipeline = pipelines.get(
        device, artifacts.library(), route[@"msl_entry_point"], error
    );
    if (!pipeline) {
        return cudaErrorInitializationError;
    }
    MTLSize physical;
    if (!resolveDispatchGeometry(route, pipeline, block, physical, error)) {
        return cudaErrorInvalidValue;
    }

    for (NSUInteger index = 0; index < [bindings count]; ++index) {
        NSDictionary* binding = bindings[index];
        const KernelArgument& argument = arguments[index];
        if ([binding[@"kind"] isEqualToString:@"buffer"]) {
            if (argument.kind != ArgumentKind::Buffer) {
                error = "Expected a device-buffer kernel argument";
                return cudaErrorInvalidValue;
            }
            id<MTLBuffer> buffer = nil;
            NSUInteger offset = 0;
            if (!buffers.find(argument.buffer_pointer, 1, buffer, offset)) {
                error = "Kernel argument is not a registered device pointer";
                return cudaErrorInvalidValue;
            }
        } else {
            if (argument.kind != ArgumentKind::Scalar || !argument.scalar_data) {
                error = "Expected a scalar kernel argument";
                return cudaErrorInvalidValue;
            }
        }
    }
    NSDictionary* threadgroup = route[@"threadgroup_memory"];
    if (threadgroup) {
        if ([threadgroup[@"schema_version"] unsignedIntegerValue] != 1 || shared_bytes == 0) {
            error = "Dynamic threadgroup memory requires a non-zero supported launch size";
            return cudaErrorInvalidValue;
        }
        NSUInteger alignment = [threadgroup[@"alignment"] unsignedIntegerValue];
        if ((alignment && shared_bytes % alignment != 0) ||
            shared_bytes > [device maxThreadgroupMemoryLength]) {
            error = "Dynamic threadgroup memory size is misaligned or exceeds the device limit";
            return cudaErrorInvalidValue;
        }
    } else if (shared_bytes != 0) {
        error = "Launch specifies dynamic shared memory for a kernel without a declaration";
        return cudaErrorInvalidValue;
    }

    auto submission_started = std::chrono::steady_clock::now();
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    for (NSUInteger index = 0; index < [bindings count]; ++index) {
        NSDictionary* binding = bindings[index];
        NSUInteger slot = [binding[@"buffer_index"] unsignedIntegerValue];
        const KernelArgument& argument = arguments[index];
        if ([binding[@"kind"] isEqualToString:@"buffer"]) {
            id<MTLBuffer> buffer = nil;
            NSUInteger offset = 0;
            buffers.find(argument.buffer_pointer, 1, buffer, offset);
            [encoder setBuffer:buffer offset:offset atIndex:slot];
        } else {
            [encoder setBytes:argument.scalar_data length:argument.scalar_size atIndex:slot];
        }
    }
    if (threadgroup) {
        NSUInteger slot = [threadgroup[@"slot"] unsignedIntegerValue];
        [encoder setThreadgroupMemoryLength:shared_bytes atIndex:slot];
    }
    [encoder dispatchThreadgroups:MTLSizeMake(grid.x, grid.y, grid.z)
             threadsPerThreadgroup:physical];
    [encoder endEncoding];
    [command commit];
    recordRuntimeMetric(
        "command_submission",
        std::chrono::duration<double>(std::chrono::steady_clock::now() - submission_started).count()
    );
    incrementRuntimeCounter("kernel_dispatches");
    submitted = command;
    return cudaSuccess;
}

}  // namespace mcu::runtime
