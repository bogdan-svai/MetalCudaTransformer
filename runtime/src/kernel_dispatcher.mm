#include "runtime_internal.h"

namespace mcu::runtime {

cudaError_t KernelDispatcher::dispatch(
    DeviceContext& context,
    BufferRegistry& buffers,
    ArtifactCatalog& artifacts,
    PipelineCache& pipelines,
    const char* kernel_name,
    dim3 grid,
    dim3 block,
    const KernelArgument* arguments,
    std::size_t argument_count,
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
        context.device(), artifacts.library(), route[@"msl_entry_point"], error
    );
    if (!pipeline) {
        return cudaErrorInitializationError;
    }
    MTLSize physical;
    if (!resolveDispatchGeometry(route, pipeline, block, physical, error)) {
        return cudaErrorInvalidValue;
    }

    id<MTLCommandBuffer> command = [context.queue() commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    for (NSUInteger index = 0; index < [bindings count]; ++index) {
        NSDictionary* binding = bindings[index];
        NSUInteger slot = [binding[@"buffer_index"] unsignedIntegerValue];
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
            [encoder setBuffer:buffer offset:offset atIndex:slot];
        } else {
            if (argument.kind != ArgumentKind::Scalar || !argument.scalar_data) {
                error = "Expected a scalar kernel argument";
                return cudaErrorInvalidValue;
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
        error = description ? [description UTF8String] : "Metal command failed";
        return cudaErrorLaunchFailure;
    }
    return cudaSuccess;
}

}  // namespace mcu::runtime
