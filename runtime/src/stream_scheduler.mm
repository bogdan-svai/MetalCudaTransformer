#include "runtime_internal.h"

#include <chrono>
#include <cstring>

namespace mcu::runtime {

StreamRegistry::~StreamRegistry() {
    for (auto& item : streams_) {
        for (PendingCommand& pending : item.second.pending) {
            [pending.command waitUntilCompleted];
            if (pending.host_destination && pending.copy_size &&
                [pending.command status] == MTLCommandBufferStatusCompleted) {
                std::memcpy(pending.host_destination, [pending.staging contents], pending.copy_size);
            }
        }
        if (item.first) {
            delete item.first;
        }
    }
}

bool StreamRegistry::ensureDefault(id<MTLCommandQueue> queue, std::string& error) {
    if (!queue) {
        error = "Default Metal command queue is unavailable";
        return false;
    }
    auto found = streams_.find(nullptr);
    if (found == streams_.end()) {
        streams_[nullptr] = {queue, {}};
    }
    return true;
}

bool StreamRegistry::create(id<MTLDevice> device, cudaStream_t* output, std::string& error) {
    if (!output) {
        error = "cudaStreamCreate requires a non-null output";
        return false;
    }
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (!queue) {
        error = "Metal command queue creation failed for CUDA stream";
        return false;
    }
    cudaStream_t handle = new mcuStream{next_identifier_++};
    streams_[handle] = {queue, {}};
    *output = handle;
    incrementRuntimeCounter("streams_created");
    return true;
}

bool StreamRegistry::destroy(cudaStream_t stream, std::string& error) {
    if (!stream) {
        error = "The default CUDA stream cannot be destroyed";
        return false;
    }
    auto found = streams_.find(stream);
    if (found == streams_.end()) {
        error = "cudaStreamDestroy received an invalid stream";
        return false;
    }
    if (!synchronizeState(found->second, error)) {
        return false;
    }
    streams_.erase(found);
    delete stream;
    incrementRuntimeCounter("streams_destroyed");
    return true;
}

StreamState* StreamRegistry::get(cudaStream_t stream, std::string& error) {
    auto found = streams_.find(stream);
    if (found == streams_.end()) {
        error = "CUDA operation received an invalid stream";
        return nullptr;
    }
    return &found->second;
}

bool StreamRegistry::prepare(cudaStream_t stream, std::string& error) {
    if (!get(stream, error)) {
        return false;
    }
    return stream ? synchronizeDefault(error) : synchronizeExplicit(error);
}

bool StreamRegistry::enqueue(cudaStream_t stream, PendingCommand command, std::string& error) {
    StreamState* state = get(stream, error);
    if (!state) {
        return false;
    }
    state->pending.push_back(std::move(command));
    incrementRuntimeCounter("command_buffers_submitted");
    return true;
}

bool StreamRegistry::synchronize(cudaStream_t stream, std::string& error) {
    StreamState* state = get(stream, error);
    return state && synchronizeState(*state, error);
}

bool StreamRegistry::synchronizeAll(std::string& error) {
    for (auto& item : streams_) {
        if (!synchronizeState(item.second, error)) {
            return false;
        }
    }
    return true;
}

bool StreamRegistry::synchronizeDefault(std::string& error) {
    auto found = streams_.find(nullptr);
    return found == streams_.end() || synchronizeState(found->second, error);
}

bool StreamRegistry::synchronizeExplicit(std::string& error) {
    for (auto& item : streams_) {
        if (item.first && !synchronizeState(item.second, error)) {
            return false;
        }
    }
    return true;
}

bool StreamRegistry::synchronizeState(StreamState& state, std::string& error) {
    if (state.pending.empty()) {
        return true;
    }
    auto wait_started = std::chrono::steady_clock::now();
    for (PendingCommand& pending : state.pending) {
        [pending.command waitUntilCompleted];
        if ([pending.command status] == MTLCommandBufferStatusError) {
            NSString* description = [[pending.command error] localizedDescription];
            error = pending.operation + ": "
                + (description ? [description UTF8String] : "Metal command failed");
            state.pending.clear();
            return false;
        }
        CFTimeInterval gpu_start = [pending.command GPUStartTime];
        CFTimeInterval gpu_end = [pending.command GPUEndTime];
        if (gpu_end >= gpu_start && gpu_start > 0.0) {
            recordRuntimeMetric("gpu_execution", gpu_end - gpu_start);
        }
        if (pending.host_destination && pending.copy_size) {
            std::memcpy(pending.host_destination, [pending.staging contents], pending.copy_size);
        }
    }
    recordRuntimeMetric(
        "synchronization_wait",
        std::chrono::duration<double>(std::chrono::steady_clock::now() - wait_started).count()
    );
    incrementRuntimeCounter("stream_synchronizations");
    state.pending.clear();
    return true;
}

cudaError_t AsyncCopyDispatcher::submit(
    id<MTLDevice> device,
    id<MTLCommandQueue> queue,
    BufferRegistry& buffers,
    void* dst,
    const void* src,
    std::size_t count,
    cudaMemcpyKind kind,
    PendingCommand& pending,
    std::string& error
) {
    if (count == 0) {
        return cudaSuccess;
    }
    cudaMemcpyKind resolved;
    if (!buffers.resolveCopyKind(dst, src, count, kind, resolved, error)) {
        return cudaErrorInvalidValue;
    }
    if (resolved == cudaMemcpyHostToHost) {
        error = "cudaMemcpyAsync does not support host-to-host copies";
        return cudaErrorInvalidValue;
    }
    id<MTLBuffer> destination = nil;
    id<MTLBuffer> source = nil;
    NSUInteger destination_offset = 0;
    NSUInteger source_offset = 0;
    id<MTLBuffer> staging = nil;
    if (resolved == cudaMemcpyHostToDevice) {
        buffers.find(dst, count, destination, destination_offset);
        staging = [device newBufferWithBytes:src length:count options:MTLResourceStorageModeShared];
        source = staging;
    } else if (resolved == cudaMemcpyDeviceToHost) {
        buffers.find(src, count, source, source_offset);
        staging = [device newBufferWithLength:count options:MTLResourceStorageModeShared];
        destination = staging;
    } else {
        buffers.find(dst, count, destination, destination_offset);
        buffers.find(src, count, source, source_offset);
    }
    if (!source || !destination) {
        error = "cudaMemcpyAsync could not resolve Metal buffers";
        return cudaErrorMemoryAllocation;
    }
    auto started = std::chrono::steady_clock::now();
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLBlitCommandEncoder> encoder = [command blitCommandEncoder];
    [encoder copyFromBuffer:source sourceOffset:source_offset
                   toBuffer:destination destinationOffset:destination_offset size:count];
    [encoder endEncoding];
    [command commit];
    recordRuntimeMetric(
        "command_submission",
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count()
    );
    incrementRuntimeCounter("async_copy_bytes", count);
    pending.command = command;
    pending.staging = staging;
    pending.host_destination = resolved == cudaMemcpyDeviceToHost ? dst : nullptr;
    pending.copy_size = resolved == cudaMemcpyDeviceToHost ? count : 0;
    pending.operation = "cudaMemcpyAsync";
    return cudaSuccess;
}

}  // namespace mcu::runtime
