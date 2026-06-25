#include "runtime_internal.h"

#include <cstdlib>
#include <chrono>
#include <iostream>
#include <mutex>

namespace {

class RuntimeFacade {
public:
    cudaError_t setArtifactDirectory(const char* path) {
        std::lock_guard<std::mutex> lock(mutex_);
        std::string error;
        if (!streams_.synchronizeAll(error)) {
            return fail(cudaErrorLaunchFailure, error);
        }
        if (!artifacts_.setDirectory(path, error)) {
            return fail(cudaErrorInvalidValue, error);
        }
        pipelines_.reset();
        return succeed();
    }

    cudaError_t allocate(void** output, std::size_t size) {
        std::lock_guard<std::mutex> lock(mutex_);
        mcu::runtime::incrementRuntimeCounter("runtime_api_calls");
        if (!output || size == 0) {
            return fail(cudaErrorInvalidValue, "cudaMalloc requires non-null output and non-zero size");
        }
        std::string error;
        if (!ensureRuntime(error)) {
            return fail(cudaErrorInitializationError, error);
        }
        if (!buffers_.allocate(context_.device(), output, size, error)) {
            return fail(cudaErrorMemoryAllocation, error);
        }
        return succeed();
    }

    cudaError_t release(void* pointer) {
        std::lock_guard<std::mutex> lock(mutex_);
        mcu::runtime::incrementRuntimeCounter("runtime_api_calls");
        std::string error;
        if (!streams_.synchronizeAll(error)) {
            return fail(cudaErrorLaunchFailure, error);
        }
        return buffers_.release(pointer, error) ? succeed() : fail(cudaErrorInvalidValue, error);
    }

    cudaError_t copy(void* dst, const void* src, std::size_t count, cudaMemcpyKind kind) {
        std::lock_guard<std::mutex> lock(mutex_);
        mcu::runtime::incrementRuntimeCounter("runtime_api_calls");
        if ((!dst || !src) && count != 0) {
            return fail(cudaErrorInvalidValue, "cudaMemcpy received a null pointer");
        }
        std::string error;
        if (!streams_.synchronizeAll(error)) {
            return fail(cudaErrorLaunchFailure, error);
        }
        auto started = std::chrono::steady_clock::now();
        bool copied = buffers_.copy(dst, src, count, kind, error);
        mcu::runtime::recordRuntimeMetric(
            "memory_copy",
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count()
        );
        mcu::runtime::incrementRuntimeCounter("memory_copy_bytes", count);
        return copied ? succeed() : fail(cudaErrorInvalidValue, error);
    }

    cudaError_t copyAsync(
        void* dst,
        const void* src,
        std::size_t count,
        cudaMemcpyKind kind,
        cudaStream_t stream
    ) {
        std::lock_guard<std::mutex> lock(mutex_);
        mcu::runtime::incrementRuntimeCounter("runtime_api_calls");
        if ((!dst || !src) && count != 0) {
            return fail(cudaErrorInvalidValue, "cudaMemcpyAsync received a null pointer");
        }
        std::string error;
        if (!ensureRuntime(error)) {
            return fail(cudaErrorInitializationError, error);
        }
        mcu::runtime::StreamState* state = streams_.get(stream, error);
        if (!state) {
            return fail(cudaErrorInvalidValue, error);
        }
        if (!streams_.prepare(stream, error)) {
            return fail(cudaErrorLaunchFailure, error);
        }
        mcu::runtime::PendingCommand pending;
        cudaError_t result = mcu::runtime::AsyncCopyDispatcher::submit(
            context_.device(), state->queue, buffers_, dst, src, count, kind, pending, error
        );
        if (result != cudaSuccess) {
            return fail(result, error);
        }
        if (pending.command && !streams_.enqueue(stream, std::move(pending), error)) {
            return fail(cudaErrorInvalidValue, error);
        }
        return succeed();
    }

    cudaError_t fill(void* dst, int value, std::size_t count) {
        std::lock_guard<std::mutex> lock(mutex_);
        mcu::runtime::incrementRuntimeCounter("runtime_api_calls");
        if (!dst && count != 0) {
            return fail(cudaErrorInvalidValue, "cudaMemset received a null pointer");
        }
        std::string error;
        if (!streams_.synchronizeAll(error)) {
            return fail(cudaErrorLaunchFailure, error);
        }
        auto started = std::chrono::steady_clock::now();
        bool filled = buffers_.fill(dst, value, count, error);
        mcu::runtime::recordRuntimeMetric(
            "memory_fill",
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count()
        );
        mcu::runtime::incrementRuntimeCounter("memory_fill_bytes", count);
        return filled ? succeed() : fail(cudaErrorInvalidValue, error);
    }

    cudaError_t dispatchKernel(
        const char* kernel_name,
        dim3 grid,
        dim3 block,
        const mcu::KernelArgument* arguments,
        std::size_t argument_count,
        std::size_t shared_bytes,
        cudaStream_t stream
    ) {
        std::lock_guard<std::mutex> lock(mutex_);
        mcu::runtime::incrementRuntimeCounter("runtime_api_calls");
        if (!kernel_name || !*kernel_name || grid.x == 0 || grid.y == 0 || grid.z == 0 ||
            block.x == 0 || block.y == 0 || block.z == 0) {
            return fail(cudaErrorInvalidValue, "Kernel name and launch dimensions must be non-zero");
        }
        std::string error;
        if (!ensureRuntime(error) || !artifacts_.ensure(context_.device(), error)) {
            return fail(cudaErrorInitializationError, error);
        }
        mcu::runtime::StreamState* state = streams_.get(stream, error);
        if (!state) {
            return fail(cudaErrorInvalidValue, error);
        }
        if (!streams_.prepare(stream, error)) {
            return fail(cudaErrorLaunchFailure, error);
        }
        id<MTLCommandBuffer> submitted = nil;
        cudaError_t result = mcu::runtime::KernelDispatcher::dispatch(
            context_.device(), state->queue, buffers_, artifacts_, pipelines_, kernel_name,
            grid, block, arguments, argument_count, shared_bytes, submitted, error
        );
        if (result != cudaSuccess) {
            return fail(result, error);
        }
        mcu::runtime::PendingCommand pending;
        pending.command = submitted;
        pending.operation = std::string("kernel ") + kernel_name;
        return streams_.enqueue(stream, std::move(pending), error)
            ? succeed()
            : fail(cudaErrorInvalidValue, error);
    }

    cudaError_t createStream(cudaStream_t* stream) {
        std::lock_guard<std::mutex> lock(mutex_);
        mcu::runtime::incrementRuntimeCounter("runtime_api_calls");
        std::string error;
        if (!ensureRuntime(error)) {
            return fail(cudaErrorInitializationError, error);
        }
        return streams_.create(context_.device(), stream, error)
            ? succeed()
            : fail(cudaErrorInvalidValue, error);
    }

    cudaError_t destroyStream(cudaStream_t stream) {
        std::lock_guard<std::mutex> lock(mutex_);
        mcu::runtime::incrementRuntimeCounter("runtime_api_calls");
        std::string error;
        if (!stream || !streams_.get(stream, error)) {
            return fail(cudaErrorInvalidValue, error.empty() ? "The default CUDA stream cannot be destroyed" : error);
        }
        if (!streams_.synchronize(stream, error)) {
            return fail(cudaErrorLaunchFailure, error);
        }
        return streams_.destroy(stream, error)
            ? succeed()
            : fail(cudaErrorInvalidValue, error);
    }

    cudaError_t synchronizeStream(cudaStream_t stream) {
        std::lock_guard<std::mutex> lock(mutex_);
        mcu::runtime::incrementRuntimeCounter("runtime_api_calls");
        std::string error;
        if (!ensureRuntime(error)) {
            return fail(cudaErrorInitializationError, error);
        }
        if (!streams_.get(stream, error)) {
            return fail(cudaErrorInvalidValue, error);
        }
        return streams_.synchronize(stream, error)
            ? succeed()
            : fail(cudaErrorLaunchFailure, error);
    }

    cudaError_t synchronize() {
        std::lock_guard<std::mutex> lock(mutex_);
        mcu::runtime::incrementRuntimeCounter("runtime_api_calls");
        std::string error;
        return streams_.synchronizeAll(error)
            ? succeed()
            : fail(cudaErrorLaunchFailure, error);
    }
    cudaError_t takeLastError() {
        std::lock_guard<std::mutex> lock(mutex_);
        cudaError_t result = last_error_;
        last_error_ = cudaSuccess;
        last_message_.clear();
        return result;
    }

private:
    bool ensureRuntime(std::string& error) {
        return context_.ensure(error) && streams_.ensureDefault(context_.queue(), error);
    }

    cudaError_t fail(cudaError_t error, const std::string& message) {
        last_error_ = error;
        last_message_ = message;
        const char* diagnostics = std::getenv("MCU_RUNTIME_DIAGNOSTICS");
        if (diagnostics && std::string(diagnostics) == "1") {
            std::cerr << "MCU runtime: " << message << '\n';
        }
        return error;
    }

    cudaError_t succeed() {
        last_error_ = cudaSuccess;
        last_message_.clear();
        return cudaSuccess;
    }

    std::mutex mutex_;
    mcu::runtime::DeviceContext context_;
    mcu::runtime::BufferRegistry buffers_;
    mcu::runtime::ArtifactCatalog artifacts_;
    mcu::runtime::PipelineCache pipelines_;
    mcu::runtime::StreamRegistry streams_;
    cudaError_t last_error_ = cudaSuccess;
    std::string last_message_;
};

RuntimeFacade& runtimeFacade() {
    static RuntimeFacade instance;
    return instance;
}

}  // namespace

cudaError_t cudaMalloc(void** device_ptr, std::size_t size) {
    @autoreleasepool { return runtimeFacade().allocate(device_ptr, size); }
}

cudaError_t cudaFree(void* device_ptr) {
    @autoreleasepool { return runtimeFacade().release(device_ptr); }
}

cudaError_t cudaMemcpy(void* dst, const void* src, std::size_t count, cudaMemcpyKind kind) {
    @autoreleasepool { return runtimeFacade().copy(dst, src, count, kind); }
}

cudaError_t cudaMemcpyAsync(
    void* dst,
    const void* src,
    std::size_t count,
    cudaMemcpyKind kind,
    cudaStream_t stream
) {
    @autoreleasepool { return runtimeFacade().copyAsync(dst, src, count, kind, stream); }
}

cudaError_t cudaMemset(void* dst, int value, std::size_t count) {
    @autoreleasepool { return runtimeFacade().fill(dst, value, count); }
}

cudaError_t cudaDeviceSynchronize() { return runtimeFacade().synchronize(); }
cudaError_t cudaStreamCreate(cudaStream_t* stream) {
    @autoreleasepool { return runtimeFacade().createStream(stream); }
}
cudaError_t cudaStreamDestroy(cudaStream_t stream) {
    @autoreleasepool { return runtimeFacade().destroyStream(stream); }
}
cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    @autoreleasepool { return runtimeFacade().synchronizeStream(stream); }
}
cudaError_t cudaGetLastError() { return runtimeFacade().takeLastError(); }

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
    @autoreleasepool { return runtimeFacade().setArtifactDirectory(path); }
}

cudaError_t mcu::dispatch(
    const char* kernel_name,
    dim3 grid,
    dim3 block,
    const KernelArgument* arguments,
    std::size_t argument_count,
    std::size_t shared_bytes,
    cudaStream_t stream
) {
    @autoreleasepool {
        return runtimeFacade().dispatchKernel(
            kernel_name, grid, block, arguments, argument_count, shared_bytes, stream
        );
    }
}
