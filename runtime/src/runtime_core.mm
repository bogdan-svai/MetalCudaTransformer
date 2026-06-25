#include "runtime_internal.h"

#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <cstdlib>
#include <chrono>
#include <iostream>
#include <mutex>
#include <algorithm>
#include <new>

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

    cublasStatus_t synchronizeLibraryStream(cudaStream_t stream) {
        std::lock_guard<std::mutex> lock(mutex_);
        std::string error;
        if (!ensureRuntime(error)) {
            fail(cudaErrorInitializationError, error);
            return CUBLAS_STATUS_NOT_INITIALIZED;
        }
        if (!streams_.get(stream, error)) {
            fail(cudaErrorInvalidValue, error);
            return CUBLAS_STATUS_INVALID_VALUE;
        }
        if (!streams_.synchronize(stream, error)) {
            fail(cudaErrorLaunchFailure, error);
            return CUBLAS_STATUS_EXECUTION_FAILED;
        }
        return CUBLAS_STATUS_SUCCESS;
    }

    cublasStatus_t blasSgemm(
        cublasHandle_t handle,
        cublasOperation_t transa,
        cublasOperation_t transb,
        int m,
        int n,
        int k,
        const float* alpha,
        const float* a,
        int lda,
        const float* b,
        int ldb,
        const float* beta,
        float* c,
        int ldc
    ) {
        std::lock_guard<std::mutex> lock(mutex_);
        mcu::runtime::incrementRuntimeCounter("runtime_api_calls");
        mcu::runtime::incrementRuntimeCounter("cublas_calls");
        auto started = std::chrono::steady_clock::now();
        std::string error;
        if (!validCublasHandle(handle)) {
            return failCublas(CUBLAS_STATUS_NOT_INITIALIZED, cudaErrorInvalidValue, "cublasSgemm received an invalid handle");
        }
        if (handle->pointer_mode != CUBLAS_POINTER_MODE_HOST) {
            return failCublas(CUBLAS_STATUS_NOT_SUPPORTED, cudaErrorNotSupported, "cublasSgemm supports host pointer mode only");
        }
        if (!alpha || !beta || !a || !b || !c) {
            return failCublas(CUBLAS_STATUS_INVALID_VALUE, cudaErrorInvalidValue, "cublasSgemm received a null pointer");
        }
        if (!_validOperation(transa) || !_validOperation(transb) || transa == CUBLAS_OP_C || transb == CUBLAS_OP_C) {
            return failCublas(CUBLAS_STATUS_NOT_SUPPORTED, cudaErrorNotSupported, "cublasSgemm supports CUBLAS_OP_N and CUBLAS_OP_T only");
        }
        if (m < 0 || n < 0 || k < 0 ||
            lda < std::max(1, transa == CUBLAS_OP_N ? m : k) ||
            ldb < std::max(1, transb == CUBLAS_OP_N ? k : n) ||
            ldc < std::max(1, m)) {
            return failCublas(CUBLAS_STATUS_INVALID_VALUE, cudaErrorInvalidValue, "cublasSgemm received invalid dimensions or leading dimensions");
        }
        if (!ensureRuntime(error)) {
            return failCublas(CUBLAS_STATUS_NOT_INITIALIZED, cudaErrorInitializationError, error);
        }
        if (!streams_.get(handle->stream, error)) {
            return failCublas(CUBLAS_STATUS_INVALID_VALUE, cudaErrorInvalidValue, error);
        }
        if (!_validateSgemmRange(a, _sgemmElements(transa, m, k, lda), "A", error) ||
            !_validateSgemmRange(b, _sgemmElements(transb, k, n, ldb), "B", error) ||
            !_validateSgemmRange(c, _matrixElements(m, n, ldc), "C", error)) {
            return failCublas(CUBLAS_STATUS_INVALID_VALUE, cudaErrorInvalidValue, error);
        }
        if (m == 0 || n == 0) {
            return succeedCublas();
        }
        if (!MPSSupportsMTLDevice(context_.device())) {
            if (!streams_.synchronize(handle->stream, error)) {
                return failCublas(CUBLAS_STATUS_EXECUTION_FAILED, cudaErrorLaunchFailure, error);
            }
            _executeSgemmOnCpu(transa, transb, m, n, k, *alpha, a, lda, b, ldb, *beta, c, ldc);
            recordCublasSgemmMetrics(started, m, n, "cpu");
            return succeedCublas();
        }
        mcu::runtime::StreamState* state = streams_.get(handle->stream, error);
        if (!state) {
            return failCublas(CUBLAS_STATUS_INVALID_VALUE, cudaErrorInvalidValue, error);
        }
        if (!streams_.prepare(handle->stream, error)) {
            return failCublas(CUBLAS_STATUS_EXECUTION_FAILED, cudaErrorLaunchFailure, error);
        }
        id<MTLCommandBuffer> command = [state->queue commandBuffer];
        if (!command) {
            return failCublas(CUBLAS_STATUS_EXECUTION_FAILED, cudaErrorLaunchFailure, "cublasSgemm could not create a Metal command buffer");
        }
        if (!_encodeSgemmWithMps(
                command, transa, transb, m, n, k, *alpha, a, lda, b, ldb, *beta, c, ldc, error
            )) {
            return failCublas(CUBLAS_STATUS_EXECUTION_FAILED, cudaErrorLaunchFailure, error);
        }
        auto submit_started = std::chrono::steady_clock::now();
        [command commit];
        mcu::runtime::recordRuntimeMetric(
            "cublas_queueing",
            std::chrono::duration<double>(std::chrono::steady_clock::now() - submit_started).count()
        );
        mcu::runtime::PendingCommand pending;
        pending.command = command;
        pending.operation = "cublasSgemm";
        if (!streams_.enqueue(handle->stream, std::move(pending), error)) {
            return failCublas(CUBLAS_STATUS_INVALID_VALUE, cudaErrorInvalidValue, error);
        }
        recordCublasSgemmMetrics(started, m, n, "mps");
        return succeedCublas();
    }

    void _executeSgemmOnCpu(
        cublasOperation_t transa,
        cublasOperation_t transb,
        int m,
        int n,
        int k,
        float alpha,
        const float* a,
        int lda,
        const float* b,
        int ldb,
        float beta,
        float* c,
        int ldc
    ) {
        for (int column = 0; column < n; ++column) {
            for (int row = 0; row < m; ++row) {
                float sum = 0.0f;
                for (int inner = 0; inner < k; ++inner) {
                    float av = transa == CUBLAS_OP_N
                        ? a[row + inner * lda]
                        : a[inner + row * lda];
                    float bv = transb == CUBLAS_OP_N
                        ? b[inner + column * ldb]
                        : b[column + inner * ldb];
                    sum += av * bv;
                }
                c[row + column * ldc] = alpha * sum + beta * c[row + column * ldc];
            }
        }
    }

    bool _encodeSgemmWithMps(
        id<MTLCommandBuffer> command,
        cublasOperation_t transa,
        cublasOperation_t transb,
        int m,
        int n,
        int k,
        float alpha,
        const float* a,
        int lda,
        const float* b,
        int ldb,
        float beta,
        float* c,
        int ldc,
        std::string& error
    ) const {
        id<MTLBuffer> buffer_a = nil;
        id<MTLBuffer> buffer_b = nil;
        id<MTLBuffer> buffer_c = nil;
        NSUInteger offset_a = 0;
        NSUInteger offset_b = 0;
        NSUInteger offset_c = 0;
        if (!buffers_.find(a, _sgemmElements(transa, m, k, lda) * sizeof(float), buffer_a, offset_a) ||
            !buffers_.find(b, _sgemmElements(transb, k, n, ldb) * sizeof(float), buffer_b, offset_b) ||
            !buffers_.find(c, _matrixElements(m, n, ldc) * sizeof(float), buffer_c, offset_c)) {
            error = "cublasSgemm could not resolve Metal buffers for MPS";
            return false;
        }

        const NSUInteger a_storage_rows = transa == CUBLAS_OP_N ? m : k;
        const NSUInteger a_storage_columns = transa == CUBLAS_OP_N ? k : m;
        const NSUInteger b_storage_rows = transb == CUBLAS_OP_N ? k : n;
        const NSUInteger b_storage_columns = transb == CUBLAS_OP_N ? n : k;
        MPSMatrixDescriptor* left_descriptor = [MPSMatrixDescriptor
            matrixDescriptorWithRows:b_storage_columns
            columns:b_storage_rows
            rowBytes:static_cast<NSUInteger>(ldb * sizeof(float))
            dataType:MPSDataTypeFloat32
        ];
        MPSMatrixDescriptor* right_descriptor = [MPSMatrixDescriptor
            matrixDescriptorWithRows:a_storage_columns
            columns:a_storage_rows
            rowBytes:static_cast<NSUInteger>(lda * sizeof(float))
            dataType:MPSDataTypeFloat32
        ];
        MPSMatrixDescriptor* result_descriptor = [MPSMatrixDescriptor
            matrixDescriptorWithRows:static_cast<NSUInteger>(n)
            columns:static_cast<NSUInteger>(m)
            rowBytes:static_cast<NSUInteger>(ldc * sizeof(float))
            dataType:MPSDataTypeFloat32
        ];
        MPSMatrix* left = [[MPSMatrix alloc] initWithBuffer:buffer_b offset:offset_b descriptor:left_descriptor];
        MPSMatrix* right = [[MPSMatrix alloc] initWithBuffer:buffer_a offset:offset_a descriptor:right_descriptor];
        MPSMatrix* result = [[MPSMatrix alloc] initWithBuffer:buffer_c offset:offset_c descriptor:result_descriptor];
        if (!left || !right || !result) {
            error = "cublasSgemm could not create MPS matrices";
            return false;
        }
        auto setup_started = std::chrono::steady_clock::now();
        MPSMatrixMultiplication* multiplication = [[MPSMatrixMultiplication alloc]
            initWithDevice:context_.device()
            transposeLeft:(transb == CUBLAS_OP_T)
            transposeRight:(transa == CUBLAS_OP_T)
            resultRows:static_cast<NSUInteger>(n)
            resultColumns:static_cast<NSUInteger>(m)
            interiorColumns:static_cast<NSUInteger>(k)
            alpha:alpha
            beta:beta
        ];
        if (!multiplication) {
            error = "cublasSgemm could not create MPSMatrixMultiplication";
            return false;
        }
        mcu::runtime::recordRuntimeMetric(
            "cublas_adapter_setup",
            std::chrono::duration<double>(std::chrono::steady_clock::now() - setup_started).count()
        );
        multiplication.options = MPSKernelOptionsSkipAPIValidation;
        [multiplication encodeToCommandBuffer:command leftMatrix:left rightMatrix:right resultMatrix:result];
        return true;
    }

    void recordCublasSgemmMetrics(
        std::chrono::steady_clock::time_point started,
        int m,
        int n,
        const char* backend
    ) {
        mcu::runtime::recordRuntimeMetric(
            "cublas_sgemm",
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count()
        );
        mcu::runtime::incrementRuntimeCounter(
            "cublas_sgemm_elements",
            static_cast<std::uint64_t>(m) * static_cast<std::uint64_t>(n)
        );
        mcu::runtime::incrementRuntimeCounter(
            std::string("cublas_sgemm_backend_").append(backend).c_str()
        );
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

    static bool validCublasHandle(cublasHandle_t handle) {
        return handle && handle->alive;
    }

    static bool _validOperation(cublasOperation_t op) {
        return op == CUBLAS_OP_N || op == CUBLAS_OP_T || op == CUBLAS_OP_C;
    }

    static std::size_t _matrixElements(int rows, int columns, int leading_dimension) {
        if (rows == 0 || columns == 0) {
            return 0;
        }
        return static_cast<std::size_t>(columns - 1) * static_cast<std::size_t>(leading_dimension) +
            static_cast<std::size_t>(rows);
    }

    static std::size_t _sgemmElements(cublasOperation_t op, int op_rows, int op_columns, int leading_dimension) {
        if (op == CUBLAS_OP_N) {
            return _matrixElements(op_rows, op_columns, leading_dimension);
        }
        return _matrixElements(op_columns, op_rows, leading_dimension);
    }

    bool _validateSgemmRange(const float* pointer, std::size_t elements, const char* name, std::string& error) const {
        if (elements == 0) {
            return true;
        }
        id<MTLBuffer> buffer = nil;
        NSUInteger offset = 0;
        if (!buffers_.find(pointer, elements * sizeof(float), buffer, offset)) {
            error = std::string("cublasSgemm device range for ") + name + " is outside registered allocation";
            return false;
        }
        return true;
    }

    cublasStatus_t failCublas(cublasStatus_t status, cudaError_t cuda_error, const std::string& message) {
        fail(cuda_error, message);
        return status;
    }

    cublasStatus_t succeedCublas() {
        succeed();
        return CUBLAS_STATUS_SUCCESS;
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
        case cudaErrorNotSupported: return "operation not supported";
    }
    return "unknown MCU runtime error";
}

cudaError_t mcuSetArtifactDirectory(const char* path) {
    @autoreleasepool { return runtimeFacade().setArtifactDirectory(path); }
}

cublasStatus_t cublasCreate(cublasHandle_t* handle) {
    if (!handle) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    @autoreleasepool {
        cublasStatus_t ready = runtimeFacade().synchronizeLibraryStream(nullptr);
        if (ready != CUBLAS_STATUS_SUCCESS) {
            return ready;
        }
    }
    auto context = new (std::nothrow) mcuCublasContext();
    if (!context) {
        return CUBLAS_STATUS_ALLOC_FAILED;
    }
    *handle = context;
    mcu::runtime::incrementRuntimeCounter("cublas_handles_created");
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDestroy(cublasHandle_t handle) {
    if (!handle || !handle->alive) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    @autoreleasepool {
        cublasStatus_t synchronized = runtimeFacade().synchronizeLibraryStream(handle->stream);
        if (synchronized != CUBLAS_STATUS_SUCCESS) {
            return synchronized;
        }
    }
    handle->alive = false;
    delete handle;
    mcu::runtime::incrementRuntimeCounter("cublas_handles_destroyed");
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetStream(cublasHandle_t handle, cudaStream_t stream) {
    if (!handle || !handle->alive) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    @autoreleasepool {
        cublasStatus_t ready = runtimeFacade().synchronizeLibraryStream(stream);
        if (ready != CUBLAS_STATUS_SUCCESS) {
            return ready;
        }
    }
    handle->stream = stream;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetStream(cublasHandle_t handle, cudaStream_t* stream) {
    if (!handle || !handle->alive) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    if (!stream) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    *stream = handle->stream;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetPointerMode(cublasHandle_t handle, cublasPointerMode_t mode) {
    if (!handle || !handle->alive) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    if (mode != CUBLAS_POINTER_MODE_HOST) {
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }
    handle->pointer_mode = mode;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetPointerMode(cublasHandle_t handle, cublasPointerMode_t* mode) {
    if (!handle || !handle->alive) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    if (!mode) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    *mode = handle->pointer_mode;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSgemm(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const float* alpha,
    const float* A,
    int lda,
    const float* B,
    int ldb,
    const float* beta,
    float* C,
    int ldc
) {
    @autoreleasepool {
        return runtimeFacade().blasSgemm(
            handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc
        );
    }
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
