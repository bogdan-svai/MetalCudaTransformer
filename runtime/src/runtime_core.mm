#include "runtime_internal.h"

#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <cstdlib>
#include <chrono>
#include <cstring>
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
        if (isCapturing(stream)) {
            cudaMemcpy3DParms parameters = {};
            parameters.srcPtr = make_cudaPitchedPtr(const_cast<void*>(src), count, count, 1);
            parameters.dstPtr = make_cudaPitchedPtr(dst, count, count, 1);
            parameters.extent = make_cudaExtent(count, 1, 1);
            parameters.kind = kind;
            cudaGraphNode_t node = nullptr;
            cudaGraphNode_t dependency = capture_tails_[stream];
            cudaError_t captured = addMemcpyNodeUnlocked(
                &node,
                active_captures_[stream],
                dependency ? &dependency : nullptr,
                dependency ? 1 : 0,
                &parameters
            );
            if (captured == cudaSuccess) {
                capture_tails_[stream] = node;
            }
            return captured;
        }
        return copyAsyncUnlocked(dst, src, count, kind, stream);
    }

    cudaError_t fillAsync(void* dst, int value, std::size_t count, cudaStream_t stream) {
        std::lock_guard<std::mutex> lock(mutex_);
        mcu::runtime::incrementRuntimeCounter("runtime_api_calls");
        if (!dst && count != 0) {
            return fail(cudaErrorInvalidValue, "cudaMemsetAsync received a null pointer");
        }
        std::string error;
        if (!ensureRuntime(error)) {
            return fail(cudaErrorInitializationError, error);
        }
        if (isCapturing(stream)) {
            cudaMemsetParams parameters = {};
            parameters.dst = dst;
            parameters.value = static_cast<unsigned int>(value);
            parameters.elementSize = 1;
            parameters.width = count;
            parameters.height = 1;
            cudaGraphNode_t node = nullptr;
            cudaGraphNode_t dependency = capture_tails_[stream];
            cudaError_t captured = addMemsetNodeUnlocked(
                &node,
                active_captures_[stream],
                dependency ? &dependency : nullptr,
                dependency ? 1 : 0,
                &parameters
            );
            if (captured == cudaSuccess) {
                capture_tails_[stream] = node;
            }
            return captured;
        }
        return fillUnlocked(dst, value, count);
    }

    cudaError_t copyAsyncUnlocked(
        void* dst,
        const void* src,
        std::size_t count,
        cudaMemcpyKind kind,
        cudaStream_t stream
    ) {
        std::string error;
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
        return fillUnlocked(dst, value, count);
    }

    cudaError_t fillUnlocked(void* dst, int value, std::size_t count) {
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
        if (isCapturing(stream)) {
            cudaGraphNode_t node = nullptr;
            cudaGraphNode_t dependency = capture_tails_[stream];
            cudaError_t captured = addKernelNodeUnlocked(
                &node,
                active_captures_[stream],
                dependency ? &dependency : nullptr,
                dependency ? 1 : 0,
                kernel_name,
                grid,
                block,
                arguments,
                argument_count,
                shared_bytes
            );
            if (captured == cudaSuccess) {
                capture_tails_[stream] = node;
            }
            return captured;
        }
        return dispatchKernelUnlocked(
            kernel_name, grid, block, arguments, argument_count, shared_bytes, stream
        );
    }

    cudaError_t createGraph(cudaGraph_t* graph, unsigned int flags) {
        std::lock_guard<std::mutex> lock(mutex_);
        mcu::runtime::incrementRuntimeCounter("runtime_api_calls");
        mcu::runtime::incrementRuntimeCounter("graph_api_calls");
        if (!graph) {
            return fail(cudaErrorInvalidValue, "cudaGraphCreate requires a non-null output");
        }
        if (flags != 0) {
            return fail(cudaErrorNotSupported, "cudaGraphCreate supports flags=0 only");
        }
        cudaGraph_t handle = new (std::nothrow) mcuGraph{next_graph_identifier_++, true, {}};
        if (!handle) {
            return fail(cudaErrorMemoryAllocation, "cudaGraph allocation failed");
        }
        *graph = handle;
        mcu::runtime::incrementRuntimeCounter("graphs_created");
        return succeed();
    }

    cudaError_t destroyGraph(cudaGraph_t graph) {
        std::lock_guard<std::mutex> lock(mutex_);
        mcu::runtime::incrementRuntimeCounter("runtime_api_calls");
        mcu::runtime::incrementRuntimeCounter("graph_api_calls");
        if (!validGraph(graph)) {
            return fail(cudaErrorInvalidResourceHandle, "cudaGraphDestroy received an invalid graph");
        }
        graph->alive = false;
        for (auto& node : graph->nodes) {
            node->alive = false;
        }
        delete graph;
        mcu::runtime::incrementRuntimeCounter("graphs_destroyed");
        return succeed();
    }

    cudaError_t addKernelNode(
        cudaGraphNode_t* output,
        cudaGraph_t graph,
        const cudaGraphNode_t* dependencies,
        std::size_t dependency_count,
        const cudaKernelNodeParams* parameters
    ) {
        std::lock_guard<std::mutex> lock(mutex_);
        mcu::runtime::incrementRuntimeCounter("runtime_api_calls");
        mcu::runtime::incrementRuntimeCounter("graph_api_calls");
        if (!output || !validGraph(graph) || !parameters) {
            return fail(cudaErrorInvalidValue, "cudaGraphAddKernelNode received null or invalid arguments");
        }
        if (!parameters->func || !parameters->kernelParams || parameters->extra) {
            return fail(cudaErrorInvalidValue, "cudaGraphAddKernelNode requires func and kernelParams; extra is not supported");
        }
        if (parameters->gridDim.x == 0 || parameters->gridDim.y == 0 || parameters->gridDim.z == 0 ||
            parameters->blockDim.x == 0 || parameters->blockDim.y == 0 || parameters->blockDim.z == 0) {
            return fail(cudaErrorInvalidValue, "cudaGraphAddKernelNode launch dimensions must be non-zero");
        }
        const char* kernel_name = reinterpret_cast<const char*>(parameters->func);
        std::string error;
        if (!ensureRuntime(error) || !artifacts_.ensure(context_.device(), error)) {
            return fail(cudaErrorInitializationError, error);
        }
        NSDictionary* route = artifacts_.route(kernel_name);
        if (!route) {
            return fail(cudaErrorInvalidValue, std::string("No routing entry for graph kernel '") + kernel_name + "'");
        }
        NSArray* bindings = route[@"bindings"];
        std::unique_ptr<mcuGraphNode> node(new (std::nothrow) mcuGraphNode{
            next_graph_node_identifier_++,
            true,
            mcuGraphNodeKind::Kernel,
            kernel_name,
            parameters->gridDim,
            parameters->blockDim,
            parameters->sharedMemBytes,
            {},
            {},
            nullptr,
            nullptr,
            0,
            cudaMemcpyDefault,
            nullptr,
            0,
            0,
            {},
        });
        if (!node) {
            return fail(cudaErrorMemoryAllocation, "cudaGraphAddKernelNode could not allocate a graph node");
        }
        node->scalar_storage.reserve([bindings count]);
        node->arguments.reserve([bindings count]);
        for (NSUInteger index = 0; index < [bindings count]; ++index) {
            if (!parameters->kernelParams[index]) {
                return fail(cudaErrorInvalidValue, "cudaGraphAddKernelNode received a null kernel parameter pointer");
            }
            NSDictionary* binding = bindings[index];
            if ([binding[@"kind"] isEqualToString:@"buffer"]) {
                void* pointer = *reinterpret_cast<void**>(parameters->kernelParams[index]);
                node->arguments.push_back({mcu::ArgumentKind::Buffer, nullptr, 0, pointer});
            } else {
                std::size_t size = scalarSize(binding[@"type"]);
                std::vector<std::byte> storage(size);
                std::memcpy(storage.data(), parameters->kernelParams[index], size);
                node->scalar_storage.push_back(std::move(storage));
                node->arguments.push_back({
                    mcu::ArgumentKind::Scalar,
                    node->scalar_storage.back().data(),
                    size,
                    nullptr,
                });
            }
        }
        if (!appendDependencies(graph, node.get(), dependencies, dependency_count, error)) {
            return fail(cudaErrorInvalidValue, error);
        }
        cudaGraphNode_t raw = node.get();
        graph->nodes.push_back(std::move(node));
        *output = raw;
        mcu::runtime::incrementRuntimeCounter("graph_kernel_nodes_created");
        return succeed();
    }

    cudaError_t addKernelNodeUnlocked(
        cudaGraphNode_t* output,
        cudaGraph_t graph,
        const cudaGraphNode_t* dependencies,
        std::size_t dependency_count,
        const char* kernel_name,
        dim3 grid,
        dim3 block,
        const mcu::KernelArgument* arguments,
        std::size_t argument_count,
        unsigned int shared_bytes
    ) {
        if (!output || !validGraph(graph) || !kernel_name || !*kernel_name || (!arguments && argument_count)) {
            return fail(cudaErrorInvalidValue, "Captured kernel node received invalid arguments");
        }
        std::string error;
        if (!ensureRuntime(error) || !artifacts_.ensure(context_.device(), error)) {
            return fail(cudaErrorInitializationError, error);
        }
        if (!artifacts_.route(kernel_name)) {
            return fail(cudaErrorInvalidValue, std::string("No routing entry for captured graph kernel '") + kernel_name + "'");
        }
        std::unique_ptr<mcuGraphNode> node(new (std::nothrow) mcuGraphNode{
            next_graph_node_identifier_++,
            true,
            mcuGraphNodeKind::Kernel,
            kernel_name,
            grid,
            block,
            shared_bytes,
            {},
            {},
            nullptr,
            nullptr,
            0,
            cudaMemcpyDefault,
            nullptr,
            0,
            0,
            {},
        });
        if (!node) {
            return fail(cudaErrorMemoryAllocation, "Captured kernel node allocation failed");
        }
        node->scalar_storage.reserve(argument_count);
        node->arguments.reserve(argument_count);
        for (std::size_t index = 0; index < argument_count; ++index) {
            const mcu::KernelArgument& argument = arguments[index];
            if (argument.kind == mcu::ArgumentKind::Buffer) {
                node->arguments.push_back(argument);
            } else {
                std::vector<std::byte> storage(argument.scalar_size);
                std::memcpy(storage.data(), argument.scalar_data, argument.scalar_size);
                node->scalar_storage.push_back(std::move(storage));
                node->arguments.push_back({
                    mcu::ArgumentKind::Scalar,
                    node->scalar_storage.back().data(),
                    argument.scalar_size,
                    nullptr,
                });
            }
        }
        if (!appendDependencies(graph, node.get(), dependencies, dependency_count, error)) {
            return fail(cudaErrorInvalidValue, error);
        }
        cudaGraphNode_t raw = node.get();
        graph->nodes.push_back(std::move(node));
        *output = raw;
        mcu::runtime::incrementRuntimeCounter("graph_kernel_nodes_created");
        return succeed();
    }

    cudaError_t addMemcpyNode(
        cudaGraphNode_t* output,
        cudaGraph_t graph,
        const cudaGraphNode_t* dependencies,
        std::size_t dependency_count,
        const cudaMemcpy3DParms* parameters
    ) {
        std::lock_guard<std::mutex> lock(mutex_);
        mcu::runtime::incrementRuntimeCounter("runtime_api_calls");
        mcu::runtime::incrementRuntimeCounter("graph_api_calls");
        return addMemcpyNodeUnlocked(output, graph, dependencies, dependency_count, parameters);
    }

    cudaError_t addMemcpyNodeUnlocked(
        cudaGraphNode_t* output,
        cudaGraph_t graph,
        const cudaGraphNode_t* dependencies,
        std::size_t dependency_count,
        const cudaMemcpy3DParms* parameters
    ) {
        if (!output || !validGraph(graph) || !parameters) {
            return fail(cudaErrorInvalidValue, "cudaGraphAddMemcpyNode received invalid arguments");
        }
        std::size_t count = parameters->extent.width;
        std::size_t height = parameters->extent.height == 0 ? 1 : parameters->extent.height;
        std::size_t depth = parameters->extent.depth == 0 ? 1 : parameters->extent.depth;
        if (height != 1 || depth != 1) {
            return fail(cudaErrorNotSupported, "cudaGraphAddMemcpyNode supports one-dimensional copies only");
        }
        void* dst = parameters->dstPtr.ptr;
        const void* src = parameters->srcPtr.ptr;
        if ((!dst || !src) && count != 0) {
            return fail(cudaErrorInvalidValue, "cudaGraphAddMemcpyNode received null copy pointers");
        }
        std::string error;
        cudaMemcpyKind resolved;
        if (!buffers_.resolveCopyKind(dst, src, count, parameters->kind, resolved, error)) {
            return fail(cudaErrorInvalidValue, error);
        }
        std::unique_ptr<mcuGraphNode> node(new (std::nothrow) mcuGraphNode{
            next_graph_node_identifier_++,
            true,
            mcuGraphNodeKind::Copy,
            "",
            dim3(1),
            dim3(1),
            0,
            {},
            {},
            dst,
            src,
            count,
            parameters->kind,
            nullptr,
            0,
            0,
            {},
        });
        if (!node) {
            return fail(cudaErrorMemoryAllocation, "cudaGraphAddMemcpyNode could not allocate a graph node");
        }
        if (!appendDependencies(graph, node.get(), dependencies, dependency_count, error)) {
            return fail(cudaErrorInvalidValue, error);
        }
        cudaGraphNode_t raw = node.get();
        graph->nodes.push_back(std::move(node));
        *output = raw;
        mcu::runtime::incrementRuntimeCounter("graph_memcpy_nodes_created");
        return succeed();
    }

    cudaError_t addMemsetNode(
        cudaGraphNode_t* output,
        cudaGraph_t graph,
        const cudaGraphNode_t* dependencies,
        std::size_t dependency_count,
        const cudaMemsetParams* parameters
    ) {
        std::lock_guard<std::mutex> lock(mutex_);
        mcu::runtime::incrementRuntimeCounter("runtime_api_calls");
        mcu::runtime::incrementRuntimeCounter("graph_api_calls");
        return addMemsetNodeUnlocked(output, graph, dependencies, dependency_count, parameters);
    }

    cudaError_t addMemsetNodeUnlocked(
        cudaGraphNode_t* output,
        cudaGraph_t graph,
        const cudaGraphNode_t* dependencies,
        std::size_t dependency_count,
        const cudaMemsetParams* parameters
    ) {
        if (!output || !validGraph(graph) || !parameters) {
            return fail(cudaErrorInvalidValue, "cudaGraphAddMemsetNode received invalid arguments");
        }
        std::size_t height = parameters->height == 0 ? 1 : parameters->height;
        unsigned int element_size = parameters->elementSize == 0 ? 1 : parameters->elementSize;
        if (height != 1 || element_size != 1) {
            return fail(cudaErrorNotSupported, "cudaGraphAddMemsetNode supports linear byte memset only");
        }
        if (!parameters->dst && parameters->width != 0) {
            return fail(cudaErrorInvalidValue, "cudaGraphAddMemsetNode received a null destination");
        }
        std::string error;
        if (!buffers_.fill(parameters->dst, static_cast<int>(parameters->value), 0, error)) {
            return fail(cudaErrorInvalidValue, error);
        }
        std::unique_ptr<mcuGraphNode> node(new (std::nothrow) mcuGraphNode{
            next_graph_node_identifier_++,
            true,
            mcuGraphNodeKind::Memset,
            "",
            dim3(1),
            dim3(1),
            0,
            {},
            {},
            nullptr,
            nullptr,
            0,
            cudaMemcpyDefault,
            parameters->dst,
            static_cast<int>(parameters->value),
            parameters->width,
            {},
        });
        if (!node) {
            return fail(cudaErrorMemoryAllocation, "cudaGraphAddMemsetNode could not allocate a graph node");
        }
        if (!appendDependencies(graph, node.get(), dependencies, dependency_count, error)) {
            return fail(cudaErrorInvalidValue, error);
        }
        cudaGraphNode_t raw = node.get();
        graph->nodes.push_back(std::move(node));
        *output = raw;
        mcu::runtime::incrementRuntimeCounter("graph_memset_nodes_created");
        return succeed();
    }

    cudaError_t addDependencies(
        cudaGraph_t graph,
        const cudaGraphNode_t* from,
        const cudaGraphNode_t* to,
        std::size_t dependency_count
    ) {
        std::lock_guard<std::mutex> lock(mutex_);
        mcu::runtime::incrementRuntimeCounter("runtime_api_calls");
        mcu::runtime::incrementRuntimeCounter("graph_api_calls");
        if (!validGraph(graph) || (!from && dependency_count) || (!to && dependency_count)) {
            return fail(cudaErrorInvalidValue, "cudaGraphAddDependencies received invalid arguments");
        }
        for (std::size_t index = 0; index < dependency_count; ++index) {
            if (!graphContains(graph, from[index]) || !graphContains(graph, to[index])) {
                return fail(cudaErrorInvalidValue, "cudaGraphAddDependencies requires nodes from the same graph");
            }
            to[index]->dependencies.push_back(from[index]);
        }
        std::vector<cudaGraphNode_t> ordered;
        std::string error;
        if (!topologicalOrder(graph, ordered, error)) {
            return fail(cudaErrorInvalidValue, error);
        }
        return succeed();
    }

    cudaError_t instantiateGraph(
        cudaGraphExec_t* graph_exec,
        cudaGraph_t graph,
        cudaGraphNode_t* error_node,
        char* log_buffer,
        std::size_t buffer_size
    ) {
        std::lock_guard<std::mutex> lock(mutex_);
        mcu::runtime::incrementRuntimeCounter("runtime_api_calls");
        mcu::runtime::incrementRuntimeCounter("graph_api_calls");
        if (!graph_exec || !validGraph(graph)) {
            return fail(cudaErrorInvalidValue, "cudaGraphInstantiate received invalid arguments");
        }
        std::vector<cudaGraphNode_t> ordered;
        std::string error;
        if (!topologicalOrder(graph, ordered, error)) {
            if (error_node) {
                *error_node = nullptr;
            }
            if (log_buffer && buffer_size) {
                std::strncpy(log_buffer, error.c_str(), buffer_size - 1);
                log_buffer[buffer_size - 1] = '\0';
            }
            return fail(cudaErrorInvalidValue, error);
        }
        cudaGraphExec_t exec = new (std::nothrow) mcuGraphExec{
            next_graph_exec_identifier_++,
            true,
            ordered,
        };
        if (!exec) {
            return fail(cudaErrorMemoryAllocation, "cudaGraphInstantiate could not allocate an executable graph");
        }
        *graph_exec = exec;
        mcu::runtime::incrementRuntimeCounter("graphs_instantiated");
        return succeed();
    }

    cudaError_t destroyGraphExec(cudaGraphExec_t graph_exec) {
        std::lock_guard<std::mutex> lock(mutex_);
        mcu::runtime::incrementRuntimeCounter("runtime_api_calls");
        mcu::runtime::incrementRuntimeCounter("graph_api_calls");
        if (!validGraphExec(graph_exec)) {
            return fail(cudaErrorInvalidResourceHandle, "cudaGraphExecDestroy received an invalid graph executable");
        }
        graph_exec->alive = false;
        delete graph_exec;
        mcu::runtime::incrementRuntimeCounter("graph_execs_destroyed");
        return succeed();
    }

    cudaError_t launchGraph(cudaGraphExec_t graph_exec, cudaStream_t stream) {
        std::lock_guard<std::mutex> lock(mutex_);
        mcu::runtime::incrementRuntimeCounter("runtime_api_calls");
        mcu::runtime::incrementRuntimeCounter("graph_api_calls");
        if (!validGraphExec(graph_exec)) {
            return fail(cudaErrorInvalidResourceHandle, "cudaGraphLaunch received an invalid graph executable");
        }
        auto started = std::chrono::steady_clock::now();
        for (cudaGraphNode_t node : graph_exec->ordered_nodes) {
            if (!node || !node->alive) {
                return fail(cudaErrorInvalidResourceHandle, "cudaGraphLaunch found an invalid graph node");
            }
            cudaError_t result = cudaSuccess;
            if (node->kind == mcuGraphNodeKind::Kernel) {
                result = dispatchKernelUnlocked(
                    node->kernel_name.c_str(),
                    node->grid,
                    node->block,
                    node->arguments.data(),
                    node->arguments.size(),
                    node->shared_bytes,
                    stream
                );
            } else if (node->kind == mcuGraphNodeKind::Copy) {
                result = copyAsyncUnlocked(
                    node->copy_dst,
                    node->copy_src,
                    node->copy_count,
                    node->copy_kind,
                    stream
                );
            } else if (node->kind == mcuGraphNodeKind::Memset) {
                result = fillUnlocked(node->memset_dst, node->memset_value, node->memset_count);
            }
            if (result != cudaSuccess) {
                return result;
            }
        }
        mcu::runtime::recordRuntimeMetric(
            "graph_launch",
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count()
        );
        mcu::runtime::incrementRuntimeCounter("graph_launches");
        return succeed();
    }

    cudaError_t getGraphNodes(cudaGraph_t graph, cudaGraphNode_t* nodes, std::size_t* count, bool roots_only) {
        std::lock_guard<std::mutex> lock(mutex_);
        mcu::runtime::incrementRuntimeCounter("runtime_api_calls");
        mcu::runtime::incrementRuntimeCounter("graph_api_calls");
        if (!validGraph(graph) || !count) {
            return fail(cudaErrorInvalidValue, "cudaGraphGetNodes requires a valid graph and count output");
        }
        std::vector<cudaGraphNode_t> selected;
        for (auto& node : graph->nodes) {
            if (!roots_only || node->dependencies.empty()) {
                selected.push_back(node.get());
            }
        }
        if (nodes) {
            std::size_t writable = std::min(*count, selected.size());
            for (std::size_t index = 0; index < writable; ++index) {
                nodes[index] = selected[index];
            }
        }
        *count = selected.size();
        return succeed();
    }

    cudaError_t beginCapture(cudaStream_t stream, unsigned int mode) {
        std::lock_guard<std::mutex> lock(mutex_);
        mcu::runtime::incrementRuntimeCounter("runtime_api_calls");
        mcu::runtime::incrementRuntimeCounter("graph_api_calls");
        std::string error;
        if (!ensureRuntime(error)) {
            return fail(cudaErrorInitializationError, error);
        }
        if (!streams_.get(stream, error)) {
            return fail(cudaErrorInvalidValue, error);
        }
        if (mode > cudaStreamCaptureModeRelaxed) {
            return fail(cudaErrorInvalidValue, "cudaStreamBeginCapture received an invalid capture mode");
        }
        if (isCapturing(stream)) {
            return fail(cudaErrorInvalidValue, "Nested stream capture is not supported");
        }
        cudaGraph_t graph = new (std::nothrow) mcuGraph{next_graph_identifier_++, true, {}};
        if (!graph) {
            return fail(cudaErrorMemoryAllocation, "cudaStreamBeginCapture could not allocate a graph");
        }
        active_captures_[stream] = graph;
        capture_tails_[stream] = nullptr;
        mcu::runtime::incrementRuntimeCounter("graphs_created");
        mcu::runtime::incrementRuntimeCounter("stream_captures_started");
        return succeed();
    }

    cudaError_t endCapture(cudaStream_t stream, cudaGraph_t* graph) {
        std::lock_guard<std::mutex> lock(mutex_);
        mcu::runtime::incrementRuntimeCounter("runtime_api_calls");
        mcu::runtime::incrementRuntimeCounter("graph_api_calls");
        if (!graph) {
            return fail(cudaErrorInvalidValue, "cudaStreamEndCapture requires a non-null graph output");
        }
        auto found = active_captures_.find(stream);
        if (found == active_captures_.end()) {
            return fail(cudaErrorInvalidValue, "cudaStreamEndCapture called for a stream without active capture");
        }
        *graph = found->second;
        active_captures_.erase(found);
        capture_tails_.erase(stream);
        mcu::runtime::incrementRuntimeCounter("stream_captures_finished");
        return succeed();
    }

    cudaError_t unsupportedGraphApi(const char* name) {
        std::lock_guard<std::mutex> lock(mutex_);
        mcu::runtime::incrementRuntimeCounter("runtime_api_calls");
        mcu::runtime::incrementRuntimeCounter("graph_api_calls");
        return fail(cudaErrorNotSupported, std::string(name) + " is planned for a later Stage 14 graph subset");
    }

    cudaError_t dispatchKernelUnlocked(
        const char* kernel_name,
        dim3 grid,
        dim3 block,
        const mcu::KernelArgument* arguments,
        std::size_t argument_count,
        std::size_t shared_bytes,
        cudaStream_t stream
    ) {
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

    static std::size_t scalarSize(NSString* type) {
        std::string text = type ? [type UTF8String] : "";
        if (text.find("double") != std::string::npos ||
            text.find("long long") != std::string::npos ||
            text.find("int64") != std::string::npos ||
            text.find("uint64") != std::string::npos) {
            return 8;
        }
        if (text.find("short") != std::string::npos) {
            return 2;
        }
        if (text.find("char") != std::string::npos) {
            return 1;
        }
        return 4;
    }

    static bool validGraph(cudaGraph_t graph) {
        return graph && graph->alive;
    }

    static bool validGraphExec(cudaGraphExec_t graph_exec) {
        return graph_exec && graph_exec->alive;
    }

    bool isCapturing(cudaStream_t stream) const {
        return active_captures_.find(stream) != active_captures_.end();
    }

    static bool graphContains(cudaGraph_t graph, cudaGraphNode_t node) {
        if (!validGraph(graph) || !node || !node->alive) {
            return false;
        }
        return std::any_of(graph->nodes.begin(), graph->nodes.end(), [node](const auto& owned) {
            return owned.get() == node;
        });
    }

    static bool appendDependencies(
        cudaGraph_t graph,
        cudaGraphNode_t node,
        const cudaGraphNode_t* dependencies,
        std::size_t dependency_count,
        std::string& error
    ) {
        if (!dependencies && dependency_count) {
            error = "Graph dependencies pointer is null";
            return false;
        }
        for (std::size_t index = 0; index < dependency_count; ++index) {
            if (!graphContains(graph, dependencies[index])) {
                error = "Graph dependency does not belong to the target graph";
                return false;
            }
            node->dependencies.push_back(dependencies[index]);
        }
        return true;
    }

    static bool topologicalOrder(cudaGraph_t graph, std::vector<cudaGraphNode_t>& ordered, std::string& error) {
        std::unordered_map<cudaGraphNode_t, int> state;
        for (auto& node : graph->nodes) {
            state[node.get()] = 0;
        }
        for (auto& node : graph->nodes) {
            if (!visitNode(node.get(), state, ordered, error)) {
                return false;
            }
        }
        return true;
    }

    static bool visitNode(
        cudaGraphNode_t node,
        std::unordered_map<cudaGraphNode_t, int>& state,
        std::vector<cudaGraphNode_t>& ordered,
        std::string& error
    ) {
        auto found = state.find(node);
        if (found == state.end()) {
            error = "Graph dependency references a node outside the graph";
            return false;
        }
        if (found->second == 2) {
            return true;
        }
        if (found->second == 1) {
            error = "Graph dependencies contain a cycle";
            return false;
        }
        found->second = 1;
        for (cudaGraphNode_t dependency : node->dependencies) {
            if (!visitNode(dependency, state, ordered, error)) {
                return false;
            }
        }
        found->second = 2;
        ordered.push_back(node);
        return true;
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

    cudaError_t getDeviceCount(int* count) {
        std::lock_guard<std::mutex> lock(mutex_);
        mcu::runtime::incrementRuntimeCounter("runtime_api_calls");
        mcu::runtime::incrementRuntimeCounter("device_api_calls");
        if (!count) {
            return fail(cudaErrorInvalidValue, "cudaGetDeviceCount requires a non-null output");
        }
        std::string error;
        if (!ensureRuntime(error)) {
            return fail(cudaErrorInitializationError, error);
        }
        *count = 1;
        return succeed();
    }

    cudaError_t getDevice(int* device) {
        std::lock_guard<std::mutex> lock(mutex_);
        mcu::runtime::incrementRuntimeCounter("runtime_api_calls");
        mcu::runtime::incrementRuntimeCounter("device_api_calls");
        if (!device) {
            return fail(cudaErrorInvalidValue, "cudaGetDevice requires a non-null output");
        }
        *device = current_device_;
        return succeed();
    }

    cudaError_t setDevice(int device) {
        std::lock_guard<std::mutex> lock(mutex_);
        mcu::runtime::incrementRuntimeCounter("runtime_api_calls");
        mcu::runtime::incrementRuntimeCounter("device_api_calls");
        if (device != 0) {
            return fail(cudaErrorInvalidDevice, "MCU exposes only logical CUDA device 0");
        }
        std::string error;
        if (!ensureRuntime(error)) {
            return fail(cudaErrorInitializationError, error);
        }
        current_device_ = 0;
        return succeed();
    }

    cudaError_t getDeviceProperties(cudaDeviceProp* properties, int device) {
        std::lock_guard<std::mutex> lock(mutex_);
        mcu::runtime::incrementRuntimeCounter("runtime_api_calls");
        mcu::runtime::incrementRuntimeCounter("device_api_calls");
        if (!properties) {
            return fail(cudaErrorInvalidValue, "cudaGetDeviceProperties requires a non-null output");
        }
        if (device != 0) {
            return fail(cudaErrorInvalidDevice, "MCU exposes only logical CUDA device 0");
        }
        std::string error;
        if (!ensureRuntime(error)) {
            return fail(cudaErrorInitializationError, error);
        }
        std::memset(properties, 0, sizeof(cudaDeviceProp));
        NSString* device_name = [context_.device() name];
        const char* name = device_name ? [device_name UTF8String] : "Apple Metal device";
        std::strncpy(properties->name, name, sizeof(properties->name) - 1);
        properties->totalGlobalMem = static_cast<std::size_t>([context_.device() recommendedMaxWorkingSetSize]);
        properties->sharedMemPerBlock = 48 * 1024;
        properties->regsPerBlock = 65536;
        properties->warpSize = 32;
        properties->maxThreadsPerBlock = 1024;
        properties->maxThreadsDim[0] = 1024;
        properties->maxThreadsDim[1] = 1024;
        properties->maxThreadsDim[2] = 64;
        properties->maxGridSize[0] = 2147483647;
        properties->maxGridSize[1] = 65535;
        properties->maxGridSize[2] = 65535;
        properties->major = 1;
        properties->minor = 0;
        properties->integrated = 1;
        properties->unifiedAddressing = 1;
        properties->managedMemory = 1;
        properties->concurrentKernels = 1;
        properties->asyncEngineCount = 1;
        properties->multiProcessorCount = 1;
        properties->maxThreadsPerMultiProcessor = 1024;
        return succeed();
    }

    cudaError_t resetDevice() {
        std::lock_guard<std::mutex> lock(mutex_);
        mcu::runtime::incrementRuntimeCounter("runtime_api_calls");
        mcu::runtime::incrementRuntimeCounter("device_api_calls");
        std::string error;
        if (!streams_.synchronizeAll(error)) {
            return fail(cudaErrorLaunchFailure, error);
        }
        last_error_ = cudaSuccess;
        last_message_.clear();
        return cudaSuccess;
    }

    cudaError_t createEvent(cudaEvent_t* event, unsigned int flags) {
        std::lock_guard<std::mutex> lock(mutex_);
        mcu::runtime::incrementRuntimeCounter("runtime_api_calls");
        if (!event) {
            return fail(cudaErrorInvalidValue, "cudaEventCreate requires a non-null output");
        }
        if (flags & (cudaEventDisableTiming | cudaEventInterprocess)) {
            return fail(cudaErrorNotSupported, "MCU does not support cudaEventDisableTiming or cudaEventInterprocess");
        }
        unsigned int allowed = cudaEventDefault | cudaEventBlockingSync;
        if (flags & ~allowed) {
            return fail(cudaErrorInvalidValue, "cudaEventCreateWithFlags received unsupported flags");
        }
        cudaEvent_t handle = new (std::nothrow) mcuEvent{
            next_event_identifier_++,
            flags,
            true,
            false,
            false,
            std::chrono::steady_clock::time_point{},
        };
        if (!handle) {
            return fail(cudaErrorMemoryAllocation, "cudaEvent allocation failed");
        }
        *event = handle;
        mcu::runtime::incrementRuntimeCounter("events_created");
        return succeed();
    }

    cudaError_t recordEvent(cudaEvent_t event, cudaStream_t stream) {
        std::lock_guard<std::mutex> lock(mutex_);
        mcu::runtime::incrementRuntimeCounter("runtime_api_calls");
        if (!validEvent(event)) {
            return fail(cudaErrorInvalidResourceHandle, "cudaEventRecord received an invalid event");
        }
        std::string error;
        if (!ensureRuntime(error)) {
            return fail(cudaErrorInitializationError, error);
        }
        if (!streams_.get(stream, error)) {
            return fail(cudaErrorInvalidValue, error);
        }
        if (!streams_.synchronize(stream, error)) {
            return fail(cudaErrorLaunchFailure, error);
        }
        event->timestamp = std::chrono::steady_clock::now();
        event->recorded = true;
        event->synchronized = true;
        mcu::runtime::incrementRuntimeCounter("events_recorded");
        return succeed();
    }

    cudaError_t synchronizeEvent(cudaEvent_t event) {
        std::lock_guard<std::mutex> lock(mutex_);
        mcu::runtime::incrementRuntimeCounter("runtime_api_calls");
        if (!validEvent(event)) {
            return fail(cudaErrorInvalidResourceHandle, "cudaEventSynchronize received an invalid event");
        }
        if (!event->recorded) {
            return fail(cudaErrorInvalidResourceHandle, "cudaEventSynchronize received an unrecorded event");
        }
        event->synchronized = true;
        mcu::runtime::incrementRuntimeCounter("event_synchronizations");
        return succeed();
    }

    cudaError_t elapsedEventTime(float* milliseconds, cudaEvent_t start, cudaEvent_t end) {
        std::lock_guard<std::mutex> lock(mutex_);
        mcu::runtime::incrementRuntimeCounter("runtime_api_calls");
        if (!milliseconds) {
            return fail(cudaErrorInvalidValue, "cudaEventElapsedTime requires a non-null output");
        }
        if (!validEvent(start) || !validEvent(end) || !start->recorded || !end->recorded) {
            return fail(cudaErrorInvalidResourceHandle, "cudaEventElapsedTime received invalid or unrecorded events");
        }
        auto duration = std::chrono::duration<double, std::milli>(end->timestamp - start->timestamp).count();
        *milliseconds = static_cast<float>(duration < 0.0 ? 0.0 : duration);
        mcu::runtime::recordRuntimeMetric("event_elapsed_query", static_cast<double>(*milliseconds) / 1000.0);
        return succeed();
    }

    cudaError_t destroyEvent(cudaEvent_t event) {
        std::lock_guard<std::mutex> lock(mutex_);
        mcu::runtime::incrementRuntimeCounter("runtime_api_calls");
        if (!validEvent(event)) {
            return fail(cudaErrorInvalidResourceHandle, "cudaEventDestroy received an invalid event");
        }
        event->alive = false;
        delete event;
        mcu::runtime::incrementRuntimeCounter("events_destroyed");
        return succeed();
    }

    cudaError_t takeLastError() {
        std::lock_guard<std::mutex> lock(mutex_);
        cudaError_t result = last_error_;
        last_error_ = cudaSuccess;
        last_message_.clear();
        return result;
    }

    cudaError_t peekLastError() {
        std::lock_guard<std::mutex> lock(mutex_);
        return last_error_;
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

    static bool validEvent(cudaEvent_t event) {
        return event && event->alive;
    }

    std::mutex mutex_;
    mcu::runtime::DeviceContext context_;
    mcu::runtime::BufferRegistry buffers_;
    mcu::runtime::ArtifactCatalog artifacts_;
    mcu::runtime::PipelineCache pipelines_;
    mcu::runtime::StreamRegistry streams_;
    std::uint64_t next_event_identifier_ = 1;
    std::uint64_t next_graph_identifier_ = 1;
    std::uint64_t next_graph_node_identifier_ = 1;
    std::uint64_t next_graph_exec_identifier_ = 1;
    std::unordered_map<cudaStream_t, cudaGraph_t> active_captures_;
    std::unordered_map<cudaStream_t, cudaGraphNode_t> capture_tails_;
    int current_device_ = 0;
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

cudaError_t cudaMemsetAsync(void* dst, int value, std::size_t count, cudaStream_t stream) {
    @autoreleasepool { return runtimeFacade().fillAsync(dst, value, count, stream); }
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
cudaError_t cudaPeekAtLastError() { return runtimeFacade().peekLastError(); }

cudaError_t cudaGetDeviceCount(int* count) {
    @autoreleasepool { return runtimeFacade().getDeviceCount(count); }
}

cudaError_t cudaGetDevice(int* device) {
    @autoreleasepool { return runtimeFacade().getDevice(device); }
}

cudaError_t cudaSetDevice(int device) {
    @autoreleasepool { return runtimeFacade().setDevice(device); }
}

cudaError_t cudaGetDeviceProperties(cudaDeviceProp* properties, int device) {
    @autoreleasepool { return runtimeFacade().getDeviceProperties(properties, device); }
}

cudaError_t cudaDeviceReset() {
    @autoreleasepool { return runtimeFacade().resetDevice(); }
}

cudaError_t cudaEventCreate(cudaEvent_t* event) {
    @autoreleasepool { return runtimeFacade().createEvent(event, cudaEventDefault); }
}

cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags) {
    @autoreleasepool { return runtimeFacade().createEvent(event, flags); }
}

cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    @autoreleasepool { return runtimeFacade().recordEvent(event, stream); }
}

cudaError_t cudaEventSynchronize(cudaEvent_t event) {
    @autoreleasepool { return runtimeFacade().synchronizeEvent(event); }
}

cudaError_t cudaEventElapsedTime(float* milliseconds, cudaEvent_t start, cudaEvent_t end) {
    @autoreleasepool { return runtimeFacade().elapsedEventTime(milliseconds, start, end); }
}

cudaError_t cudaEventDestroy(cudaEvent_t event) {
    @autoreleasepool { return runtimeFacade().destroyEvent(event); }
}

cudaError_t cudaGraphCreate(cudaGraph_t* graph, unsigned int flags) {
    @autoreleasepool { return runtimeFacade().createGraph(graph, flags); }
}

cudaError_t cudaGraphDestroy(cudaGraph_t graph) {
    @autoreleasepool { return runtimeFacade().destroyGraph(graph); }
}

cudaError_t cudaGraphInstantiate(
    cudaGraphExec_t* graph_exec,
    cudaGraph_t graph,
    cudaGraphNode_t* error_node,
    char* log_buffer,
    std::size_t buffer_size
) {
    @autoreleasepool {
        return runtimeFacade().instantiateGraph(graph_exec, graph, error_node, log_buffer, buffer_size);
    }
}

cudaError_t cudaGraphExecDestroy(cudaGraphExec_t graph_exec) {
    @autoreleasepool { return runtimeFacade().destroyGraphExec(graph_exec); }
}

cudaError_t cudaGraphLaunch(cudaGraphExec_t graph_exec, cudaStream_t stream) {
    @autoreleasepool { return runtimeFacade().launchGraph(graph_exec, stream); }
}

cudaError_t cudaGraphAddKernelNode(
    cudaGraphNode_t* node,
    cudaGraph_t graph,
    const cudaGraphNode_t* dependencies,
    std::size_t dependency_count,
    const cudaKernelNodeParams* parameters
) {
    @autoreleasepool {
        return runtimeFacade().addKernelNode(node, graph, dependencies, dependency_count, parameters);
    }
}

cudaError_t cudaGraphAddDependencies(
    cudaGraph_t graph,
    const cudaGraphNode_t* from,
    const cudaGraphNode_t* to,
    std::size_t dependency_count
) {
    @autoreleasepool {
        return runtimeFacade().addDependencies(graph, from, to, dependency_count);
    }
}

cudaError_t cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t* nodes, std::size_t* count) {
    @autoreleasepool { return runtimeFacade().getGraphNodes(graph, nodes, count, false); }
}

cudaError_t cudaGraphGetRootNodes(cudaGraph_t graph, cudaGraphNode_t* nodes, std::size_t* count) {
    @autoreleasepool { return runtimeFacade().getGraphNodes(graph, nodes, count, true); }
}

cudaError_t cudaGraphAddMemcpyNode(
    cudaGraphNode_t* node,
    cudaGraph_t graph,
    const cudaGraphNode_t* dependencies,
    std::size_t dependency_count,
    const cudaMemcpy3DParms* parameters
) {
    @autoreleasepool {
        return runtimeFacade().addMemcpyNode(node, graph, dependencies, dependency_count, parameters);
    }
}

cudaError_t cudaGraphAddMemsetNode(
    cudaGraphNode_t* node,
    cudaGraph_t graph,
    const cudaGraphNode_t* dependencies,
    std::size_t dependency_count,
    const cudaMemsetParams* parameters
) {
    @autoreleasepool {
        return runtimeFacade().addMemsetNode(node, graph, dependencies, dependency_count, parameters);
    }
}

cudaError_t cudaStreamBeginCapture(cudaStream_t stream, unsigned int mode) {
    @autoreleasepool { return runtimeFacade().beginCapture(stream, mode); }
}

cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t* graph) {
    @autoreleasepool { return runtimeFacade().endCapture(stream, graph); }
}

const char* cudaGetErrorString(cudaError_t error) {
    switch (error) {
        case cudaSuccess: return "success";
        case cudaErrorInvalidValue: return "invalid value";
        case cudaErrorMemoryAllocation: return "memory allocation failed";
        case cudaErrorInitializationError: return "runtime initialization failed";
        case cudaErrorLaunchFailure: return "kernel launch failed";
        case cudaErrorNotSupported: return "operation not supported";
        case cudaErrorInvalidDevice: return "invalid device";
        case cudaErrorInvalidResourceHandle: return "invalid resource handle";
    }
    return "unknown MCU runtime error";
}

cudaError_t mcuSetArtifactDirectory(const char* path) {
    @autoreleasepool { return runtimeFacade().setArtifactDirectory(path); }
}

void* mcuKernel(const char* kernel_name) {
    return const_cast<char*>(kernel_name);
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
