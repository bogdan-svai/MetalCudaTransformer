#include "runtime_internal.h"

#include <chrono>

namespace mcu::runtime {

void PipelineCache::reset() {
    pipelines_.clear();
}

id<MTLComputePipelineState> PipelineCache::get(
    id<MTLDevice> device,
    id<MTLLibrary> library,
    NSString* entry,
    std::string& error
) {
    std::string key = [entry UTF8String];
    auto cached = pipelines_.find(key);
    if (cached != pipelines_.end()) {
        incrementRuntimeCounter("pipeline_cache_hits");
        return cached->second;
    }
    auto started = std::chrono::steady_clock::now();
    id<MTLFunction> function = [library newFunctionWithName:entry];
    if (!function) {
        error = "Metal entry point was not found";
        return nil;
    }
    NSError* nativeError = nil;
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&nativeError];
    if (!pipeline) {
        error = nativeError ? [[nativeError localizedDescription] UTF8String] : "Pipeline creation failed";
        return nil;
    }
    pipelines_[key] = pipeline;
    incrementRuntimeCounter("pipeline_cache_misses");
    recordRuntimeMetric(
        "pipeline_creation",
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count()
    );
    return pipeline;
}

}  // namespace mcu::runtime
