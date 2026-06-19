#include "runtime_internal.h"

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
        return cached->second;
    }
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
    return pipeline;
}

}  // namespace mcu::runtime
