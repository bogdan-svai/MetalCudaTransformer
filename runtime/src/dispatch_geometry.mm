#include "runtime_internal.h"

namespace mcu::runtime {

bool resolveDispatchGeometry(
    NSDictionary* route,
    id<MTLComputePipelineState> pipeline,
    dim3 block,
    MTLSize& physical,
    std::string& error
) {
    physical = MTLSizeMake(block.x, block.y, block.z);
    NSDictionary* metrics = route[@"optimization_metrics"];
    NSUInteger factor = [metrics[@"coarsening_factor"] unsignedIntegerValue];
    factor = factor == 0 ? 1 : factor;
    if (factor > 1) {
        id originalValue = metrics[@"original_block"];
        NSArray* original = [originalValue isKindOfClass:[NSArray class]] ? originalValue : nil;
        if (original && [original count] == 3 &&
            ([original[0] unsignedIntValue] != block.x ||
             [original[1] unsignedIntValue] != block.y ||
             [original[2] unsignedIntValue] != block.z)) {
            error = "Runtime block does not match the coarsening plan";
            return false;
        }
        if (block.x % factor != 0) {
            error = "Runtime block.x is not divisible by coarsening factor";
            return false;
        }
        physical.width = block.x / factor;
    }
    id recordedValue = metrics[@"physical_block"];
    NSArray* recorded = [recordedValue isKindOfClass:[NSArray class]] ? recordedValue : nil;
    if (recorded && [recorded count] == 3 &&
        ([recorded[0] unsignedIntegerValue] != physical.width ||
         [recorded[1] unsignedIntegerValue] != physical.height ||
         [recorded[2] unsignedIntegerValue] != physical.depth)) {
        error = "Physical block does not match routing metadata";
        return false;
    }
    NSUInteger volume = physical.width * physical.height * physical.depth;
    if (volume > [pipeline maxTotalThreadsPerThreadgroup]) {
        error = "Physical block exceeds Metal pipeline threadgroup limit";
        return false;
    }
    return true;
}

}  // namespace mcu::runtime
