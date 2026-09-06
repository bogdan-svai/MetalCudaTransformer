#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <cstdlib>
#include <climits>

static const unsigned char root[32] = {
    0x56,0x15,0x9c,0x1e,0x12,0xbd,0x68,0x51,0x43,0xa1,0xb7,0xb1,0x74,0x15,0x02,0x12,
    0x4d,0xc0,0x7f,0xf6,0x2f,0x57,0xce,0xdb,0x3f,0x2b,0xb7,0x11,0x87,0xe9,0xb2,0x44
};

int main(int argc, char** argv) {
    @autoreleasepool {
        int samples = argc > 1 ? std::atoi(argv[1]) : 10;
        int warmups = argc > 2 ? std::atoi(argv[2]) : 3;
        uint32_t limit = 500000u;
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) return 2;
        NSError* error = nil;
        NSString* path = [[[NSProcessInfo processInfo] environment] objectForKey:@"MCU_NATIVE_METALLIB"];
        id<MTLLibrary> library = [device newLibraryWithFile:path error:&error];
        if (!library) { std::fprintf(stderr, "%s\n", error.localizedDescription.UTF8String); return 3; }
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"native_find_nonce"] error:&error];
        if (!pipeline) return 4;
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLBuffer> rootBuffer = [device newBufferWithBytes:root length:sizeof(root) options:MTLResourceStorageModeShared];
        id<MTLBuffer> limitBuffer = [device newBufferWithBytes:&limit length:sizeof(limit) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bestBuffer = [device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        for (int run = 0; run < samples + warmups; ++run) {
            *(uint32_t*)bestBuffer.contents = UINT_MAX;
            id<MTLCommandBuffer> command = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:rootBuffer offset:0 atIndex:0];
            [encoder setBuffer:limitBuffer offset:0 atIndex:1];
            [encoder setBuffer:bestBuffer offset:0 atIndex:2];
            [encoder dispatchThreads:MTLSizeMake(524288,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
            [encoder endEncoding];
            [command commit];
            [command waitUntilCompleted];
            uint32_t best = *(uint32_t*)bestBuffer.contents;
            if (best != 20632u) { std::fprintf(stderr, "incorrect nonce: %u\n", best); return 5; }
            if (run >= warmups)
                std::printf("gpu_seconds=%.9f\n", command.GPUEndTime - command.GPUStartTime);
        }
        std::printf("device=%s\n", device.name.UTF8String);
        std::printf("nonce=%u\n", *(uint32_t*)bestBuffer.contents);
    }
    return 0;
}
