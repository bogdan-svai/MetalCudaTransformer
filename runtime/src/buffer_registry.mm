#include "runtime_internal.h"

#include <cstring>
#include <cstdint>

namespace mcu::runtime {

bool BufferRegistry::allocate(
    id<MTLDevice> device,
    void** output,
    std::size_t size,
    std::string& error
) {
    id<MTLBuffer> buffer = [device newBufferWithLength:size options:MTLResourceStorageModeShared];
    if (!buffer || ![buffer contents]) {
        error = "Metal shared buffer allocation failed";
        return false;
    }
    void* pointer = [buffer contents];
    allocations_[pointer] = {buffer, size};
    *output = pointer;
    return true;
}

bool BufferRegistry::release(void* pointer, std::string& error) {
    if (!pointer) {
        return true;
    }
    auto found = allocations_.find(pointer);
    if (found == allocations_.end()) {
        error = "cudaFree received an unknown device pointer";
        return false;
    }
    allocations_.erase(found);
    return true;
}

bool BufferRegistry::copy(
    void* dst,
    const void* src,
    std::size_t count,
    cudaMemcpyKind kind,
    std::string& error
) {
    if (!validateCopyRange(dst, count, kind, true, error) ||
        !validateCopyRange(src, count, kind, false, error)) {
        return false;
    }
    std::memmove(dst, src, count);
    return true;
}

bool BufferRegistry::find(
    const void* pointer,
    std::size_t count,
    id<MTLBuffer> __strong& buffer,
    NSUInteger& offset
) const {
    auto address = reinterpret_cast<std::uintptr_t>(pointer);
    for (const auto& item : allocations_) {
        auto base = reinterpret_cast<std::uintptr_t>(item.first);
        if (address >= base && address + count <= base + item.second.size) {
            buffer = item.second.buffer;
            offset = address - base;
            return true;
        }
    }
    return false;
}

bool BufferRegistry::validateCopyRange(
    const void* pointer,
    std::size_t count,
    cudaMemcpyKind kind,
    bool destination,
    std::string& error
) const {
    bool requiresDevice = destination
        ? kind == cudaMemcpyHostToDevice || kind == cudaMemcpyDeviceToDevice
        : kind == cudaMemcpyDeviceToHost || kind == cudaMemcpyDeviceToDevice;
    if (!requiresDevice || kind == cudaMemcpyDefault) {
        return true;
    }
    id<MTLBuffer> buffer = nil;
    NSUInteger offset = 0;
    if (!find(pointer, count, buffer, offset)) {
        error = "cudaMemcpy device range is outside registered allocation";
        return false;
    }
    return true;
}

}  // namespace mcu::runtime
