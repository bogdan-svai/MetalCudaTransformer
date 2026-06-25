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
    if (count == 0) {
        return true;
    }
    cudaMemcpyKind resolved;
    if (!resolveCopyKind(dst, src, count, kind, resolved, error)) {
        return false;
    }
    if (resolved == cudaMemcpyDeviceToDevice && dst != src) {
        auto destination = reinterpret_cast<std::uintptr_t>(dst);
        auto source = reinterpret_cast<std::uintptr_t>(src);
        if (destination < source + count && source < destination + count) {
            error = "cudaMemcpy does not support overlapping device ranges";
            return false;
        }
    }
    std::memmove(dst, src, count);
    return true;
}

bool BufferRegistry::resolveCopyKind(
    const void* dst,
    const void* src,
    std::size_t count,
    cudaMemcpyKind requested,
    cudaMemcpyKind& resolved,
    std::string& error
) const {
    if (requested < cudaMemcpyHostToHost || requested > cudaMemcpyDefault) {
        error = "cudaMemcpy received an invalid copy kind";
        return false;
    }
    resolved = requested;
    if (resolved == cudaMemcpyDefault) {
        bool dstDevice = contains(dst, count);
        bool srcDevice = contains(src, count);
        resolved = dstDevice
            ? (srcDevice ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice)
            : (srcDevice ? cudaMemcpyDeviceToHost : cudaMemcpyHostToHost);
    }
    if (!validateCopyRange(dst, count, resolved, true, error) ||
        !validateCopyRange(src, count, resolved, false, error)) {
        return false;
    }
    if (resolved == cudaMemcpyDeviceToDevice && dst != src) {
        auto destination = reinterpret_cast<std::uintptr_t>(dst);
        auto source = reinterpret_cast<std::uintptr_t>(src);
        if (destination < source + count && source < destination + count) {
            error = "cudaMemcpy does not support overlapping device ranges";
            return false;
        }
    }
    return true;
}

bool BufferRegistry::fill(
    void* dst,
    int value,
    std::size_t count,
    std::string& error
) {
    if (count == 0) {
        return true;
    }
    if (!contains(dst, count)) {
        error = "cudaMemset device range is outside registered allocation";
        return false;
    }
    std::memset(dst, value, count);
    return true;
}

bool BufferRegistry::find(
    const void* pointer,
    std::size_t count,
    id<MTLBuffer> __strong& buffer,
    NSUInteger& offset
) const {
    auto address = reinterpret_cast<std::uintptr_t>(pointer);
    if (count > UINTPTR_MAX - address) {
        return false;
    }
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
    bool expectedDevice = destination
        ? kind == cudaMemcpyHostToDevice || kind == cudaMemcpyDeviceToDevice
        : kind == cudaMemcpyDeviceToHost || kind == cudaMemcpyDeviceToDevice;
    bool isDevice = contains(pointer, count);
    if (isDevice != expectedDevice) {
        error = expectedDevice
            ? "cudaMemcpy device range is outside registered allocation"
            : "cudaMemcpy host argument refers to a registered device allocation";
        return false;
    }
    return true;
}

bool BufferRegistry::contains(const void* pointer, std::size_t count) const {
    id<MTLBuffer> buffer = nil;
    NSUInteger offset = 0;
    return find(pointer, count, buffer, offset);
}

}  // namespace mcu::runtime
