#include "runtime_internal.h"

#include <chrono>

namespace mcu::runtime {

bool DeviceContext::ensure(std::string& error) {
    if (device_) {
        return true;
    }
    auto started = std::chrono::steady_clock::now();
    device_ = MTLCreateSystemDefaultDevice();
    if (!device_) {
        error = "No Metal device is available";
        return false;
    }
    queue_ = [device_ newCommandQueue];
    if (!queue_) {
        error = "Metal command queue creation failed";
        return false;
    }
    NSString* name = [device_ name];
    setRuntimeDeviceName(name ? [name UTF8String] : "unknown");
    recordRuntimeMetric(
        "device_initialization",
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count()
    );
    return true;
}

}  // namespace mcu::runtime
