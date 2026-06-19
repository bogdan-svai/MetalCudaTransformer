#include "runtime_internal.h"

namespace mcu::runtime {

bool DeviceContext::ensure(std::string& error) {
    if (device_) {
        return true;
    }
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
    return true;
}

}  // namespace mcu::runtime
