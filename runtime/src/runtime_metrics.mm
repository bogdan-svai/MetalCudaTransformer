#include "runtime_internal.h"

#include <chrono>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>

namespace mcu::runtime {
namespace {

struct Aggregate {
    double seconds = 0.0;
    std::uint64_t count = 0;
};

class RuntimeMetrics {
public:
    RuntimeMetrics() : started_(std::chrono::steady_clock::now()) {}

    ~RuntimeMetrics() {
        const char* path = std::getenv("MCU_METRICS_FILE");
        if (!path || !*path) {
            return;
        }
        @autoreleasepool {
            NSMutableDictionary* phases = [NSMutableDictionary dictionary];
            NSMutableDictionary* counters = [NSMutableDictionary dictionary];
            std::lock_guard<std::mutex> lock(mutex_);
            for (const auto& item : phases_) {
                NSString* name = [NSString stringWithUTF8String:item.first.c_str()];
                phases[name] = @{
                    @"seconds": @(item.second.seconds),
                    @"count": @(item.second.count),
                };
            }
            for (const auto& item : counters_) {
                counters[[NSString stringWithUTF8String:item.first.c_str()]] = @(item.second);
            }
            double process_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - started_
            ).count();
            NSDictionary* root = @{
                @"schema_version": @1,
                @"device": [NSString stringWithUTF8String:device_.c_str()],
                @"process_seconds": @(process_seconds),
                @"phases": phases,
                @"counters": counters,
            };
            NSError* error = nil;
            NSData* data = [NSJSONSerialization dataWithJSONObject:root options:NSJSONWritingPrettyPrinted error:&error];
            if (data) {
                NSString* target = [NSString stringWithUTF8String:path];
                [data writeToFile:target options:NSDataWritingAtomic error:&error];
            }
        }
    }

    void record(const char* phase, double seconds) {
        std::lock_guard<std::mutex> lock(mutex_);
        Aggregate& aggregate = phases_[phase];
        aggregate.seconds += seconds;
        aggregate.count += 1;
    }

    void increment(const char* name, std::uint64_t amount) {
        std::lock_guard<std::mutex> lock(mutex_);
        counters_[name] += amount;
    }

    void setDevice(const char* name) {
        std::lock_guard<std::mutex> lock(mutex_);
        device_ = name && *name ? name : "unknown";
    }

private:
    std::chrono::steady_clock::time_point started_;
    std::mutex mutex_;
    std::map<std::string, Aggregate> phases_;
    std::map<std::string, std::uint64_t> counters_;
    std::string device_ = "uninitialized";
};

RuntimeMetrics& metrics() {
    static RuntimeMetrics instance;
    return instance;
}

}  // namespace

void recordRuntimeMetric(const char* phase, double seconds) {
    metrics().record(phase, seconds);
}

void incrementRuntimeCounter(const char* name, std::uint64_t amount) {
    metrics().increment(name, amount);
}

void setRuntimeDeviceName(const char* name) {
    metrics().setDevice(name);
}

}  // namespace mcu::runtime
