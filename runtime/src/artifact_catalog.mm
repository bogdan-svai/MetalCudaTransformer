#include "runtime_internal.h"

#include <mach-o/dyld.h>
#include <vector>

namespace mcu::runtime {

bool ArtifactCatalog::setDirectory(const char* path, std::string& error) {
    if (!path || !*path) {
        error = "Artifact directory is empty";
        return false;
    }
    directory_ = path;
    library_ = nil;
    routing_ = nil;
    return true;
}

bool ArtifactCatalog::ensure(id<MTLDevice> device, std::string& error) {
    if (library_ && routing_) {
        return true;
    }
    if (directory_.empty()) {
        directory_ = executableDirectory();
    }
    NSString* directory = [NSString stringWithUTF8String:directory_.c_str()];
    NSString* metalPath = [directory stringByAppendingPathComponent:@"generated.metal"];
    NSString* routingPath = [directory stringByAppendingPathComponent:@"routing_table.txt"];
    NSError* nativeError = nil;
    NSString* source = [NSString stringWithContentsOfFile:metalPath
                                                  encoding:NSUTF8StringEncoding
                                                     error:&nativeError];
    if (!source) {
        error = nativeError ? [[nativeError localizedDescription] UTF8String] : "Cannot load generated.metal";
        return false;
    }
    library_ = [device newLibraryWithSource:source options:nil error:&nativeError];
    if (!library_) {
        error = nativeError ? [[nativeError localizedDescription] UTF8String] : "MSL JIT compilation failed";
        return false;
    }
    NSData* data = [NSData dataWithContentsOfFile:routingPath options:0 error:&nativeError];
    NSDictionary* root = data ? [NSJSONSerialization JSONObjectWithData:data options:0 error:&nativeError] : nil;
    if (!root || [root[@"schema_version"] integerValue] != 1) {
        error = nativeError ? [[nativeError localizedDescription] UTF8String] : "Unsupported routing schema";
        return false;
    }
    routing_ = root;
    return true;
}

NSDictionary* ArtifactCatalog::route(const char* kernel_name) const {
    NSDictionary* kernels = routing_[@"translated_kernels"];
    return kernels[[NSString stringWithUTF8String:kernel_name]];
}

std::string ArtifactCatalog::executableDirectory() {
    uint32_t size = 0;
    _NSGetExecutablePath(nullptr, &size);
    std::vector<char> path(size);
    if (_NSGetExecutablePath(path.data(), &size) != 0) {
        return ".";
    }
    NSString* executable = [NSString stringWithUTF8String:path.data()];
    return [[[executable stringByStandardizingPath] stringByDeletingLastPathComponent] UTF8String];
}

}  // namespace mcu::runtime
