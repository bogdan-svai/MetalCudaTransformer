from core.backends.host import has_host_code, transform_host_source
from core.frontend.regex_parser import extract_launches


def test_host_transform_removes_kernel_and_rewrites_triple_chevron():
    source = """
    __global__ void scale(float* values, int count) {
        values[threadIdx.x] *= 2.0f;
    }

    int main() {
        float* values = nullptr;
        scale<<<4, 256>>>(values, 1024);
        return 0;
    }
    """

    transformed = transform_host_source(source)

    assert "__global__" not in transformed
    assert "threadIdx" not in transformed
    assert "cudaLaunch(scale, 4, 256, values, 1024);" in transformed
    assert transformed.startswith('#include "cuda_to_metal.h"')


def test_kernel_only_source_is_not_a_host_compilation_unit():
    transformed = transform_host_source("__global__ void only_kernel(float* values) { values[0] = 1; }")

    assert has_host_code(transformed) is False


def test_cuda_launch_keeps_typed_arguments():
    source = "void run(float* data) { cudaLaunch(work, dim3(2), dim3(32), data, 64, 2.0f); }"

    transformed = transform_host_source(source)

    assert "cudaLaunch(work, dim3(2), dim3(32), data, 64, 2.0f);" in transformed


def test_launch_parser_stops_at_balanced_call_inside_if(tmp_path):
    source = """
    void run(float* data) {
        if (cudaLaunch(work, 2, 32, data, 64, 2.0f) != 0) { return; }
        cudaDeviceSynchronize();
    }
    """

    launches = extract_launches(tmp_path / "case.cu", source)

    assert len(launches) == 1
    assert launches[0].args == ["data", "64", "2.0f"]
