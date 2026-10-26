import json
import platform
import subprocess
import sys
import zipfile
from pathlib import Path

from core.frontend.project_parser import parse_project
from core.frontend.semantic import parse_semantic_project
from core.passes.device_linker import build_device_link_plan
from core.registry.catalog import default_registry
import pytest


ROOT = Path(__file__).resolve().parents[1]
MCU = ROOT / "mcu"


def _write_case(tmp_path, source):
    target = tmp_path / "case.cu"
    target.write_text(source, encoding="utf-8")
    return target


def _build(target, *extra):
    result = subprocess.run(
        [sys.executable, str(MCU), "build", str(target), "--json", *extra],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result, json.loads(result.stdout)


def test_device_helper_is_emitted_and_recorded(tmp_path):
    target = _write_case(
        tmp_path,
        """
        __device__ int twice(int value) { return value * 2; }
        __global__ void work(int* values) {
            values[threadIdx.x] = twice(threadIdx.x);
        }
        void host() { cudaLaunch(work, 1, 32, 0); }
        """,
    )

    result, payload = _build(target)

    assert result.returncode == 0, result.stderr
    assert payload["build"]["symbol_manifest"] == "symbol_manifest.json"
    with zipfile.ZipFile(ROOT / "mcu_artifacts.zip") as archive:
        metal = archive.read("generated.metal").decode()
        manifest = json.loads(archive.read("symbol_manifest.json"))
    helper = next(
        symbol for symbol in manifest["symbols"].values()
        if symbol["name"] == "twice"
    )
    generated_name = helper["generated_variants"][0]["generated_name"]
    assert f"inline int {generated_name}(int value)" in metal
    assert f"{generated_name}(thread_position_in_threadgroup.x)" in metal


def test_address_space_specializes_device_helper(tmp_path):
    target = _write_case(
        tmp_path,
        """
        __device__ int load_value(int* value) { return value[0]; }
        __global__ void work(int* input, int* output) {
            int local[1];
            local[0] = 7;
            output[0] = load_value(input) + load_value(local);
        }
        void host() { cudaLaunch(work, 1, 1, 0, 0); }
        """,
    )
    metadata = parse_project(target)
    semantic = parse_semantic_project(metadata)
    plan = build_device_link_plan(semantic.project)

    assert semantic.complete
    assert plan.ok, plan.diagnostics
    variants = [
        variant for variant in plan.ordered_variants
        if semantic.project.functions[variant.symbol_id].name == "load_value"
    ]
    assert {variant.parameter_address_spaces for variant in variants} == {
        ("device",),
        ("thread",),
    }


def test_missing_device_definition_is_source_mapped(tmp_path):
    target = _write_case(
        tmp_path,
        """
        __device__ int missing(int value);
        __global__ void work(int* output) {
            output[0] = missing(3);
        }
        void host() { cudaLaunch(work, 1, 1, 0); }
        """,
    )

    result, payload = _build(target)

    assert result.returncode == 1
    diagnostic = next(item for item in payload["diagnostics"] if item["code"] == "MissingDeviceDefinition")
    assert diagnostic["file"] == target.as_posix()
    assert diagnostic["line"] == 4


def test_recursive_device_call_is_rejected(tmp_path):
    target = _write_case(
        tmp_path,
        """
        __device__ int recurse(int value) {
            return value == 0 ? 0 : recurse(value - 1);
        }
        __global__ void work(int* output) { output[0] = recurse(2); }
        void host() { cudaLaunch(work, 1, 1, 0); }
        """,
    )

    result, payload = _build(target)

    assert result.returncode == 1
    assert any(item["code"] == "RecursiveDeviceCall" for item in payload["diagnostics"])


def test_device_helper_using_implicit_geometry_is_rejected(tmp_path):
    target = _write_case(
        tmp_path,
        """
        __device__ int lane() { return threadIdx.x; }
        __global__ void work(int* output) { output[0] = lane(); }
        void host() { cudaLaunch(work, 1, 1, 0); }
        """,
    )

    result, payload = _build(target)

    assert result.returncode == 1
    assert any(item["code"] == "DeviceHelperUsesGeometry" for item in payload["diagnostics"])


def test_host_transform_removes_device_only_and_keeps_dual_function():
    from core.backends.host import transform_host_source

    transformed = transform_host_source(
        """
        __device__ int gpu_only(int value) { return value; }
        __host__ __device__ int dual(int value) { return value + 1; }
        int host() { return dual(2); }
        """
    )

    assert "gpu_only" not in transformed
    assert "int dual(int value)" in transformed
    assert "__device__" not in transformed


def test_translation_registry_rejects_wrong_intrinsic_arity():
    registry = default_registry()

    assert registry.resolve("memset", 3)
    assert registry.resolve("memset", 2) is False
    assert registry.resolve("unknown_intrinsic", 1) is None


def test_overloads_keep_distinct_canonical_symbols_and_bodies(tmp_path):
    target = _write_case(
        tmp_path,
        """
        __device__ int convert(int value) { return value + 1; }
        __device__ float convert(float value) { return value * 2.0f; }
        __global__ void work(int* integers, float* floats) {
            integers[0] = convert(3);
            floats[0] = convert(4.0f);
        }
        void host() { cudaLaunch(work, 1, 1, 0, 0); }
        """,
    )

    result, _payload = _build(target)

    assert result.returncode == 0, result.stderr
    with zipfile.ZipFile(ROOT / "mcu_artifacts.zip") as archive:
        metal = archive.read("generated.metal").decode()
    assert "return value + 1;" in metal
    assert "return value * 2.0f;" in metal


@pytest.mark.skipif(platform.system() != "Darwin", reason="Metal diagnostics require Xcode")
def test_device_helper_metal_error_maps_to_cuda_helper(tmp_path):
    target = _write_case(
        tmp_path,
        """__device__ int broken_helper(int value) {
    return value + missing_device_value;
}
__global__ void work(int* output) {
    output[0] = broken_helper(1);
}
int main() { return 0; }
""",
    )

    result, payload = _build(target, "--native", "--output", str(tmp_path / "native"))

    assert result.returncode == 1
    diagnostic = next(item for item in payload["diagnostics"] if item["code"] == "MetalCompileError")
    assert diagnostic["file"] == target.as_posix()
    assert diagnostic["line"] == 2
    assert "missing_device_value" in diagnostic["message"]
