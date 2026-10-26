"""Intrinsic lowering must follow original resolved calls, not shared spellings."""

import json
import platform
import subprocess
import sys
from pathlib import Path

import pytest

from core.backends.device import rewrite_kernel_calls
from core.backends.metal import transpile
from core.frontend.project_parser import parse_project
from core.frontend.semantic import parse_semantic_project
from core.ir.symbols import ProjectSymbolIR
from core.passes.device_linker import build_device_link_plan


@pytest.fixture
def compile_intrinsics(tmp_path, monkeypatch):
    monkeypatch.setenv("MCU_CACHE_DIR", str(tmp_path / "cache"))

    def compile(body, declarations=""):
        path = tmp_path / "intrinsics.cu"
        path.write_text(
            declarations + "\n__global__ void work(int* p, int* out) {\n" + body + "\n}\n",
            encoding="utf-8",
        )
        metadata = parse_project(path)
        result = parse_semantic_project(metadata)
        assert result.complete, result.diagnostics
        project = result.project
        (kernel,) = project.functions_named("work")
        plan = build_device_link_plan(project)
        return metadata, project, kernel, plan

    return compile


@pytest.mark.parametrize(
    "call",
    [
        "atomicAdd(p, 1)",
        "atomicCAS(p, 0, 1)",
        "__syncthreads()",
        "__syncwarp()",
        "__activemask()",
        "__shfl_sync(0xffffffffu, 1, 0)",
        "__shfl_down_sync(0xffffffffu, 1, 1, 32)",
        "__any_sync(0xffffffffu, 1)",
        "__all_sync(0xffffffffu, 1)",
        "memset(p, 0, 4)",
    ],
)
def test_intrinsic_calls_leave_comments_and_literals_intact(compile_intrinsics, call):
    body = f'// șir: {call}\n(void)"{call}";\n{call};'
    metadata, project, kernel, plan = compile_intrinsics(body)
    assert plan.ok, plan.diagnostics
    rewritten = rewrite_kernel_calls(kernel.body, kernel, plan)
    assert f"// șir: {call}" in rewritten
    assert f'"{call}"' in rewritten
    assert f"\n{call};" not in rewritten
    metal = transpile(metadata, semantic_project=project, link_plan=plan)["metal_source"]
    assert f"// șir: {call}" in metal
    assert f'"{call}"' in metal
    restored = ProjectSymbolIR.from_dict(project.to_dict())
    (restored_kernel,) = restored.functions_named("work")
    assert rewrite_kernel_calls(restored_kernel.body, restored_kernel, plan) == rewritten


def test_nested_intrinsics_and_helper_calls_keep_all_edits(compile_intrinsics):
    _, _, kernel, plan = compile_intrinsics(
        "out[0] = __shfl_sync(__activemask(), twice(atomicAdd(p, atomicAdd(p, 1))), 0);",
        "__device__ int twice(int x) { return 2*x; }",
    )
    assert plan.ok, plan.diagnostics
    body = rewrite_kernel_calls(kernel.body, kernel, plan)
    assert body.count("atomic_fetch_add_explicit(") == 2
    assert body.count("memory_order_relaxed") == 2
    assert "twice__mcu_" in body
    assert ", 32, thread_index_in_simdgroup)" in body
    assert "__shfl_sync(" not in body
    assert "__activemask(" not in body


@pytest.mark.parametrize("name, arguments", [("atomicAdd", "p, 1"), ("__syncthreads", "")])
def test_member_named_like_intrinsic_is_rejected(compile_intrinsics, name, arguments):
    declaration = (
        f'struct Object {{ int {name}({"int* p, int x" if arguments else ""}) {{ return 1; }} }};'
    )
    _, _, _, plan = compile_intrinsics(f"Object object; object.{name}({arguments});", declaration)
    assert not plan.ok
    assert any(d["code"] == "UnverifiedIntrinsicDeclaration" for d in plan.diagnostics)


def test_user_helper_named_atomic_add_is_not_an_atomic_operation(compile_intrinsics):
    _, _, kernel, plan = compile_intrinsics(
        "out[0] = atomicAdd(p, 1);",
        "__device__ int atomicAdd(int* p, int x) { return x + 7; }",
    )
    assert plan.ok, plan.diagnostics
    assert not plan.atomic_parameters
    body = rewrite_kernel_calls(kernel.body, kernel, plan)
    assert "atomicAdd__mcu_" in body
    assert "memory_order_relaxed" not in body


@pytest.mark.parametrize("call", ["atomicAdd(p, 1)", "__syncthreads()", "__syncwarp()"])
def test_macro_intrinsic_is_diagnosed_before_emission(compile_intrinsics, call):
    _, _, _, plan = compile_intrinsics("CALL;", f"#define CALL {call}")
    assert not plan.ok
    assert any(d["code"] == "UnsafeIntrinsicCallRewrite" for d in plan.diagnostics)


def test_build_ignores_fake_invalid_warp_call_inside_literal(tmp_path, monkeypatch):
    from core.application.pipeline import CompilerPipeline

    monkeypatch.setenv("MCU_CACHE_DIR", str(tmp_path / "cache"))
    target = tmp_path / "literal.cu"
    target.write_text('__global__ void work() { (void)"__shfl_sync(bad)"; __syncthreads(); }')
    result = CompilerPipeline(
        Path(__file__).resolve().parents[1], str(tmp_path / "result.zip")
    ).build(target)
    assert result.exit_code == 0, result.payload


@pytest.mark.parametrize(
    "name, body",
    [
        ("memset", "out[0] = memset(1, 2, 3);"),
        ("__shfl_sync", "out[0] = __shfl_sync(1, 2, 3);"),
    ],
)
def test_user_intrinsic_names_keep_helper_semantics(compile_intrinsics, name, body):
    _, _, kernel, plan = compile_intrinsics(
        body, f"__device__ int {name}(int a, int b, int c) {{ return a+b+c; }}"
    )
    assert plan.ok, plan.diagnostics
    assert not plan.intrinsic_calls
    assert f"{name}__mcu_" in rewrite_kernel_calls(kernel.body, kernel, plan)


@pytest.mark.skipif(platform.system() != "Darwin", reason="Metal execution requires macOS")
def test_nested_atomic_helper_and_warp_calls_match_gpu_oracle(tmp_path, monkeypatch):
    monkeypatch.setenv("MCU_CACHE_DIR", str(tmp_path / "cache"))
    root = Path(__file__).resolve().parents[1]
    source = tmp_path / "nested.cu"
    source.write_text(
        """
#include <cuda_runtime.h>
__device__ int twice(int x) { return 2*x; }
__global__ void work(int* counter, int* out) {
    int value = 0;
    if (threadIdx.x == 0) value = twice(atomicAdd(counter, atomicAdd(counter, 1)));
    out[threadIdx.x] = __shfl_sync(__activemask(), value, 0);
}
int main() {
    int *counter = nullptr, *out = nullptr;
    if (cudaMalloc(&counter, sizeof(int)) != cudaSuccess) return 10;
    if (cudaMalloc(&out, 32*sizeof(int)) != cudaSuccess) return 11;
    if (cudaMemset(counter, 0, sizeof(int)) != cudaSuccess) return 12;
    work<<<1,32>>>(counter, out);
    int values[32] = {};
    if (cudaMemcpy(values, out, sizeof(values), cudaMemcpyDeviceToHost) != cudaSuccess) return 13;
    for (int i=0; i<32; ++i) if (values[i] != 2) return 20;
    cudaFree(counter);
    cudaFree(out);
    return 0;
}
""",
        encoding="utf-8",
    )
    executable = tmp_path / "nested"
    build = subprocess.run(
        [
            sys.executable,
            str(root / "mcu"),
            "compile",
            str(source),
            "-o",
            str(executable),
            "--json",
        ],
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stdout + build.stderr
    assert json.loads(build.stdout)["ok"]
    result = subprocess.run([str(executable)], capture_output=True, text=True)
    if result.returncode == 10:
        pytest.skip("Metal device unavailable")
    assert result.returncode == 0, result.stdout + result.stderr
