"""Integration tests for original Clang identity through IR, linking and emission."""

from dataclasses import replace
from pathlib import Path

import pytest

from core.backends.device import _apply_call_edits, rewrite_kernel_calls
from core.frontend.project_parser import parse_project
from core.frontend.semantic import parse_semantic_project
from core.ir.symbols import ProjectSymbolIR, generated_symbol_name
from core.passes.device_linker import build_device_link_plan


@pytest.fixture
def compile_calls(tmp_path, monkeypatch):
    monkeypatch.setenv("MCU_CACHE_DIR", str(tmp_path / "cache"))

    def compile(source):
        path = tmp_path / "calls.cu"
        path.write_text(source, encoding="utf-8")
        metadata = parse_project(path)
        semantic = parse_semantic_project(metadata)
        assert semantic.complete, semantic.diagnostics
        (kernel,) = semantic.project.functions_named("work")
        plan = build_device_link_plan(semantic.project)
        return metadata, semantic.project, kernel, plan

    return compile


def test_original_ranges_preserve_utf8_comments_strings_and_overload_order(compile_calls):
    metadata, project, kernel, plan = compile_calls(
        """
__device__ int convert(int x) { return x + 1; }
__device__ float convert(float x) { return x + 2.0f; }
__global__ void work(int* out) {
    // șir: convert(99)
    (void)"convert(98)";
    int convert_count = 0;
    out[0] = convert(1) + int(convert(2.0f)) + convert(convert(3));
}
"""
    )
    assert plan.ok, plan.diagnostics
    assert len(kernel.calls) == 4
    calls = kernel.calls
    assert calls[0].callee_usr != calls[1].callee_usr
    assert calls[0].callee_usr == calls[2].callee_usr == calls[3].callee_usr
    body = rewrite_kernel_calls(kernel.body, kernel, plan)
    assert "// șir: convert(99)" in body
    assert '"convert(98)"' in body
    assert "int convert_count = 0;" in body
    integer = generated_symbol_name("convert(int)", (None,))
    floating = generated_symbol_name("convert(float)", (None,))
    assert f"out[0] = {integer}(1) + int({floating}(2.0f)) + {integer}({integer}(3));" in body
    for call in calls:
        span = call.identifier_range
        assert span is not None
        assert Path(span.file).read_bytes()[span.start : span.end] == b"convert"
    restored = ProjectSymbolIR.from_dict(project.to_dict())
    assert restored.to_dict() == project.to_dict()
    (restored_kernel,) = restored.functions_named("work")
    assert rewrite_kernel_calls(restored_kernel.body, restored_kernel, plan) == body
    cached = parse_semantic_project(metadata)
    assert cached.cached
    assert cached.project.to_dict() == project.to_dict()


def test_function_identity_and_body_location_come_from_original_unit(compile_calls):
    _, project, kernel, plan = compile_calls(
        """
__device__ int helper(int x) { return x + 1; }
__global__ void work(int* out) { out[0] = helper(1); }
"""
    )
    assert plan.ok, plan.diagnostics
    assert kernel.source.file.endswith("calls.cu")
    assert kernel.translation_unit == kernel.source.file
    assert kernel.body_range.file == kernel.source.file
    assert kernel.body_range.start < kernel.body_range.end
    assert kernel.body == " out[0] = helper(1); "
    assert project.functions_named("helper")[0].translation_unit.endswith("calls.cu")


def test_launch_ir_resolves_kernel_symbol_across_translation_units(tmp_path, monkeypatch):
    monkeypatch.setenv("MCU_CACHE_DIR", str(tmp_path / "cache"))
    kernel_path = tmp_path / "kernel.cu"
    host_path = tmp_path / "host.cu"
    kernel_path.write_text(
        "__global__ void work(int* out) { out[threadIdx.x] = 1; }\n",
        encoding="ascii",
    )
    host_path.write_text(
        "void host() { cudaLaunch(work, 1, 32, 0); }\n",
        encoding="ascii",
    )
    metadata = parse_project([kernel_path, host_path])
    semantic = parse_semantic_project(metadata)
    assert semantic.complete, semantic.diagnostics
    assert len(semantic.project.launches) == 1
    launch = semantic.project.launches[0]
    assert launch.kernel == "work"
    assert launch.kernel_id == semantic.project.functions_named("work")[0].symbol_id
    restored = ProjectSymbolIR.from_dict(semantic.project.to_dict())
    assert restored.to_dict() == semantic.project.to_dict()


def test_macro_identity_and_location_come_from_original_preprocessor(compile_calls):
    _, project, _, plan = compile_calls(
        """
#define LOCAL_ADD(x) ((x) + 1)
__global__ void work(int* out) { out[0] = LOCAL_ADD(4); }
"""
    )
    assert plan.ok, plan.diagnostics
    macro = project.macros["LOCAL_ADD"]
    assert macro.parameters == ("x",)
    assert "x" in macro.expression
    assert macro.source.file.endswith("calls.cu")
    assert macro.source.line == 1


def test_type_locations_come_from_original_translation_unit(compile_calls, tmp_path):
    _, project, _, plan = compile_calls(
        """
typedef unsigned int Word;
struct Payload { Word value; };
enum Mode { kMode = 1 };
__global__ void work(int* out) { out[0] = 1; }
"""
    )
    assert plan.ok, plan.diagnostics
    assert project.types["Payload"].source.file.endswith("calls.cu")
    assert project.types["Payload"].source.line == 3
    assert project.types["Mode"].source.line == 4
    assert project.types["Word"].source.line == 2


@pytest.mark.parametrize("invocation", ["CALL(1)", "APPLY(convert, 1)"])
def test_macro_helper_call_is_diagnosed_before_emission(compile_calls, invocation):
    _, _, _, plan = compile_calls(
        f"""
__device__ int convert(int x) {{ return x; }}
#define CALL(x) convert(x)
#define APPLY(f, x) f(x)
__global__ void work(int* out) {{ out[0] = {invocation}; }}
"""
    )
    assert not plan.ok
    (diagnostic,) = [d for d in plan.diagnostics if d["code"] == "UnsafeDeviceCallRewrite"]
    assert diagnostic["line"] == 5
    assert "Macro" in diagnostic["message"]


def test_member_call_does_not_bind_to_free_helper(compile_calls):
    _, _, kernel, plan = compile_calls(
        """
__device__ int helper(int x) { return x; }
struct Object { int helper(int x) { return x; } };
__global__ void work(Object* object, int* out) { out[0] = object->helper(1); }
"""
    )
    (call,) = kernel.calls
    assert call.callee_usr
    assert call.callee_id is None
    assert not plan.ok
    assert not plan.call_targets
    assert any(d["code"] == "MissingDeviceDefinition" for d in plan.diagnostics)


def test_call_edits_reject_stale_body_and_overlaps(compile_calls):
    _, _, kernel, plan = compile_calls(
        """
__device__ int helper(int x) { return x; }
__global__ void work(int* out) { out[0] = helper(1); }
"""
    )
    assert plan.ok, plan.diagnostics
    (call,) = kernel.calls
    with pytest.raises(ValueError, match="unchanged original"):
        _apply_call_edits(" " + kernel.body, kernel, [(call, "renamed")])
    with pytest.raises(ValueError, match="Overlapping"):
        _apply_call_edits(kernel.body, kernel, [(call, "first"), (call, "second")])
    wrong = replace(call, identifier_range=replace(call.identifier_range, start=0))
    with pytest.raises(ValueError, match="original identifier"):
        _apply_call_edits(kernel.body, kernel, [(wrong, "renamed")])


def test_namespace_collision_does_not_choose_an_unrelated_definition(compile_calls):
    _, _, _, plan = compile_calls(
        """
namespace first {
__device__ int helper(int x) { return x + 1; }
}
namespace second {
__device__ int helper(int x) { return x + 2; }
}
__global__ void work(int* out) { out[0] = first::helper(1); }
"""
    )
    assert not plan.ok
    assert not plan.call_targets
    assert any(
        d["code"] in {"UnsafeDeviceCallRewrite", "MissingDeviceDefinition"}
        for d in plan.diagnostics
    )


def test_cross_unit_declaration_resolves_to_original_definition(tmp_path, monkeypatch):
    monkeypatch.setenv("MCU_CACHE_DIR", str(tmp_path / "cache"))
    header = tmp_path / "helper.h"
    header.write_text("__device__ int helper(int);\n", encoding="utf-8")
    helper = tmp_path / "helper.cu"
    helper.write_text('#include "helper.h"\n__device__ int helper(int x) { return x + 1; }\n')
    kernel_path = tmp_path / "kernel.cu"
    kernel_path.write_text(
        '#include "helper.h"\n__global__ void work(int* out) { out[0] = helper(1); }\n'
    )
    result = parse_semantic_project(parse_project([kernel_path, helper]))
    assert result.complete, result.diagnostics
    plan = build_device_link_plan(result.project)
    assert plan.ok, plan.diagnostics
    (kernel,) = result.project.functions_named("work")
    (definition,) = result.project.functions_named("helper")
    (call,) = kernel.calls
    assert call.callee_usr == definition.clang_usr
    assert call.callee_id == definition.symbol_id
    assert call.identifier_range.file == str(kernel_path)
    assert definition.body_range.file == str(helper)
