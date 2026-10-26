"""Executable feasibility probes for original CUDA call locations.

These exercise libclang directly, not the production synthetic frontend. Native
CUDA attributes are supplied in a forced header so source bytes stay untouched.
"""

from ctypes import POINTER, byref, c_uint

import pytest
from clang.cindex import Config, CursorKind, File, Index, SourceLocation, c_object_p, conf

from core.frontend.clang_probe import _configure
from core.frontend.semantic import _build_semantic_source, _clang_args


@pytest.fixture
def parse_cuda(tmp_path):
    assert _configure(Config), "The CUDA location probes require configured libclang"
    header = tmp_path / "cuda_attributes.h"
    header.write_text(
        "#define __device__ __attribute__((device))\n"
        "#define __global__ __attribute__((global))\n",
        encoding="utf-8",
    )

    def parse(source, name="calls.cu"):
        path = tmp_path / name
        path.write_text(source, encoding="utf-8")
        tu = Index.create().parse(
            str(path),
            args=[
                "-x",
                "cuda",
                "--cuda-host-only",
                "-nocudainc",
                "-nocudalib",
                "-std=c++17",
                "-include",
                str(header),
            ],
        )
        errors = [str(d) for d in tu.diagnostics if d.severity >= d.Error]
        assert not errors, errors
        return path, tu

    return parse


def _calls(tu):
    return [c for c in tu.cursor.walk_preorder() if c.kind == CursorKind.CALL_EXPR]


def _physical_location(location, kind):
    """Query spelling/expansion explicitly; cindex.location exposes instantiation."""
    function = getattr(conf.lib, f"clang_get{kind}Location")
    function.argtypes = [SourceLocation, POINTER(c_object_p), *([POINTER(c_uint)] * 3)]
    function.restype = None
    file = c_object_p()
    line, column, offset = c_uint(), c_uint(), c_uint()
    function(location, byref(file), byref(line), byref(column), byref(offset))
    return File(file).name, offset.value


def test_direct_cuda_calls_have_distinct_symbols_and_exact_utf8_ranges(parse_cuda):
    source = """__device__ int helper(int x) { return x; }
__device__ float helper(float x) { return x; }
namespace other { __device__ int helper(int x) { return x; } }
struct Object { __device__ int helper(int x) { return x; } };
__global__ void kernel(Object* object, int* out) {
    // șir: helper(100)
    const char* text = "helper(200)";
    *out = helper(1) + int(helper(2.0f)) + other::helper(3) + object->helper(4);
}
"""
    path, tu = parse_cuda(source)
    calls = _calls(tu)
    assert len(calls) == 4  # Neither comments nor string contents are calls.
    assert len({c.referenced.get_usr() for c in calls}) == 4
    assert all(c.referenced.get_usr() for c in calls)
    assert calls[-1].referenced.kind == CursorKind.CXX_METHOD
    assert calls[2].referenced.semantic_parent.spelling == "other"
    data = source.encode("utf-8")
    for call, argument in zip(calls, ("1", "2.0f", "3", "4")):
        token = next(t for t in call.get_tokens() if t.spelling == "helper")
        start, end = token.extent.start, token.extent.end
        expected = data.index(f"helper({argument})".encode())
        assert (start.offset, end.offset) == (expected, expected + len(b"helper"))
        assert data[start.offset : end.offset] == b"helper"
        assert start.offset != source.index(f"helper({argument})")
        assert str(start.file) == str(path)
        assert _physical_location(start, "Spelling") == (str(path), expected)
        assert _physical_location(start, "Expansion") == (str(path), expected)


@pytest.mark.parametrize(
    ("definition", "invocation", "spelling_text", "reference_bytes"),
    [
        ("#define CALL(x) helper(x)", "CALL(2)", "CALL(2)", b"CALL(2)"),
        ("#define APPLY(f,x) f(x)", "APPLY(helper,3)", "helper,3", b""),
    ],
)
def test_macro_calls_require_explicit_mapping_policy(
    parse_cuda, definition, invocation, spelling_text, reference_bytes
):
    source = (
        "__device__ int helper(int x) { return x; }\n"
        f"{definition}\n__global__ void kernel(int* out) {{ *out = {invocation}; }}\n"
    )
    path, tu = parse_cuda(source)
    (call,) = _calls(tu)
    assert call.referenced.spelling == "helper"
    (reference,) = [
        c
        for c in call.walk_preorder()
        if c.kind == CursorKind.DECL_REF_EXPR and c.spelling == "helper"
    ]
    data = source.encode()
    expansion = data.index(invocation.encode())
    spelling = data.index(spelling_text.encode())
    assert _physical_location(reference.location, "Expansion") == (str(path), expansion)
    assert _physical_location(reference.location, "Spelling") == (str(path), spelling)
    assert data[reference.extent.start.offset : reference.extent.end.offset] == reference_bytes
    # A direct-call edit must never treat either of these extents as "helper".
    assert reference_bytes != b"helper"
    token = next(t for t in reference.get_tokens() if t.spelling == "helper")
    expected_token = data.index(b"helper(x)") if invocation == "CALL(2)" else spelling
    assert token.extent.start.offset == expected_token


def test_internal_symbols_are_distinct_across_original_translation_units(parse_cuda):
    source = (
        "static __device__ int helper(int x) { return x; }\n"
        "__global__ void kernel(int* out) { *out = helper(1); }\n"
    )
    first_path, first = parse_cuda(source, "first.cu")
    second_path, second = parse_cuda(source, "second.cu")
    (first_call,) = _calls(first)
    (second_call,) = _calls(second)
    assert first_call.referenced.get_usr() != second_call.referenced.get_usr()
    assert str(first_call.location.file) == str(first_path)
    assert str(second_call.location.file) == str(second_path)


def test_synthetic_frontend_offsets_do_not_address_original_source(parse_cuda, tmp_path):
    source = (
        "__device__ int helper(int x) { return x; }\n"
        "__global__ void kernel(int* out) { *out = helper(1); }\n"
    )
    original, _ = parse_cuda(source)
    synthetic = tmp_path / "synthetic.cpp"
    synthetic.write_text(_build_semantic_source([original]), encoding="utf-8")
    tu = Index.create().parse(str(synthetic), args=_clang_args([original]))
    assert not [str(d) for d in tu.diagnostics if d.severity >= d.Error]
    (call,) = [c for c in _calls(tu) if c.spelling == "helper"]
    token = next(t for t in call.get_tokens() if t.spelling == "helper")
    offset = token.extent.start.offset
    assert str(token.location.file) == str(synthetic)
    assert offset != source.encode().index(b"helper(1)")
    assert source.encode()[offset : offset + 6] != b"helper"
    assert synthetic.read_bytes()[offset : offset + 6] == b"helper"


def test_cuda_launch_keeps_original_kernel_reference(parse_cuda, tmp_path):
    # Minimal host-side declarations needed by Clang's native CUDA launch AST.
    header = tmp_path / "launch_runtime.h"
    header.write_text(
        "struct dim3 { dim3(unsigned x = 1); };\n"
        "int cudaConfigureCall(dim3, dim3, unsigned long = 0, void* = nullptr);\n",
        encoding="utf-8",
    )
    source = (
        '#include "launch_runtime.h"\n'
        "__global__ void kernel(int* out) { *out = 1; }\n"
        "void launch(int* out) { kernel<<<1, 32>>>(out); }\n"
    )
    path, tu = parse_cuda(source)
    (reference,) = [
        c
        for c in tu.cursor.walk_preorder()
        if c.kind == CursorKind.DECL_REF_EXPR and c.spelling == "kernel"
    ]
    (definition,) = [
        c
        for c in tu.cursor.get_children()
        if c.kind == CursorKind.FUNCTION_DECL and c.spelling == "kernel"
    ]
    assert reference.referenced.get_usr() == definition.get_usr()
    start, end = reference.extent.start, reference.extent.end
    assert str(start.file) == str(path)
    assert source.encode()[start.offset : end.offset] == b"kernel"
    assert start.offset == source.encode().index(b"kernel<<<")


def test_original_translation_unit_preserves_include_locations(parse_cuda, tmp_path):
    header = tmp_path / "helper.h"
    header.write_text("__device__ int helper(int);\n", encoding="utf-8")
    source = '#include "helper.h"\n' "__global__ void kernel(int* out) { *out = helper(1); }\n"
    path, tu = parse_cuda(source)
    (call,) = _calls(tu)
    assert str(call.referenced.location.file) == str(header)
    assert str(call.location.file) == str(path)
    assert call.referenced.get_usr()
