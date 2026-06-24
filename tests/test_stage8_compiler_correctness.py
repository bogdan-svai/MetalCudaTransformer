from pathlib import Path

from core.backends.metal import transpile
from core.driver import NvccInvocationError, parse_nvcc_arguments
from core.frontend.project_parser import parse_project
from core.frontend.semantic import parse_semantic_project
from core.passes.device_linker import build_device_link_plan
from core.passes.validation import validate


def _semantic_build(tmp_path, source):
    target = tmp_path / "case.cu"
    target.write_text(source, encoding="ascii")
    metadata = parse_project([target])
    semantic = parse_semantic_project(metadata)
    assert semantic.complete, semantic.diagnostics
    assert not [item for item in semantic.diagnostics if item["severity"] == "error"]
    plan = build_device_link_plan(semantic.project)
    assert not [item for item in plan.diagnostics if item["severity"] == "error"], plan.diagnostics
    return semantic.project, plan, transpile(metadata, semantic_project=semantic.project, link_plan=plan)


def test_parameter_arrays_decay_and_recovered_calls_are_emitted(tmp_path):
    project, _plan, generated = _semantic_build(
        tmp_path,
        """typedef unsigned char BYTE;
__device__ unsigned long d_strlen(const char* text) {
    unsigned long count = 0;
    while (text[count]) count++;
    return count;
}
__device__ unsigned long consume(const BYTE data[]) {
    return d_strlen((const char*)data);
}
__global__ void kernel(BYTE* input, unsigned long* output) {
    output[0] = consume(input);
}
""",
    )

    consume = project.functions_named("consume")[0]
    assert consume.parameters[0].type.canonical == "const unsigned char*"
    assert consume.parameters[0].type.pointer_depth == 1
    assert {call.callee_name for call in consume.calls} == {"d_strlen"}
    metal = generated["metal_source"]
    assert "uchar[]" not in metal
    assert "d_strlen(" not in metal
    assert "reinterpret_cast<const device char*>(data)" in metal


def test_recovered_memset_uses_typed_intrinsic_prelude(tmp_path):
    _project, plan, generated = _semantic_build(
        tmp_path,
        """typedef unsigned char BYTE;
__device__ int clear_local() {
    BYTE local[8];
    memset(local, 0, 8);
    return local[0];
}
__global__ void kernel(int* output) { output[0] = clear_local(); }
""",
    )

    assert "memset" in plan.intrinsics_used
    metal = generated["metal_source"]
    assert "mcu_device_memset(local, 0, 8)" in metal
    assert "inline thread T* mcu_device_memset" in metal


def test_nvcc_link_accepts_libraries_and_rejects_them_during_compile(tmp_path):
    source = tmp_path / "main.cu"
    source.write_text("int main() { return 0; }\n", encoding="ascii")
    linked = parse_nvcc_arguments(
        [str(source), "-L", "native libs", "-lm", "-l", "z", "-o", "app"],
        working_directory=tmp_path,
    )
    assert linked.library_paths == [str(tmp_path / "native libs")]
    assert linked.libraries == ["m", "z"]

    try:
        parse_nvcc_arguments(["-c", str(source), "-lm"], working_directory=tmp_path)
    except NvccInvocationError as exc:
        assert exc.code == "InvalidNvccInvocation"
    else:
        raise AssertionError("Compile-only invocation accepted a link-only library flag")


def test_ast_memory_ir_accepts_fixed_extent_and_rejects_vla(tmp_path):
    target = tmp_path / "arrays.cu"
    target.write_text(
        """#define WIDTH (8 * 4)
__global__ void fixed_array(int* output) {
    int values[WIDTH];
    values[0] = 7;
    output[0] = values[0];
}
__global__ void variable_array(int* output, int count) {
    int values[count];
    values[0] = count;
    output[0] = values[0];
}
""",
        encoding="ascii",
    )
    metadata = parse_project([target])
    semantic = parse_semantic_project(metadata)
    assert semantic.complete
    fixed = semantic.project.functions_named("fixed_array")[0]
    variable = semantic.project.functions_named("variable_array")[0]
    assert fixed.local_objects[0].array_extent == 32
    assert fixed.local_objects[0].variable_length is False
    assert variable.local_objects[0].variable_length is True
    plan = build_device_link_plan(semantic.project)
    assert any(item["code"] == "UnsupportedVariableLengthArray" for item in plan.diagnostics)


def test_dynamic_shared_launch_contract_is_validated_independently_of_names(tmp_path):
    target = tmp_path / "shared.cu"
    target.write_text(
        """__global__ void dynamic_kernel(unsigned int* output) {
    extern __shared__ unsigned int workspace[];
    workspace[threadIdx.x] = threadIdx.x;
    output[threadIdx.x] = workspace[threadIdx.x];
}
void host(unsigned int* output) {
    dynamic_kernel<<<1, 32>>>(output);
}
""",
        encoding="ascii",
    )
    metadata = parse_project([target])
    diagnostics = validate(metadata)
    assert any(item["code"] == "DynamicSharedSizeMissing" for item in diagnostics)


def test_multiple_dynamic_shared_declarations_are_rejected_from_ast(tmp_path):
    target = tmp_path / "duplicate_shared.cu"
    target.write_text(
        """__global__ void duplicate(unsigned int* output) {
    extern __shared__ unsigned int first[];
    extern __shared__ unsigned int second[];
    output[0] = first[0] + second[0];
}
""",
        encoding="ascii",
    )
    metadata = parse_project([target])
    semantic = parse_semantic_project(metadata)
    assert semantic.complete
    plan = build_device_link_plan(semantic.project)
    assert any(item["code"] == "InvalidSharedDeclaration" for item in plan.diagnostics)


def test_mixed_device_and_thread_pointer_expression_is_rejected(tmp_path):
    target = tmp_path / "mixed_spaces.cu"
    target.write_text(
        """__device__ int load_value(const int* value) { return value[0]; }
__global__ void mixed(int* input, int* output, int choose_device) {
    int local[1];
    local[0] = 3;
    output[0] = load_value(choose_device ? input : local);
}
""",
        encoding="ascii",
    )
    metadata = parse_project([target])
    semantic = parse_semantic_project(metadata)
    assert semantic.complete
    plan = build_device_link_plan(semantic.project)
    assert any(item["code"] == "IllegalAddressSpaceCast" for item in plan.diagnostics)


def test_unsupported_atomic_value_type_is_source_mapped(tmp_path):
    target = tmp_path / "float_atomic.cu"
    target.write_text(
        """__global__ void unsupported(float* value) { atomicMin(value, 1.0f); }
""",
        encoding="ascii",
    )
    metadata = parse_project([target])
    semantic = parse_semantic_project(metadata)
    assert semantic.complete
    plan = build_device_link_plan(semantic.project)
    diagnostic = next(item for item in plan.diagnostics if item["code"] == "UnsupportedAtomicType")
    assert diagnostic["file"] == str(target)
    assert diagnostic["line"] == 1


def test_atomic_requirement_propagates_from_helper_to_shared_caller(tmp_path):
    project, plan, generated = _semantic_build(
        tmp_path,
        """__device__ void increment(unsigned int* value) { atomicAdd(value, 1u); }
__global__ void caller(unsigned int* output) {
    __shared__ unsigned int shared_value;
    if (threadIdx.x == 0) shared_value = 0u;
    __syncthreads();
    increment(&shared_value);
    __syncthreads();
    if (threadIdx.x == 0) output[0] = shared_value;
}
""",
    )
    kernel = project.functions_named("caller")[0]
    assert plan.atomic_locals[kernel.symbol_id]["shared_value"]["type"] == "atomic_uint"
    metal = generated["metal_source"]
    assert "threadgroup atomic_uint shared_value" in metal
    assert "threadgroup atomic_uint* value" in metal


def test_atomic_helper_rejects_thread_private_storage(tmp_path):
    target = tmp_path / "private_atomic.cu"
    target.write_text(
        """__device__ void increment(int* value) { atomicAdd(value, 1); }
__global__ void caller(int* output) {
    int private_value = 0;
    increment(&private_value);
    output[0] = private_value;
}
""",
        encoding="ascii",
    )
    metadata = parse_project([target])
    semantic = parse_semantic_project(metadata)
    assert semantic.complete
    plan = build_device_link_plan(semantic.project)
    assert any(item["code"] == "UnsupportedAtomicAddressSpace" for item in plan.diagnostics)


def test_local_pointer_with_conflicting_pointee_spaces_is_rejected(tmp_path):
    target = tmp_path / "pointer_flow.cu"
    target.write_text(
        """__device__ int load_value(const int* value) { return value[0]; }
__global__ void caller(int* input, int* output, int choose_device) {
    int local[1];
    int* selected;
    if (choose_device) selected = input;
    else selected = local;
    output[0] = load_value(selected);
}
""",
        encoding="ascii",
    )
    metadata = parse_project([target])
    semantic = parse_semantic_project(metadata)
    assert semantic.complete
    plan = build_device_link_plan(semantic.project)
    assert any(item["code"] == "IllegalAddressSpaceCast" for item in plan.diagnostics)


def test_direct_dereference_of_mixed_local_pointer_is_rejected(tmp_path):
    target = tmp_path / "direct_pointer_flow.cu"
    target.write_text(
        """__global__ void caller(int* input, int* output, int choose_device) {
    int local[1];
    int* selected;
    selected = choose_device ? input : local;
    output[0] = *selected;
}
""",
        encoding="ascii",
    )
    metadata = parse_project([target])
    semantic = parse_semantic_project(metadata)
    assert semantic.complete
    plan = build_device_link_plan(semantic.project)
    assert any(item["code"] == "IllegalAddressSpaceCast" for item in plan.diagnostics)


def test_indexed_atomic_pointer_loads_and_stores_are_lowered(tmp_path):
    _project, _plan, generated = _semantic_build(
        tmp_path,
        """__global__ void indexed(unsigned int* values, unsigned int* output) {
    unsigned int index = threadIdx.x;
    atomicAdd(&values[index], 1u);
    if (index == 0) values[index] = 7u;
    output[index] = values[index];
}
""",
    )
    metal = generated["metal_source"]
    assert "device atomic_uint* values" in metal
    assert "atomic_store_explicit(&values[index], 7u" in metal
    assert "atomic_load_explicit(&values[index], memory_order_relaxed)" in metal


def test_shared_declaration_inside_device_helper_is_rejected(tmp_path):
    target = tmp_path / "helper_shared.cu"
    target.write_text(
        """__device__ int helper() { __shared__ int value; return value; }
__global__ void caller(int* output) { output[0] = helper(); }
""",
        encoding="ascii",
    )
    metadata = parse_project([target])
    semantic = parse_semantic_project(metadata)
    assert semantic.complete
    plan = build_device_link_plan(semantic.project)
    assert any(item["code"] == "InvalidSharedDeclaration" for item in plan.diagnostics)


def test_fallback_parser_does_not_invent_functions_from_control_flow(tmp_path):
    target = tmp_path / "host.cpp"
    target.write_text(
        """int read_value(char* buffer, void* file) {
    if (!fgets(buffer, sizeof(buffer), file)) {
        return -1;
    }
    return 0;
}
""",
        encoding="ascii",
    )
    metadata = parse_project([target])
    assert [item["name"] for item in metadata["functions"]] == ["read_value"]


def test_compound_plain_access_to_atomic_target_is_rejected(tmp_path):
    target = tmp_path / "compound_atomic.cu"
    target.write_text(
        """__global__ void invalid(unsigned int* values) {
    atomicAdd(values, 1u);
    (*values)++;
}
""",
        encoding="ascii",
    )
    metadata = parse_project([target])
    semantic = parse_semantic_project(metadata)
    assert semantic.complete
    plan = build_device_link_plan(semantic.project)
    assert any(item["code"] == "InvalidAtomicNonAtomicAccess" for item in plan.diagnostics)


def test_multidimensional_array_parameter_decay_fails_explicitly(tmp_path):
    target = tmp_path / "matrix_parameter.cu"
    target.write_text(
        """__global__ void matrix(int values[][4], int* output) { output[0] = values[0][0]; }
""",
        encoding="ascii",
    )
    metadata = parse_project([target])
    semantic = parse_semantic_project(metadata)
    assert semantic.complete
    plan = build_device_link_plan(semantic.project)
    assert any(item["code"] == "InvalidArrayParameterDecay" for item in plan.diagnostics)


def test_device_memset_rejects_non_pointer_destination(tmp_path):
    target = tmp_path / "invalid_memset.cu"
    target.write_text(
        """__global__ void invalid(int* output) {
    int value = 0;
    memset(value, 0, 4);
    output[0] = value;
}
""",
        encoding="ascii",
    )
    metadata = parse_project([target])
    semantic = parse_semantic_project(metadata)
    assert semantic.complete
    plan = build_device_link_plan(semantic.project)
    assert any(item["code"] == "UnsupportedDeviceMemset" for item in plan.diagnostics)
