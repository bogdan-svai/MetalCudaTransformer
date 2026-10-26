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
    return (
        semantic.project,
        plan,
        transpile(metadata, semantic_project=semantic.project, link_plan=plan),
    )


def test_persistent_semantic_cache_hits_and_invalidates_on_content_change(tmp_path, monkeypatch):
    monkeypatch.setenv("MCU_CACHE_DIR", str(tmp_path / "cache"))
    target = tmp_path / "cached.cu"
    target.write_text("__global__ void first(int* out) { out[0] = 1; }", encoding="ascii")
    metadata = parse_project([target])
    cold = parse_semantic_project(metadata)
    warm = parse_semantic_project(metadata)
    assert cold.cached is False
    assert warm.cached is True
    assert warm.project.functions_named("first")

    target.write_text("__global__ void second(int* out) { out[0] = 2; }", encoding="ascii")
    changed = parse_semantic_project(parse_project([target]))
    assert changed.cached is False
    assert changed.project.functions_named("second")


def test_multifile_semantic_cache_reparses_only_changed_unit(tmp_path, monkeypatch):
    import core.frontend.semantic as semantic

    monkeypatch.setenv("MCU_CACHE_DIR", str(tmp_path / "cache"))
    first = tmp_path / "first.cu"
    second = tmp_path / "second.cu"
    first.write_text("__global__ void first(int* out) { out[0] = 1; }\n", encoding="ascii")
    second.write_text("__global__ void second(int* out) { out[0] = 2; }\n", encoding="ascii")
    metadata = parse_project([first, second])
    original = semantic._collect_project
    calls = []

    def counted(*args, **kwargs):
        calls.append(args[2][0].name)
        return original(*args, **kwargs)

    monkeypatch.setattr(semantic, "_collect_project", counted)
    cold = parse_semantic_project(metadata)
    assert cold.cached is False
    assert len(calls) == 2

    calls.clear()
    warm = parse_semantic_project(metadata)
    assert warm.cached is True
    assert calls == []

    first.write_text("__global__ void first(int* out) { out[0] = 3; }\n", encoding="ascii")
    changed = parse_semantic_project(parse_project([first, second]))
    assert changed.cached is False
    assert calls == [first.name]


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


def test_readonly_literal_arrays_are_marked_const_in_metal(tmp_path):
    _project, _plan, generated = _semantic_build(
        tmp_path,
        """__device__ unsigned int pick(int index) {
    unsigned int k[4] = {1, 2, 3, 4};
    unsigned int m[4] = {0, 0, 0, 0};
    m[index & 3] = k[index & 3];
    return m[index & 3];
}
__global__ void kernel(unsigned int* output) { output[0] = pick(2); }
""",
    )

    metal = generated["metal_source"]
    assert "const unsigned int k[4] = {1, 2, 3, 4};" in metal
    assert "unsigned int m[4] = {0, 0, 0, 0};" in metal


def test_large_literal_constant_arrays_are_promoted_and_deduplicated(tmp_path):
    values = ", ".join(str(index) for index in range(16))
    _project, _plan, generated = _semantic_build(
        tmp_path,
        f"""__device__ unsigned int first(int index) {{
    const unsigned int table[16] = {{{values}}};
    return table[index & 15];
}}
__device__ unsigned int second(int index) {{
    const unsigned int values[16] = {{{values}}};
    return values[index & 15];
}}
__global__ void kernel(unsigned int* output) {{
    output[0] = first(3) + second(4);
}}
""",
    )

    metal = generated["metal_source"]
    assert "const unsigned int table[16]" not in metal
    assert "const unsigned int values[16]" not in metal
    assert metal.count("constant unsigned int mcu_const_") == 1
    promotion = next(
        item
        for item in generated["optimization_report"]["passes"]
        if item["name"] == "promote_literal_constant_arrays"
    )
    assert promotion["replacements"] == 2
    assert promotion["details"]["unique_tables"] == 1
    assert promotion["details"]["local_bytes_removed"] == 128


def test_readonly_nonconst_literal_table_is_promoted(tmp_path):
    values = ", ".join(str(index) for index in range(64))
    _project, _plan, generated = _semantic_build(
        tmp_path,
        f"""__global__ void kernel(unsigned int* output, int index) {{
    unsigned int table[64] = {{{values}}};
    output[0] = table[index & 63];
}}
""",
    )

    metal = generated["metal_source"]
    assert "unsigned int table[64]" not in metal
    assert "constant unsigned int mcu_const_" in metal
    promotion = next(
        item
        for item in generated["optimization_report"]["passes"]
        if item["name"] == "promote_literal_constant_arrays"
    )
    assert promotion["replacements"] == 1
    assert promotion["details"]["local_bytes_removed"] == 256


def test_promoted_constant_is_emitted_after_its_type_alias(tmp_path):
    values = ", ".join(str(index) + "u" for index in range(16))
    _project, _plan, generated = _semantic_build(
        tmp_path,
        f"""typedef unsigned int WORD;
__device__ WORD pick(int index) {{
    const WORD table[16] = {{{values}}};
    return table[index & 15];
}}
__global__ void kernel(WORD* output) {{ output[0] = pick(3); }}
""",
    )

    metal = generated["metal_source"]
    constant_position = metal.index("constant WORD mcu_const_")
    assert metal.index("typedef uint WORD;") < constant_position
    assert constant_position < metal.index("inline ", constant_position)


def test_promoted_constant_does_not_split_template_from_function():
    from core.passes.metal_optimizer import promote_literal_constant_arrays

    values = ", ".join(str(index) for index in range(16))
    source = f"""typedef uint WORD;
template <typename T>
inline thread T* helper(thread T* value) {{ return value; }}
inline WORD pick(int index) {{
    WORD table[16] = {{{values}}};
    return table[index & 15];
}}
"""

    metal, report = promote_literal_constant_arrays(source)

    assert report.replacements == 1
    assert metal.index("constant WORD mcu_const_") < metal.index("template <typename T>")
    assert "template <typename T>\ninline thread T* helper" in metal


def test_literal_constant_promotion_rejects_small_and_symbolic_tables(tmp_path):
    _project, _plan, generated = _semantic_build(
        tmp_path,
        """#define VALUE 7
__global__ void kernel(unsigned int* output, int index) {
    const unsigned int small[4] = {1, 2, 3, 4};
    const unsigned int symbolic[16] = {
        VALUE, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    };
    output[0] = small[index & 3] + symbolic[index & 15];
}
""",
    )

    metal = generated["metal_source"]
    assert "const unsigned int small[4]" in metal
    assert "const unsigned int symbolic[16]" in metal
    promotion = next(
        item
        for item in generated["optimization_report"]["passes"]
        if item["name"] == "promote_literal_constant_arrays"
    )
    assert promotion["replacements"] == 0


def test_rotate_macros_use_metal_intrinsics(tmp_path):
    _project, _plan, generated = _semantic_build(
        tmp_path,
        """#define ROTRIGHT(a,b) (((a) >> (b)) | ((a) << (32-(b))))
__device__ unsigned int rotate_word(unsigned int value) {
    return ROTRIGHT(value, 7);
}
__global__ void kernel(unsigned int* output) { output[0] = rotate_word(5); }
""",
    )

    metal = generated["metal_source"]
    assert "#define ROTRIGHT(a,b) rotate((a), uint(32 - (b)))" in metal


def test_small_static_loops_are_unrolled_in_metal(tmp_path):
    _project, _plan, generated = _semantic_build(
        tmp_path,
        """__device__ int sum4() {
    int total = 0;
    for (int i = 0; i < 4; ++i) {
        total += i;
    }
    return total;
}
__global__ void kernel(int* output) { output[0] = sum4(); }
""",
    )

    metal = generated["metal_source"]
    assert "for (int i = 0; i < 4; ++i)" not in metal
    assert "total += 0;" in metal
    assert "total += 3;" in metal
    report = generated["optimization_report"]
    unroll = next(item for item in report["passes"] if item["name"] == "unroll_small_static_loops")
    assert unroll["replacements"] == 1
    assert unroll["details"]["unrolled_iterations"] == 4


def test_static_loop_unroll_limit_can_be_tuned_for_benchmarking(tmp_path, monkeypatch):
    monkeypatch.setenv("MCU_METAL_UNROLL_MAX", "16")
    _project, _plan, generated = _semantic_build(
        tmp_path,
        """__global__ void kernel(int* output) {
    for (int i = 0; i < 16; ++i) {
        output[i] = i;
    }
}
""",
    )

    metal = generated["metal_source"]
    assert "for (int i = 0; i < 16; ++i)" not in metal
    unroll = next(
        item
        for item in generated["optimization_report"]["passes"]
        if item["name"] == "unroll_small_static_loops"
    )
    assert unroll["details"]["max_iterations"] == 16
    assert unroll["details"]["unrolled_iterations"] == 16


def test_metal_optimization_report_includes_local_memory_pressure(tmp_path):
    _project, _plan, generated = _semantic_build(
        tmp_path,
        """__global__ void kernel(int* output) {
    int value, local[8];
    local[threadIdx.x & 7] = 1;
    output[threadIdx.x] = local[threadIdx.x & 7];
}
""",
    )

    report = generated["optimization_report"]
    local_memory = report["analysis"]["local_memory"]
    assert local_memory["max_estimated_bytes_per_thread"] >= 32
    assert local_memory["functions"][0]["function"] == "kernel_msl"
    assert local_memory["functions"][0]["arrays"][0]["name"] == "local"
    assert generated["routing_table"]["metal_optimization"] == report


def test_small_constant_index_arrays_are_scalarized_after_unroll(tmp_path):
    _project, _plan, generated = _semantic_build(
        tmp_path,
        """__global__ void kernel(int* output) {
    int local[4];
    for (int i = 0; i < 4; ++i) {
        local[i] = i + 1;
    }
    output[0] = local[0] + local[3];
}
""",
    )

    metal = generated["metal_source"]
    assert "int local[4]" not in metal
    assert "int local_0;" in metal
    assert "local_0 = 0 + 1;" in metal
    assert "output[0] = local_0 + local_3;" in metal
    scalarize = next(
        item
        for item in generated["optimization_report"]["passes"]
        if item["name"] == "scalarize_small_constant_index_arrays"
    )
    assert scalarize["replacements"] == 1
    assert scalarize["details"]["scalar_variables"] == 4


def test_dynamic_index_arrays_are_not_scalarized(tmp_path):
    _project, _plan, generated = _semantic_build(
        tmp_path,
        """__global__ void kernel(int* output, int index) {
    int local[4];
    local[index & 3] = 7;
    output[0] = local[index & 3];
}
""",
    )

    metal = generated["metal_source"]
    assert "int local[4]" in metal
    scalarize = next(
        item
        for item in generated["optimization_report"]["passes"]
        if item["name"] == "scalarize_small_constant_index_arrays"
    )
    assert scalarize["replacements"] == 0


def test_scalarization_uses_ir_array_identity_per_function(tmp_path):
    _project, _plan, generated = _semantic_build(
        tmp_path,
        """__global__ void kernel(int* output, int index) {
    int approved[4];
    int dynamic[4];
    approved[0] = 1;
    approved[3] = 4;
    dynamic[index & 3] = 7;
    output[0] = approved[0] + approved[3] + dynamic[index & 3];
}
""",
    )

    metal = generated["metal_source"]
    assert "int approved[4]" not in metal
    assert "int dynamic[4]" in metal
    scalarize = next(
        item
        for item in generated["optimization_report"]["passes"]
        if item["name"] == "scalarize_small_constant_index_arrays"
    )
    assert scalarize["replacements"] == 1


def test_dynamic_recurrence_arrays_are_reported_as_sliding_window_candidates(tmp_path, monkeypatch):
    monkeypatch.setenv("MCU_METAL_RECURRENCE_FUSION", "0")
    _project, _plan, generated = _semantic_build(
        tmp_path,
        """__global__ void kernel(unsigned int* output) {
    unsigned int buffer[32];
    for (unsigned int i = 16; i < 32; ++i) {
        buffer[i] = buffer[i - 2] + buffer[i - 7] + buffer[i - 16];
    }
    for (unsigned int i = 0; i < 32; ++i) {
        output[i] = buffer[i];
    }
}
""",
    )

    candidates = generated["optimization_report"]["analysis"]["dynamic_array_candidates"][
        "candidates"
    ]
    buffer = next(item for item in candidates if item["array"] == "buffer")
    assert buffer["recurrence"]["index_variable"] == "i"
    assert buffer["recurrence"]["distances"] == [2, 7, 16]
    assert buffer["sliding_window_elements"] == 16
    assert buffer["sliding_window_bytes_saved"] == 64
    assert buffer["requires_loop_fusion"] is True


def test_register_pressure_analysis_exposes_general_cost_signal(tmp_path, monkeypatch):
    monkeypatch.setenv("MCU_METAL_RECURRENCE_FUSION", "0")
    _project, _plan, generated = _semantic_build(
        tmp_path,
        """__global__ void kernel(int* output, int index) {
    int values[128];
    for (int i = 0; i < 128; ++i) values[i] = i + index;
    output[index & 3] = values[index & 127];
}
""",
    )

    functions = generated["optimization_report"]["analysis"]["register_pressure"]["functions"]
    summary = next(item for item in functions if item["function"] == "kernel_msl")
    assert summary["estimated_local_bytes"] == 512
    assert summary["dynamic_array_accesses"] >= 2
    assert summary["risk"] == "high"


def test_predicate_materialization_candidates_are_reported_generically(tmp_path, monkeypatch):
    from core.passes.metal_optimizer import analyze_predicate_materialization

    source = """inline void kernel_msl() {
    int value[32];
    produce(value);
    if (accept(value)) output[0] = 1;
}
"""
    candidates = analyze_predicate_materialization(source)["candidates"]
    assert any(item["buffer"] == "value" and item["producer"] == "produce" for item in candidates)


def test_metal_optimization_analysis_reuses_final_function_scan():
    from core.passes.metal_optimizer import (
        analyze_dynamic_array_candidates,
        analyze_predicate_materialization,
        analyze_register_pressure,
        estimate_local_memory_pressure,
        optimize_metal_source,
    )

    source = """inline void kernel_msl() {
    int values[32];
    produce(values);
    if (accept(values)) values[0] = 1;
}
"""
    result = optimize_metal_source(source)
    assert result.analysis["local_memory"] == estimate_local_memory_pressure(result.source)
    assert result.analysis["dynamic_array_candidates"] == analyze_dynamic_array_candidates(
        result.source
    )
    assert result.analysis["register_pressure"] == analyze_register_pressure(result.source)
    assert result.analysis["predicate_materialization"] == analyze_predicate_materialization(
        result.source
    )


def test_semantic_ir_collects_ast_effect_summary(tmp_path):
    project, _plan, _generated = _semantic_build(
        tmp_path,
        """__device__ int pure_add(int a, int b) { return a + b; }
__device__ void write_value(int* output) { output[0] = 7; }
__global__ void kernel(int* output) {
    output[0] = pure_add(2, 3);
    write_value(output);
}
""",
    )

    pure = next(item for item in project.functions.values() if item.name == "pure_add")
    writer = next(item for item in project.functions.values() if item.name == "write_value")
    assert pure.effects["pure"] is True
    assert writer.effects["writes_memory"] is True
    assert writer.effects["writes_parameters"] == ["output"]
    assert (
        "write_value"
        in next(item for item in project.functions.values() if item.name == "kernel").effects[
            "calls"
        ]
    )


def test_semantic_ir_propagates_effects_through_call_graph(tmp_path):
    project, _plan, _generated = _semantic_build(
        tmp_path,
        """__device__ void write_value(int* output) { output[0] = 7; }
__device__ void wrapper(int* output) { write_value(output); }
__global__ void kernel(int* output) { wrapper(output); }
""",
    )

    wrapper = next(item for item in project.functions.values() if item.name == "wrapper")
    kernel = next(item for item in project.functions.values() if item.name == "kernel")
    assert wrapper.effects["writes_memory"] is True
    assert wrapper.effects["pure"] is False
    assert "write_value" in wrapper.effects["propagated_impure_calls"]
    assert wrapper.effects["writes_parameters"] == ["output"]
    assert kernel.effects["pure"] is False
    assert "wrapper" in kernel.effects["propagated_impure_calls"]
    assert kernel.effects["writes_parameters"] == ["output"]


def test_semantic_ir_retains_typed_statement_structure(tmp_path):
    project, _plan, _generated = _semantic_build(
        tmp_path,
        """__global__ void kernel(int* output) {
    for (int i = 0; i < 4; ++i) {
        if (i > 1) output[i] = i;
    }
}
""",
    )

    kernel = next(item for item in project.functions.values() if item.name == "kernel")
    kinds = [item.kind for item in kernel.statements]
    assert "FOR_STMT" in kinds
    assert "IF_STMT" in kinds
    loop = next(item for item in kernel.statements if item.kind == "FOR_STMT")
    assert loop.source.line is not None
    assert "i < 4" in loop.text
    assert loop.attributes["parent_index"] is not None
    assert loop.attributes["child_indices"]
    assert loop.attributes["cfg_successors"]
    assert loop.attributes["cfg_back_edges"]


def test_semantic_ir_retains_branch_and_switch_cfg_edges(tmp_path):
    project, _plan, _generated = _semantic_build(
        tmp_path,
        """__global__ void kernel(int* output, int mode) {
    for (int i = 0; i < 2; ++i) {
        if (mode) output[i] = i;
        else output[i] = i + 1;
        switch (mode) { case 0: output[0] = 3; break; default: output[1] = 4; }
    }
}
""",
    )
    kernel = next(item for item in project.functions.values() if item.name == "kernel")
    by_kind = {item.kind: item for item in kernel.statements}
    assert by_kind["IF_STMT"].attributes["cfg_successors"]
    assert by_kind["SWITCH_STMT"].attributes["cfg_successors"]
    assert by_kind["FOR_STMT"].attributes["cfg_back_edges"]


def test_semantic_ir_tracks_alias_when_helper_writes_through_local_pointer(tmp_path):
    project, _plan, _generated = _semantic_build(
        tmp_path,
        """__device__ void store(int* output, int value) { output[0] = value; }
__device__ void wrapper(int* output) { int* alias = output; store(alias, 9); }
__global__ void kernel(int* output) { wrapper(output); }
""",
    )

    wrapper = next(item for item in project.functions.values() if item.name == "wrapper")
    # The call target and the local pointer alias are both resolved from AST.
    assert "store" in wrapper.effects["propagated_impure_calls"]
    assert wrapper.effects["pointer_aliases"]["alias"] == "output"
    assert "output" in wrapper.effects["writes_argument_symbols"]
    assert wrapper.effects["writes_memory"] is True


def test_unbraced_recurrence_loop_is_fused_into_consumer_loop(tmp_path, monkeypatch):
    monkeypatch.setenv("MCU_METAL_RECURRENCE_FUSION", "1")
    _project, _plan, generated = _semantic_build(
        tmp_path,
        """__global__ void kernel(unsigned int* output) {
    unsigned int value, i, j, buffer[32];
    for (i = 0, j = 0; i < 16; ++i, j += 1)
    buffer[i] = j;
    for ( ; i < 32; ++i)
    buffer[i] = buffer[i - 2] + buffer[i - 7] + buffer[i - 16];
    for (i = 0; i < 32; ++i) {
    value = buffer[i];
    output[i] = value;
    }
}
""",
    )

    metal = generated["metal_source"]
    assert "buffer[16]" in metal
    assert "buffer[32]" not in metal
    assert "if (i >= 16)" in metal
    assert "buffer[i & 15] = buffer[(i - 2) & 15]" in metal
    assert "value = buffer[i & 15];" in metal
    fusion = next(
        item
        for item in generated["optimization_report"]["passes"]
        if item["name"] == "fuse_recurrent_array_consumers"
    )
    assert fusion["replacements"] == 1
    assert fusion["details"]["bytes_saved"] == 64


def test_braced_recurrence_with_temporaries_is_fused_into_typed_consumer(tmp_path, monkeypatch):
    monkeypatch.setenv("MCU_METAL_RECURRENCE_FUSION", "1")
    _project, _plan, generated = _semantic_build(
        tmp_path,
        """__global__ void kernel(unsigned int* output) {
    unsigned int buffer[32];
    for (unsigned int i = 0; i < 16; ++i) {
        buffer[i] = i;
    }
    for (unsigned int i = 16; i < 32; ++i) {
        unsigned int x = buffer[i - 15];
        unsigned int y = buffer[i - 2];
        buffer[i] = buffer[i - 16] + x + buffer[i - 7] + y;
    }
    unsigned int accumulator = 7;
    for (unsigned int i = 0; i < 32; ++i) {
        accumulator += buffer[i];
        output[i] = accumulator;
    }
}
""",
    )

    metal = generated["metal_source"]
    assert "buffer[16]" in metal
    assert "buffer[32]" not in metal
    assert "if (i >= 16)" in metal
    assert "unsigned int x = buffer[(i - 15) & 15];" in metal
    assert "buffer[i & 15] = buffer[(i - 16) & 15]" in metal
    assert "unsigned int accumulator = 7;" in metal
    assert "for (unsigned int i = 16; i < 32; ++i)" not in metal
    fusion = next(
        item
        for item in generated["optimization_report"]["passes"]
        if item["name"] == "fuse_recurrent_array_consumers"
    )
    assert fusion["replacements"] == 1
    assert fusion["details"]["bytes_saved"] == 64


def test_braced_recurrence_with_unsafe_array_index_is_not_fused(tmp_path, monkeypatch):
    monkeypatch.setenv("MCU_METAL_RECURRENCE_FUSION", "1")
    _project, _plan, generated = _semantic_build(
        tmp_path,
        """__global__ void kernel(unsigned int* output, unsigned int other) {
    unsigned int buffer[32];
    for (unsigned int i = 16; i < 32; ++i) {
        unsigned int x = buffer[other];
        buffer[i] = buffer[i - 16] + x;
    }
    for (unsigned int i = 0; i < 32; ++i) {
        output[i] = buffer[i];
    }
}
""",
    )

    metal = generated["metal_source"]
    assert "buffer[32]" in metal
    fusion = next(
        item
        for item in generated["optimization_report"]["passes"]
        if item["name"] == "fuse_recurrent_array_consumers"
    )
    assert fusion["replacements"] == 0


def test_readonly_pointer_parameters_are_const_in_metal(tmp_path):
    _project, _plan, generated = _semantic_build(
        tmp_path,
        """__global__ void kernel(int* input, int* output) {
    int idx = threadIdx.x;
    output[idx] = input[idx] + 1;
}
""",
    )

    metal = generated["metal_source"]
    assert "const device int* input [[buffer(0)]]" in metal
    assert "device int* output [[buffer(1)]]" in metal


def test_readonly_device_helper_pointer_parameters_are_const_in_metal(tmp_path):
    _project, _plan, generated = _semantic_build(
        tmp_path,
        """__device__ int pick(int* input, int index) {
    return input[index];
}
__global__ void kernel(int* input, int* output) {
    int idx = threadIdx.x;
    output[idx] = pick(input, idx);
}
""",
    )

    metal = generated["metal_source"]
    assert "const device int* input" in metal
    assert "device int* input [[buffer(0)]]" in metal


def test_pointer_parameters_with_field_writes_stay_mutable(tmp_path):
    _project, _plan, generated = _semantic_build(
        tmp_path,
        """struct Box { int value; };
__device__ void reset(Box* box) {
    box->value = 0;
}
__global__ void kernel(Box* box) { reset(box); }
""",
    )

    metal = generated["metal_source"]
    assert "inline void reset" in metal
    assert "device Box* box" in metal
    assert "const device Box* box" not in metal


def test_pointer_parameters_with_mutable_derived_pointers_stay_mutable(tmp_path):
    _project, _plan, generated = _semantic_build(
        tmp_path,
        """__global__ void kernel(int* input, int* output) {
    int idx = threadIdx.x;
    int* current = input + idx;
    output[idx] = *current;
}
""",
    )

    metal = generated["metal_source"]
    assert "device int* input [[buffer(0)]]" in metal


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
