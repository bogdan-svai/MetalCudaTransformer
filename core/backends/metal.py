import json
import re

from core.backends.device import (
    readonly_pointer_parameters,
    render_device_components,
    rewrite_kernel_calls,
)
from core.ir.models import BindingIR, KernelRouteIR
from core.ir.symbols import manifest_text
from core.passes.ir_optimization import (
    analyze_global_memory_accesses,
    analyze_loop_invariant_inputs,
    analyze_recurrent_arrays,
    analyze_scalarizable_arrays,
    analyze_static_loops,
)
from core.passes.metal_optimizer import optimize_metal_source
from core.registry.catalog import default_registry

GEOMETRY_ARGUMENTS = [
    "uint3 thread_position_in_grid [[thread_position_in_grid]]",
    "uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]",
    "uint3 threads_per_threadgroup [[threads_per_threadgroup]]",
    "uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]]",
    "uint3 threadgroups_per_grid [[threadgroups_per_grid]]",
    "uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]",
]

CUDA_TO_MSL_REPLACEMENTS = default_registry().token_replacements()


def transpile(metadata, coarsening_plans=None, semantic_project=None, link_plan=None):
    kernels = metadata.get("kernels", [])
    coarsening_plans = coarsening_plans or {}
    metal_functions = []
    translated = {}
    declarations = []
    helpers = []
    if semantic_project and link_plan:
        declarations, helpers = render_device_components(semantic_project, link_plan)

    for kernel in kernels:
        optimization = coarsening_plans.get(kernel["name"], default_optimization())
        entry_point = f"{kernel['name']}_msl"
        semantic_kernels = (
            semantic_project.functions_named(kernel["name"]) if semantic_project else []
        )
        semantic_kernel = next((item for item in semantic_kernels if item.kind == "KERNEL"), None)
        atomic_parameters = (
            link_plan.atomic_parameters.get(semantic_kernel.symbol_id, {})
            if semantic_kernel and link_plan
            else {}
        )
        source = {
            "file": kernel.get("file"),
            "line": kernel.get("body_line") or kernel.get("line"),
        }
        factor = optimization["coarsening_factor"]
        raw_body = semantic_kernel.body if semantic_kernel else kernel.get("body", "")
        if semantic_kernel and link_plan:
            raw_body = rewrite_kernel_calls(semantic_kernel.body, semantic_kernel, link_plan)
        atomic_locals = (
            link_plan.atomic_locals.get(semantic_kernel.symbol_id, {})
            if semantic_kernel and link_plan
            else {}
        )
        raw_body, shared_parameter, shared_descriptor = translate_shared_memory(
            raw_body,
            atomic_locals,
        )
        readonly_parameters = (
            readonly_pointer_parameters(
                semantic_kernel.parameters,
                raw_body,
                atomic_parameters,
                effects=semantic_kernel.effects,
            )
            if semantic_kernel
            else set()
        )
        parameters, bindings = translate_parameters(
            kernel.get("parameters", []),
            atomic_parameters=atomic_parameters,
            readonly_parameters=readonly_parameters,
        )
        signature_parameters = parameters + GEOMETRY_ARGUMENTS
        if shared_parameter:
            signature_parameters.append(shared_parameter)
        body = mutate_body(
            raw_body,
            coarsening_factor=factor,
            source=source if factor > 1 else None,
        )
        metal_functions.append(
            render_kernel(
                entry_point,
                signature_parameters,
                body,
                source if factor == 1 else None,
            )
        )
        route = KernelRouteIR(
            msl_entry_point=entry_point,
            bindings=bindings,
            optimization_metrics=optimization,
            source=source,
        ).to_dict()
        if shared_descriptor:
            route["threadgroup_memory"] = shared_descriptor
        translated[kernel["name"]] = route

    metal_source = render_metal_source(metal_functions, declarations=declarations, helpers=helpers)
    ir_analysis = (
        {
            "static_loops": analyze_static_loops(semantic_project),
            "scalarizable_arrays": analyze_scalarizable_arrays(semantic_project),
            "recurrent_arrays": analyze_recurrent_arrays(semantic_project),
            "global_memory_accesses": analyze_global_memory_accesses(semantic_project),
            "loop_invariant_inputs": analyze_loop_invariant_inputs(semantic_project),
        }
        if semantic_project
        else None
    )
    optimization_result = optimize_metal_source(metal_source, ir_analysis=ir_analysis)
    metal_source = optimization_result.source
    routing_table = {
        "schema_version": 2,
        "library_adapters": [],
        "metal_optimization": optimization_result.to_dict(),
        "translated_kernels": translated,
    }
    if ir_analysis:
        routing_table["ir_optimization"] = ir_analysis
    result = {
        "metal_source": metal_source,
        "optimization_report": optimization_result.to_dict(),
        "routing_table": routing_table,
        "routing_table_text": json.dumps(routing_table, indent=2, sort_keys=True),
    }
    if semantic_project and link_plan:
        manifest = link_plan.to_manifest(semantic_project)
        result["symbol_manifest"] = manifest
        result["symbol_manifest_text"] = manifest_text(manifest)
    return result


def translate_parameters(parameters, atomic_parameters=None, readonly_parameters=None):
    atomic_parameters = atomic_parameters or {}
    readonly_parameters = readonly_parameters or set()
    translated = []
    bindings = []
    for index, parameter in enumerate(parameters):
        parsed = parse_parameter(parameter)
        space = "device" if parsed["is_pointer"] else "constant"
        binding_target = f"buffer({index})"
        if parsed["is_pointer"]:
            atomic_type = atomic_parameters.get(parsed["identifier"], {}).get("type")
            parameter_type = f"{atomic_type}*" if atomic_type else parsed["type"]
            const_prefix = (
                "const "
                if not atomic_type
                and parsed["identifier"] in readonly_parameters
                and not parameter_type.strip().startswith("const ")
                else ""
            )
            msl_param = (
                f"{const_prefix}device {parameter_type} {parsed['identifier']} [[{binding_target}]]"
            )
        else:
            msl_param = f"constant {parsed['type']}& {parsed['identifier']} [[{binding_target}]]"
        translated.append(msl_param)
        bindings.append(
            BindingIR(
                argument_index=index,
                identifier=parsed["identifier"],
                type=parsed["type"],
                buffer_index=index,
                kind="buffer" if parsed["is_pointer"] else "scalar",
                space=space,
            )
        )
    return translated, bindings


def parse_parameter(parameter):
    text = re.sub(r"\s+", " ", parameter).strip()
    text = text.replace(" *", "*").replace("* ", "* ")
    match = re.match(r"(?P<type>.+?)(?P<identifier>[A-Za-z_]\w*)$", text)
    if not match:
        return {"type": text, "identifier": f"arg", "is_pointer": "*" in text}

    type_part = match.group("type").strip()
    identifier = match.group("identifier")
    if type_part.endswith("*"):
        type_name = type_part
    elif "*" in type_part:
        type_name = type_part.replace(" *", "*").strip()
    else:
        type_name = type_part
    return {
        "type": type_name,
        "identifier": identifier,
        "is_pointer": "*" in type_name,
    }


def translate_shared_memory(body, atomic_locals=None):
    atomic_locals = atomic_locals or {}
    dynamic = re.search(
        r"\bextern\s+__shared__\s+(?P<type>(?:(?:unsigned|signed)\s+)?[A-Za-z_]\w*)\s+"
        r"(?P<name>[A-Za-z_]\w*)\s*\[\s*\]\s*;",
        body,
    )
    parameter = None
    descriptor = None
    if dynamic:
        type_name = _map_kernel_type(dynamic.group("type"))
        name = dynamic.group("name")
        parameter_type = atomic_locals.get(name, {}).get("type", type_name)
        parameter = f"threadgroup {parameter_type}* {name} [[threadgroup(0)]]"
        descriptor = {
            "schema_version": 1,
            "slot": 0,
            "element_type": type_name,
            "alignment": _type_alignment(type_name),
        }
        body = body[: dynamic.start()] + body[dynamic.end() :]
    body = re.sub(r"\b__shared__\b", "threadgroup", body)
    for name, use in atomic_locals.items():
        atomic_type = use["type"]
        body = re.sub(
            rf"\bthreadgroup\s+(?:(?:unsigned|signed)\s+)?[A-Za-z_]\w*\s+"
            rf"(?P<name>{re.escape(name)})\b",
            rf"threadgroup {atomic_type} \g<name>",
            body,
        )
        body = re.sub(
            rf"(?<![=!<>])\b(?P<value>{re.escape(name)}(?:\s*\[[^]]+\])?)\s*=\s*(?P<rhs>[^;]+);",
            rf"atomic_store_explicit(&\g<value>, \g<rhs>, memory_order_relaxed);",
            body,
        )
        body = re.sub(
            rf"(?P<prefix>=\s*|\breturn\s+)(?P<value>{re.escape(name)}(?:\s*\[[^]]+\])?)",
            rf"\g<prefix>atomic_load_explicit(&\g<value>, memory_order_relaxed)",
            body,
        )
    return body, parameter, descriptor


def _map_kernel_type(type_name):
    mapped = type_name
    for pattern, replacement in [
        (r"\bunsigned\s+char\b", "uchar"),
        (r"\bunsigned\s+int\b", "uint"),
    ]:
        mapped = re.sub(pattern, replacement, mapped)
    return mapped


def _type_alignment(type_name):
    if type_name in {"char", "uchar"}:
        return 1
    if type_name in {"short", "ushort"}:
        return 2
    if type_name in {"double", "long", "ulong", "int64_t", "uint64_t"}:
        return 8
    return 4


def mutate_body(body, coarsening_factor=1, source=None):
    mutated = body
    replacements = dict(CUDA_TO_MSL_REPLACEMENTS)
    if coarsening_factor > 1:
        for axis in "xyz":
            replacements[f"threadIdx.{axis}"] = f"mcu_thread_idx.{axis}"
            replacements[f"blockDim.{axis}"] = f"mcu_block_dim.{axis}"

    for cuda_token, metal_token in replacements.items():
        mutated = mutated.replace(cuda_token, metal_token)
    mutated = mutated.rstrip()
    if coarsening_factor == 1:
        return mutated
    return render_coarsened_body(mutated, coarsening_factor, source=source)


def render_coarsened_body(body, factor, source=None):
    inner = indent_lines(body, 1)
    source_directive = render_line_directive(source)
    return "\n".join(
        [
            f"for (uint mcu_step = 0; mcu_step < {factor}; ++mcu_step) {{",
            "    uint3 mcu_thread_idx = uint3(",
            "        thread_position_in_threadgroup.x + mcu_step * threads_per_threadgroup.x,",
            "        thread_position_in_threadgroup.y,",
            "        thread_position_in_threadgroup.z",
            "    );",
            "    uint3 mcu_block_dim = uint3(",
            f"        threads_per_threadgroup.x * {factor},",
            "        threads_per_threadgroup.y,",
            "        threads_per_threadgroup.z",
            "    );",
            source_directive,
            inner,
            "}",
        ]
    )


def render_kernel(entry_point, parameters, body, source=None):
    formatted_params = ",\n    ".join(parameters)
    body_text = indent_body(body)
    source_directive = ""
    if source and source.get("file") and source.get("line"):
        source_directive = "\n" + render_line_directive(source)
    return (
        f"kernel void {entry_point}(\n    {formatted_params}\n) {{{source_directive}{body_text}\n}}"
    )


def render_line_directive(source):
    if not source or not source.get("file") or not source.get("line"):
        return ""
    filename = str(source["file"]).replace("\\", "\\\\").replace('"', '\\"')
    return f'#line {source["line"]} "{filename}"'


def indent_body(body):
    lines = body.splitlines()
    if not lines:
        return ""
    rendered = []
    for line in lines:
        stripped = line.rstrip()
        if stripped:
            rendered.append("    " + stripped.lstrip())
        else:
            rendered.append("")
    return "\n" + "\n".join(rendered)


def indent_lines(body, levels):
    prefix = "    " * levels
    return "\n".join(prefix + line if line.strip() else "" for line in body.splitlines())


def default_optimization():
    return {
        "mode": "none",
        "axis": "x",
        "coarsening_factor": 1,
        "original_block": None,
        "physical_block": None,
    }


def render_metal_source(functions, declarations=None, helpers=None):
    header = [
        "#include <metal_stdlib>",
        "using namespace metal;",
        "",
    ]
    sections = header
    if declarations:
        sections.extend(declarations + [""])
    if helpers:
        sections.extend(helpers + [""])
    sections.extend(functions)
    return "\n".join(sections) + "\n"
