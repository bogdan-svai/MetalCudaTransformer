import json
import re

from core.ir.models import BindingIR, KernelRouteIR
from core.ir.symbols import manifest_text
from core.backends.device import render_device_components, rewrite_kernel_calls
from core.registry.catalog import default_registry


GEOMETRY_ARGUMENTS = [
    "uint3 thread_position_in_grid [[thread_position_in_grid]]",
    "uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]",
    "uint3 threads_per_threadgroup [[threads_per_threadgroup]]",
    "uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]]",
    "uint3 threadgroups_per_grid [[threadgroups_per_grid]]",
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
        parameters, bindings = translate_parameters(kernel.get("parameters", []))
        signature_parameters = parameters + GEOMETRY_ARGUMENTS
        source = {
            "file": kernel.get("file"),
            "line": kernel.get("body_line") or kernel.get("line"),
        }
        factor = optimization["coarsening_factor"]
        raw_body = kernel.get("body", "")
        semantic_kernels = semantic_project.functions_named(kernel["name"]) if semantic_project else []
        semantic_kernel = next((item for item in semantic_kernels if item.kind == "KERNEL"), None)
        if semantic_kernel and link_plan:
            raw_body = rewrite_kernel_calls(raw_body, semantic_kernel, link_plan)
        body = mutate_body(
            raw_body,
            coarsening_factor=factor,
            source=source if factor > 1 else None,
        )
        metal_functions.append(render_kernel(
            entry_point,
            signature_parameters,
            body,
            source if factor == 1 else None,
        ))
        translated[kernel["name"]] = KernelRouteIR(
            msl_entry_point=entry_point,
            bindings=bindings,
            optimization_metrics=optimization,
            source=source,
        ).to_dict()

    metal_source = render_metal_source(metal_functions, declarations=declarations, helpers=helpers)
    routing_table = {
        "schema_version": 1,
        "translated_kernels": translated,
    }
    result = {
        "metal_source": metal_source,
        "routing_table": routing_table,
        "routing_table_text": json.dumps(routing_table, indent=2, sort_keys=True),
    }
    if semantic_project and link_plan:
        manifest = link_plan.to_manifest(semantic_project)
        result["symbol_manifest"] = manifest
        result["symbol_manifest_text"] = manifest_text(manifest)
    return result


def translate_parameters(parameters):
    translated = []
    bindings = []
    for index, parameter in enumerate(parameters):
        parsed = parse_parameter(parameter)
        space = "device" if parsed["is_pointer"] else "constant"
        binding_target = f"buffer({index})"
        if parsed["is_pointer"]:
            msl_param = f"device {parsed['type']} {parsed['identifier']} [[{binding_target}]]"
        else:
            msl_param = f"constant {parsed['type']}& {parsed['identifier']} [[{binding_target}]]"
        translated.append(msl_param)
        bindings.append(BindingIR(
            argument_index=index,
            identifier=parsed["identifier"],
            type=parsed["type"],
            buffer_index=index,
            kind="buffer" if parsed["is_pointer"] else "scalar",
            space=space,
        ))
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
    return "\n".join([
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
    ])


def render_kernel(entry_point, parameters, body, source=None):
    formatted_params = ",\n    ".join(parameters)
    body_text = indent_body(body)
    source_directive = ""
    if source and source.get("file") and source.get("line"):
        source_directive = "\n" + render_line_directive(source)
    return f"kernel void {entry_point}(\n    {formatted_params}\n) {{{source_directive}{body_text}\n}}"


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
