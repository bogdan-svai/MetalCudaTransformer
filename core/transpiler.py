import json
import re


GEOMETRY_ARGUMENTS = [
    "uint3 thread_position_in_grid [[thread_position_in_grid]]",
    "uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]",
    "uint3 threads_per_threadgroup [[threads_per_threadgroup]]",
    "uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]]",
]

CUDA_TO_MSL_REPLACEMENTS = {
    "__syncthreads()": "threadgroup_barrier(mem_flags::mem_threadgroup)",
    "threadIdx.x": "thread_position_in_threadgroup.x",
    "threadIdx.y": "thread_position_in_threadgroup.y",
    "threadIdx.z": "thread_position_in_threadgroup.z",
    "blockIdx.x": "threadgroup_position_in_grid.x",
    "blockIdx.y": "threadgroup_position_in_grid.y",
    "blockIdx.z": "threadgroup_position_in_grid.z",
    "blockDim.x": "threads_per_threadgroup.x",
    "blockDim.y": "threads_per_threadgroup.y",
    "blockDim.z": "threads_per_threadgroup.z",
    "gridDim.x": "threadgroups_per_grid.x",
    "gridDim.y": "threadgroups_per_grid.y",
    "gridDim.z": "threadgroups_per_grid.z",
}


def transpile(metadata):
    kernels = metadata.get("kernels", [])
    metal_functions = []
    translated = {}

    for kernel in kernels:
        entry_point = f"{kernel['name']}_msl"
        parameters, bindings = translate_parameters(kernel.get("parameters", []))
        signature_parameters = parameters + GEOMETRY_ARGUMENTS
        body = mutate_body(kernel.get("body", ""))
        metal_functions.append(render_kernel(entry_point, signature_parameters, body))
        translated[kernel["name"]] = {
            "msl_entry_point": entry_point,
            "argument_count": len(bindings),
            "bindings": bindings,
        }

    metal_source = render_metal_source(metal_functions)
    routing_table = {"translated_kernels": translated}
    return {
        "metal_source": metal_source,
        "routing_table": routing_table,
        "routing_table_text": json.dumps(routing_table, indent=2, sort_keys=True),
    }


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
        bindings.append({
            "argument_index": index,
            "identifier": parsed["identifier"],
            "type": parsed["type"],
            "binding_target": binding_target,
            "space": space,
        })
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


def mutate_body(body):
    mutated = body
    for cuda_token, metal_token in CUDA_TO_MSL_REPLACEMENTS.items():
        mutated = mutated.replace(cuda_token, metal_token)
    return mutated.rstrip()


def render_kernel(entry_point, parameters, body):
    formatted_params = ",\n    ".join(parameters)
    body_text = indent_body(body)
    return f"kernel void {entry_point}(\n    {formatted_params}\n) {{{body_text}\n}}"


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


def render_metal_source(functions):
    header = [
        "#include <metal_stdlib>",
        "using namespace metal;",
        "",
    ]
    return "\n".join(header + functions) + "\n"
