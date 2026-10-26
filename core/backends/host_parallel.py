"""Generate Metal and host dispatch only from proven original host loop records."""

import hashlib
import json

from core.ir.host_loops import HostLoopIR
from core.ir.models import KernelRouteIR
from core.backends.metal import translate_parameters
from core.passes.host_parallel import unwrap


def generate_loop(item, source_bytes):
    proof = item["proof"]
    if not (proof["eligible"] or proof["conditional"]):
        raise ValueError("Host loop has no complete or conditional proof")
    fields = {k: item[k] for k in HostLoopIR.__dataclass_fields__}
    loop = HostLoopIR.from_dict(fields)
    if hashlib.sha256(source_bytes).hexdigest() != loop.source_sha256:
        raise ValueError("Original source changed after host analysis")
    if not 0 <= loop.node.start < loop.node.end <= len(source_bytes):
        raise ValueError("Invalid original source range")
    original = source_bytes[loop.node.start : loop.node.end].decode("utf-8")
    # Clang for extents omit the trailing semicolon of an expression body.
    end = loop.node.end
    if loop.body.kind != "COMPOUND_STMT" and source_bytes[end : end + 1] == b";":
        original += ";"
        end += 1
    identity = json.dumps(
        [loop.function_usr, loop.translation_unit, loop.node.start], sort_keys=True
    )
    name = "mcu_host_" + hashlib.sha256(identity.encode()).hexdigest()[:12]
    prefix = name + "_"
    if prefix.encode() in source_bytes:
        raise ValueError("Generated host identifier collides with original source")
    objects = {}
    captures = {}
    for access in proof["accesses"]:
        origin = access["origin_usr"]
        if origin not in objects:
            objects[origin] = {
                "slot": len(objects),
                "name": access["origin_name"],
                "extent": access["origin_extent"],
                "read": False,
                "writes": set(),
            }
        capture = objects[origin]
        capture["read"] |= access["mode"] == "read"
        if access["mode"] == "write":
            capture["writes"].add(access["offset_elements"])
        captures[access["symbol_usr"]] = (capture["slot"], access["offset_elements"])
    iterator = loop.init.children[0].symbol_usr

    def expression(node):
        node = unwrap(node)
        if node.kind == "ARRAY_SUBSCRIPT_EXPR":
            base, index = map(unwrap, node.children)
            if index.symbol_usr != iterator:
                raise ValueError("Unproven index")
            slot, offset = captures[base.symbol_usr]
            return f"b{slot}[tid + {offset}u]"
        if node.kind == "DECL_REF_EXPR" and node.symbol_usr == iterator:
            return "tid"
        if node.kind == "INTEGER_LITERAL" and node.integer_value is not None:
            return f"{node.integer_value}u"
        if node.kind == "BINARY_OPERATOR" and node.operator in {"+", "-", "*", "&", "|", "^"}:
            return (
                f"({expression(node.children[0])} {node.operator} {expression(node.children[1])})"
            )
        raise ValueError("Unsupported proven expression")

    def statement(node):
        node = unwrap(node)
        if node.kind == "COMPOUND_STMT":
            return "\n".join(statement(c) for c in node.children)
        if node.kind == "BINARY_OPERATOR" and node.operator == "=":
            return f"{expression(node.children[0])} = {expression(node.children[1])};"
        raise ValueError("Unsupported proven statement")

    params = [f"unsigned int* b{v['slot']}" for v in objects.values()] + ["unsigned int count"]
    metal_params, bindings = translate_parameters(params)
    metal = "#include <metal_stdlib>\nusing namespace metal;\n"
    metal += (
        f"kernel void {name}({', '.join(metal_params)}, uint tid [[thread_position_in_grid]]) {{\n"
    )
    metal += "if (tid >= count) return;\n" + statement(loop.body) + "\n}\n"
    route = KernelRouteIR(
        name, bindings, {"coarsening_factor": 1}, {"file": loop.node.file, "line": loop.line}
    ).to_dict()
    bound = unwrap(loop.condition.children[1])
    count = bound.spelling if proof["conditional"] else str(proof["iterations"]) + "u"
    guards = (
        " && ".join(f"{prefix}count <= {g['maximum']}u" for g in proof["runtime_guards"]) or "true"
    )
    host = [
        "{",
        'static_assert(sizeof(unsigned) == 4, "Host loop requires 32-bit unsigned");',
        f"const unsigned {prefix}count = {count};",
        f"if ({prefix}count == 0) {{}} else if ({guards}) {{",
        f"bool {prefix}submitted = false;",
        "try {",
    ]
    # RAII frees resources even if a later operation throws. Failures never replay CPU.
    host += [
        f'auto {prefix}check = [](cudaError_t e) {{ if(e != cudaSuccess) throw std::runtime_error("Host loop GPU operation failed"); }};'
    ]
    for capture in objects.values():
        slot = capture["slot"]
        host += [
            f"struct {prefix}owner{slot} {{ unsigned* p=nullptr; ~{prefix}owner{slot}() {{ if(p) cudaFree(p); }} }} {prefix}b{slot};",
            f"{prefix}check(cudaMalloc((void**)&{prefix}b{slot}.p, {capture['extent']}ull * sizeof(unsigned)));",
        ]
        if capture["read"]:
            host += [
                f"{prefix}check(cudaMemcpy({prefix}b{slot}.p, {capture['name']}, {capture['extent']}ull * sizeof(unsigned), cudaMemcpyHostToDevice));"
            ]
    arguments = ", ".join(f"{prefix}b{c['slot']}.p" for c in objects.values()) + f", {prefix}count"
    host += [
        f"{prefix}submitted = true;",
        f'{prefix}check(::mcu::launch("{name}", dim3({prefix}count / 64u + ({prefix}count % 64u != 0)), dim3(64), {arguments}));',
        f"{prefix}check(cudaDeviceSynchronize());",
    ]
    for capture in objects.values():
        for offset in sorted(capture["writes"]):
            host += [
                f"{prefix}check(cudaMemcpy({capture['name']} + {offset}u, {prefix}b{capture['slot']}.p + {offset}u, static_cast<size_t>({prefix}count) * sizeof(unsigned), cudaMemcpyDeviceToHost));"
            ]
    host += [
        "} catch (const std::runtime_error&) {",
        f"if ({prefix}submitted) throw;",
        original,
        "}",
        "} else {",
        original,
        "}",
        "}",
    ]
    return {
        "name": name,
        "metal_source": metal,
        "route": route,
        "start": loop.node.start,
        "end": end,
        "host_replacement": "\n".join(host),
        "captures": [{**c, "writes": sorted(c["writes"])} for c in objects.values()],
    }
