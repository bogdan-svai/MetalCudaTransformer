import re

from core.ir.link import DeviceLinkPlan
from core.ir.symbols import FunctionVariantIR, generated_symbol_name
from core.registry.catalog import default_registry


def build_device_link_plan(project, registry=None):
    registry = registry or default_registry()
    diagnostics = []
    resolved_calls = {}
    intrinsics = {}

    device_functions = {
        symbol_id: function
        for symbol_id, function in project.functions.items()
        if function.is_device_callable
    }
    kernels = sorted(
        symbol_id for symbol_id, function in device_functions.items()
        if function.kind == "KERNEL"
    )

    for symbol_id, function in device_functions.items():
        if function.kind != "KERNEL" and re.search(
            r"\b(?:threadIdx|blockIdx|blockDim|gridDim)\b",
            function.body,
        ):
            diagnostics.append({
                "code": "DeviceHelperUsesGeometry",
                "severity": "error",
                "message": (
                    f"Device helper '{function.name}' uses implicit CUDA geometry; "
                    "hidden geometry parameters are not supported in Stage 6."
                ),
                "file": function.source.file,
                "line": function.source.line,
                "column": function.source.column,
            })

    for symbol_id, function in device_functions.items():
        for call in function.calls:
            resolution = _resolve_call(call, project, registry, diagnostics)
            if resolution is None:
                continue
            if isinstance(resolution, str):
                resolved_calls[_call_key(call)] = resolution
            else:
                intrinsics[call.callee_name] = resolution.to_dict()

    _detect_cycles(kernels, device_functions, resolved_calls, diagnostics)
    if _has_errors(diagnostics):
        return DeviceLinkPlan(kernels, [], {}, intrinsics, diagnostics)

    ordered = []
    seen_variants = set()
    call_targets = {}

    def visit(symbol_id, spaces):
        function = device_functions[symbol_id]
        variant_key = (symbol_id, spaces)
        if function.kind != "KERNEL" and variant_key in seen_variants:
            return
        if function.kind != "KERNEL":
            seen_variants.add(variant_key)

        for call in function.calls:
            callee_id = resolved_calls.get(_call_key(call))
            if not callee_id or callee_id not in device_functions:
                continue
            callee = device_functions[callee_id]
            callee_spaces = _infer_callee_spaces(function, spaces, callee, call, diagnostics)
            if callee_spaces is None:
                continue
            visit(callee_id, callee_spaces)
            target = generated_symbol_name(callee_id, callee_spaces)
            call_targets[(
                symbol_id,
                spaces,
                call.callee_name,
                call.source.file,
                call.source.line,
                call.source.column,
            )] = target

        if function.kind != "KERNEL":
            ordered.append(FunctionVariantIR(
                symbol_id=symbol_id,
                generated_name=generated_symbol_name(symbol_id, spaces),
                parameter_address_spaces=spaces,
            ))

    for kernel_id in kernels:
        kernel = device_functions[kernel_id]
        root_spaces = tuple(
            "device" if parameter.type.pointer_depth else None
            for parameter in kernel.parameters
        )
        visit(kernel_id, root_spaces)

    return DeviceLinkPlan(kernels, ordered, call_targets, intrinsics, diagnostics)


def _resolve_call(call, project, registry, diagnostics):
    if call.callee_name in project.types:
        return None
    if call.callee_id and call.callee_id in project.functions:
        callee = project.functions[call.callee_id]
        if callee.is_device_callable:
            return call.callee_id
        return _diagnose(
            diagnostics,
            "MissingDeviceDefinition",
            f"Function '{call.callee_name}' is not device-callable.",
            call,
        )

    candidates = [
        function for function in project.functions_named(call.callee_name)
        if function.is_device_callable and len(function.parameters) == len(call.arguments)
    ]
    if len(candidates) == 1:
        return candidates[0].symbol_id
    if len(candidates) > 1:
        return _diagnose(
            diagnostics,
            "AmbiguousDeviceOverload",
            f"Call to '{call.callee_name}' matches multiple device functions.",
            call,
        )

    rule = registry.resolve(call.callee_name, len(call.arguments))
    if rule is False:
        return _diagnose(
            diagnostics,
            "UnsupportedIntrinsicSignature",
            f"Intrinsic '{call.callee_name}' does not support {len(call.arguments)} arguments.",
            call,
        )
    if rule:
        return rule

    if call.callee_name.startswith("operator"):
        return None
    return _diagnose(
        diagnostics,
        "MissingDeviceDefinition",
        f"No device definition or translation rule exists for '{call.callee_name}'.",
        call,
    )


def _detect_cycles(roots, functions, resolved_calls, diagnostics):
    visiting = set()
    visited = set()

    def walk(symbol_id):
        if symbol_id in visiting:
            function = functions[symbol_id]
            diagnostics.append({
                "code": "RecursiveDeviceCall",
                "severity": "error",
                "message": f"Recursive device call cycle includes '{function.name}'.",
                "file": function.source.file,
                "line": function.source.line,
                "column": function.source.column,
            })
            return
        if symbol_id in visited:
            return
        visiting.add(symbol_id)
        for call in functions[symbol_id].calls:
            target = resolved_calls.get(_call_key(call))
            if target in functions:
                walk(target)
        visiting.remove(symbol_id)
        visited.add(symbol_id)

    for root in roots:
        walk(root)


def _infer_callee_spaces(caller, caller_spaces, callee, call, diagnostics):
    caller_space_by_parameter = {
        parameter.name: caller_spaces[index]
        for index, parameter in enumerate(caller.parameters)
    }
    local_names = _local_variables(caller.body)
    spaces = []
    for index, parameter in enumerate(callee.parameters):
        if not parameter.type.pointer_depth:
            spaces.append(None)
            continue
        expression = call.arguments[index] if index < len(call.arguments) else ""
        matched = [
            space for name, space in caller_space_by_parameter.items()
            if space and re.search(rf"\b{re.escape(name)}\b", expression)
        ]
        if matched:
            spaces.append(matched[0])
            continue
        if any(re.search(rf"\b{re.escape(name)}\b", expression) for name in local_names):
            spaces.append("thread")
            continue
        diagnostics.append({
            "code": "AmbiguousAddressSpace",
            "severity": "error",
            "message": (
                f"Cannot infer address space for argument {index} of "
                f"'{callee.name}' from expression '{expression}'."
            ),
            "file": call.source.file,
            "line": call.source.line,
            "column": call.source.column,
        })
        return None
    return tuple(spaces)


def _local_variables(body):
    pattern = re.compile(
        r"(?:^|[;{}])\s*(?:const\s+)?[A-Za-z_]\w*(?:\s*<[^;>]+>)?"
        r"(?:\s+|\s*\*\s*)([A-Za-z_]\w*)\s*(?:\[[^]]+\])?\s*(?:[=;,{])",
        re.MULTILINE,
    )
    return {match.group(1) for match in pattern.finditer(body)}


def _call_key(call):
    return call.caller_id, call.callee_name, call.source.file, call.source.line, call.source.column


def _diagnose(diagnostics, code, message, call):
    diagnostics.append({
        "code": code,
        "severity": "error",
        "message": message,
        "file": call.source.file,
        "line": call.source.line,
        "column": call.source.column,
    })
    return None


def _has_errors(diagnostics):
    return any(item["severity"] == "error" for item in diagnostics)
