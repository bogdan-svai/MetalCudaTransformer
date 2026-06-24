import re

from core.ir.link import DeviceLinkPlan
from core.ir.symbols import FunctionVariantIR, generated_symbol_name
from core.passes.atomics import analyze_atomic_parameters
from core.passes.memory_model import validate_memory_model
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
    diagnostics.extend(validate_memory_model(project))

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
    atomic_parameters, atomic_locals, atomic_diagnostics = analyze_atomic_parameters(project, resolved_calls)
    diagnostics.extend(atomic_diagnostics)
    if _has_errors(diagnostics):
        return DeviceLinkPlan(kernels, [], {}, intrinsics, diagnostics, atomic_parameters, atomic_locals)

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
            atomic_callee_parameters = atomic_parameters.get(callee_id, {})
            unsupported_atomic_space = next((
                (callee.parameters[index].name, space)
                for index, space in enumerate(callee_spaces)
                if callee.parameters[index].name in atomic_callee_parameters
                and space not in {"device", "threadgroup"}
            ), None)
            if unsupported_atomic_space:
                name, space = unsupported_atomic_space
                diagnostics.append({
                    "code": "UnsupportedAtomicAddressSpace",
                    "severity": "error",
                    "message": f"Atomic parameter '{name}' resolves to unsupported '{space}' storage.",
                    "file": call.source.file,
                    "line": call.source.line,
                    "column": call.source.column,
                })
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

    return DeviceLinkPlan(kernels, ordered, call_targets, intrinsics, diagnostics, atomic_parameters, atomic_locals)


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
    local_objects = {item.name: item for item in caller.local_objects}
    pointer_flow_sources = {
        **caller_space_by_parameter,
        **{
            item.name: item.storage
            for item in caller.local_objects
            if item.type.kind == "array"
        },
    }
    scalar_local_spaces = {
        item.name: item.storage for item in caller.local_objects
        if not item.type.pointer_depth and item.type.kind != "array"
    }
    inferred_local_spaces, conflicting_locals = _infer_local_pointer_spaces(
        caller.body,
        pointer_flow_sources,
        scalar_local_spaces,
    )
    local_spaces = {
        item.name: item.storage
        for item in caller.local_objects
        if item.type.kind == "array"
    }
    local_spaces.update(inferred_local_spaces)
    spaces = []
    for index, parameter in enumerate(callee.parameters):
        if not parameter.type.pointer_depth:
            spaces.append(None)
            continue
        expression = call.arguments[index] if index < len(call.arguments) else ""
        argument_symbol = (
            call.argument_symbols[index]
            if index < len(call.argument_symbols) else None
        )
        if argument_symbol in caller_space_by_parameter and caller_space_by_parameter[argument_symbol]:
            matched = [caller_space_by_parameter[argument_symbol]]
            local_matched = []
        elif argument_symbol in local_spaces:
            matched = []
            local_matched = [local_spaces[argument_symbol]]
        else:
            matched = [
                space for name, space in caller_space_by_parameter.items()
                if space and re.search(rf"\b{re.escape(name)}\b", expression)
            ]
            local_matched = [
                space for name, space in local_spaces.items()
                if re.search(rf"\b{re.escape(name)}\b", expression)
            ]
        local_matched.extend(
            space for name, space in scalar_local_spaces.items()
            if re.search(rf"&\s*{re.escape(name)}\b", expression)
        )
        inferred_spaces = set(matched + local_matched)
        if len(inferred_spaces) > 1:
            diagnostics.append({
                "code": "IllegalAddressSpaceCast",
                "severity": "error",
                "message": f"Argument '{expression}' combines incompatible pointer address spaces.",
                "file": call.source.file,
                "line": call.source.line,
                "column": call.source.column,
            })
            return None
        if inferred_spaces:
            spaces.append(next(iter(inferred_spaces)))
            continue
        if argument_symbol in local_spaces:
            spaces.append(local_spaces[argument_symbol])
            continue
        if argument_symbol in conflicting_locals:
            diagnostics.append({
                "code": "IllegalAddressSpaceCast",
                "severity": "error",
                "message": f"Local pointer '{argument_symbol}' may refer to incompatible address spaces.",
                "file": call.source.file,
                "line": call.source.line,
                "column": call.source.column,
            })
            return None
        if argument_symbol in local_objects and local_objects[argument_symbol].type.pointer_depth:
            diagnostics.append({
                "code": "AmbiguousAddressSpace",
                "severity": "error",
                "message": f"Cannot prove pointee address space for local pointer '{argument_symbol}'.",
                "file": call.source.file,
                "line": call.source.line,
                "column": call.source.column,
            })
            return None
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


def _infer_local_pointer_spaces(body, parameter_spaces, scalar_local_spaces=None):
    spaces = dict(parameter_spaces)
    scalar_local_spaces = scalar_local_spaces or {}
    for match in re.finditer(
        r"\b(?:extern\s+)?__shared__\s+[A-Za-z_]\w*\s+([A-Za-z_]\w*)",
        body,
    ):
        spaces[match.group(1)] = "threadgroup"
    declarations = list(re.finditer(
        r"(?m)^\s*(?:const\s+)?[A-Za-z_]\w*\s*\*\s*"
        r"(?P<name>[A-Za-z_]\w*)\s*(?:=\s*(?P<value>[^;]+))?;",
        body,
    ))
    conflicts = set()
    for _pass in range(len(declarations) + 1):
        changed = False
        for declaration in declarations:
            name = declaration.group("name")
            expressions = [declaration.group("value") or ""]
            expressions.extend(
                match.group(1)
                for match in re.finditer(rf"\b{re.escape(name)}\s*=\s*([^;]+)", body)
            )
            candidates = {
                space
                for expression in expressions
                for source, space in spaces.items()
                if expression and space and re.search(rf"\b{re.escape(source)}\b", expression)
            }
            candidates.update(
                space for expression in expressions for local, space in scalar_local_spaces.items()
                if expression and re.search(rf"&\s*{re.escape(local)}\b", expression)
            )
            if len(candidates) > 1:
                conflicts.add(name)
                continue
            inferred = next(iter(candidates), None)
            if inferred and spaces.get(name) != inferred:
                spaces[name] = inferred
                changed = True
        if not changed:
            break
    return {
        name: space for name, space in spaces.items()
        if name not in parameter_spaces
    }, conflicts


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
