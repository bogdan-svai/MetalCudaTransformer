import re


ATOMIC_OPERATIONS = {"atomicAdd", "atomicCAS", "atomicMin", "atomicMax", "atomicExch"}


def analyze_atomic_parameters(project, resolved_calls):
    uses = {}
    local_uses = {}
    diagnostics = []
    device_functions = {
        symbol_id: function
        for symbol_id, function in project.functions.items()
        if function.is_device_callable
    }

    for symbol_id, function in device_functions.items():
        parameters = {parameter.name: parameter for parameter in function.parameters}
        locals_by_name = {item.name: item for item in function.local_objects}
        for call in function.calls:
            if call.callee_name not in ATOMIC_OPERATIONS:
                continue
            pointer = (
                call.argument_symbols[0]
                if call.argument_symbols and call.argument_symbols[0]
                else _root_identifier(call.arguments[0] if call.arguments else "")
            )
            parameter = parameters.get(pointer)
            local = locals_by_name.get(pointer)
            if parameter and parameter.type.pointer_depth:
                canonical = parameter.type.canonical
                destination = uses.setdefault(symbol_id, {})
            elif local and local.storage == "threadgroup":
                canonical = local.type.canonical
                destination = local_uses.setdefault(symbol_id, {})
            else:
                diagnostics.append(_diagnostic(
                    "UnsupportedAtomicAddressSpace",
                    f"Atomic target '{call.arguments[0] if call.arguments else ''}' must refer to device or shared memory.",
                    call,
                ))
                continue
            atomic_type = _atomic_type(canonical)
            if not atomic_type:
                diagnostics.append(_diagnostic(
                    "UnsupportedAtomicType",
                    f"Atomic operation '{call.callee_name}' requires a signed or unsigned 32-bit integer pointer.",
                    call,
                ))
                continue
            if any(not _supported_value_type(item) for item in call.argument_types[1:]):
                diagnostics.append(_diagnostic(
                    "UnsupportedAtomicType",
                    f"Atomic operation '{call.callee_name}' requires 32-bit integral values.",
                    call,
                ))
                continue
            destination.setdefault(pointer, {
                "type": atomic_type,
                "operations": [],
                "memory_order": "memory_order_relaxed",
            })["operations"].append(call.callee_name)

    changed = True
    while changed:
        changed = False
        for symbol_id, function in device_functions.items():
            caller_parameters = {parameter.name: parameter for parameter in function.parameters}
            caller_locals = {item.name: item for item in function.local_objects}
            for call in function.calls:
                callee_id = resolved_calls.get(_call_key(call))
                callee_uses = uses.get(callee_id, {})
                if not callee_uses or callee_id not in device_functions:
                    continue
                callee = device_functions[callee_id]
                for index, callee_parameter in enumerate(callee.parameters):
                    if callee_parameter.name not in callee_uses or index >= len(call.arguments):
                        continue
                    caller_name = (
                        call.argument_symbols[index]
                        if index < len(call.argument_symbols) and call.argument_symbols[index]
                        else _root_identifier(call.arguments[index])
                    )
                    inherited = dict(callee_uses[callee_parameter.name])
                    if caller_name in caller_parameters:
                        destination = uses.setdefault(symbol_id, {})
                    elif caller_name in caller_locals and caller_locals[caller_name].storage == "threadgroup":
                        destination = local_uses.setdefault(symbol_id, {})
                    else:
                        continue
                    current = destination.get(caller_name)
                    if current != inherited:
                        destination[caller_name] = inherited
                        changed = True

    for parameters in uses.values():
        for use in parameters.values():
            use["operations"] = sorted(set(use["operations"]))
    for locals_for_function in local_uses.values():
        for use in locals_for_function.values():
            use["operations"] = sorted(set(use["operations"]))
    diagnostics.extend(_validate_atomic_accesses(device_functions, uses, local_uses))
    return uses, local_uses, diagnostics


def _validate_atomic_accesses(functions, parameter_uses, local_uses):
    diagnostics = []
    for symbol_id, function in functions.items():
        names = set(parameter_uses.get(symbol_id, {})) | set(local_uses.get(symbol_id, {}))
        for name in names:
            unsupported = re.search(
                rf"(?:\*\s*{re.escape(name)}\b\s*\)?|\b{re.escape(name)}\s*\[[^]]+\])\s*"
                r"(?:\+\+|--|[+\-*/%&|^]=)",
                function.body,
            )
            if not unsupported:
                continue
            diagnostics.append({
                "code": "InvalidAtomicNonAtomicAccess",
                "severity": "error",
                "message": f"Compound non-atomic access to atomic target '{name}' is unsupported; use a CUDA atomic operation.",
                "file": function.source.file,
                "line": (function.body_line or function.source.line or 1) + function.body[:unsupported.start()].count("\n"),
                "column": None,
            })
    return diagnostics


def _atomic_type(canonical):
    base = re.sub(r"\s*\[[^]]*\]\s*$", "", canonical)
    base = base.replace("const", "").replace("*", "").strip()
    normalized = " ".join(base.split())
    if normalized in {"unsigned int", "uint32_t", "uint"}:
        return "atomic_uint"
    if normalized in {"int", "int32_t"}:
        return "atomic_int"
    return None


def _supported_value_type(type_ir):
    canonical = " ".join(type_ir.canonical.replace("const", "").split())
    return "*" not in canonical and not re.search(r"\b(?:float|double|half)\b", canonical)


def _root_identifier(expression):
    match = re.search(r"\b([A-Za-z_]\w*)\b", expression or "")
    return match.group(1) if match else ""


def _call_key(call):
    return call.caller_id, call.callee_name, call.source.file, call.source.line, call.source.column


def _diagnostic(code, message, call):
    return {
        "code": code,
        "severity": "error",
        "message": message,
        "file": call.source.file,
        "line": call.source.line,
        "column": call.source.column,
    }
