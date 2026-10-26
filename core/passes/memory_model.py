import re


THREADGROUP_MEMORY_LIMIT = 32 * 1024


def validate_memory_model(project):
    diagnostics = []
    for function in project.functions.values():
        if not function.is_device_callable:
            continue
        dynamic_shared = []
        for parameter in function.parameters:
            if parameter.type.pointer_depth and "[" in parameter.type.canonical:
                diagnostics.append(
                    {
                        "code": "InvalidArrayParameterDecay",
                        "severity": "error",
                        "message": f"Parameter '{parameter.name}' has an unsupported multidimensional array decay.",
                        "file": function.source.file,
                        "line": function.source.line,
                        "column": function.source.column,
                    }
                )
        for item in function.local_objects:
            if item.storage == "threadgroup" and function.kind != "KERNEL":
                diagnostics.append(
                    _diagnostic(
                        "InvalidSharedDeclaration",
                        f"Shared object '{item.name}' must be declared in a kernel entry point.",
                        item,
                    )
                )
            if item.variable_length:
                diagnostics.append(
                    _diagnostic(
                        "UnsupportedVariableLengthArray",
                        f"Device array '{item.name}' has a runtime extent; only compile-time extents are supported.",
                        item,
                    )
                )
            if item.incomplete_array:
                if item.storage == "threadgroup":
                    dynamic_shared.append(item)
                else:
                    diagnostics.append(
                        _diagnostic(
                            "InvalidSharedDeclaration",
                            f"Incomplete device array '{item.name}' is not dynamic shared memory.",
                            item,
                        )
                    )
            if item.storage == "threadgroup" and item.array_extent:
                byte_size = item.array_extent * _type_size(item.type.canonical)
                if byte_size > THREADGROUP_MEMORY_LIMIT:
                    diagnostics.append(
                        _diagnostic(
                            "ThreadgroupMemoryLimitExceeded",
                            f"Static shared array '{item.name}' requires {byte_size} bytes; limit is {THREADGROUP_MEMORY_LIMIT}.",
                            item,
                        )
                    )
        if len(dynamic_shared) > 1:
            diagnostics.append(
                _diagnostic(
                    "InvalidSharedDeclaration",
                    "A kernel may declare at most one dynamic shared-memory array.",
                    dynamic_shared[1],
                )
            )
        diagnostics.extend(_validate_mixed_pointer_casts(function))
        diagnostics.extend(_validate_local_pointer_flow(function))
        diagnostics.extend(_validate_device_memset(function))
    return diagnostics


def _validate_mixed_pointer_casts(function):
    diagnostics = []
    storage = {
        parameter.name: "device" if function.kind == "KERNEL" else "parameter"
        for parameter in function.parameters
        if parameter.type.pointer_depth
    }
    storage.update({item.name: item.storage for item in function.local_objects})
    semantic_casts = [
        cast
        for statement in function.statements
        for cast in (statement.attributes or {}).get("pointer_casts", [])
    ]
    for cast in semantic_casts:
        spaces = {storage[name] for name in cast.get("sources", []) if name in storage}
        if len(spaces - {"parameter"}) > 1:
            diagnostics.append(
                {
                    "code": "IllegalAddressSpaceCast",
                    "severity": "error",
                    "message": "Pointer cast combines values from incompatible CUDA storage classes.",
                    "file": function.source.file,
                    "line": function.source.line,
                    "column": function.source.column,
                }
            )
    if semantic_casts:
        return diagnostics
    pattern = re.compile(
        r"\((?:const\s+)?[A-Za-z_]\w*\s*\*\)\s*"
        r"(?P<expression>\([^;]+?\)|[A-Za-z_]\w*(?:\s*(?:->|\.)\s*[A-Za-z_]\w*)?)"
    )
    for match in pattern.finditer(function.body):
        spaces = {
            space
            for name, space in storage.items()
            if re.search(rf"\b{re.escape(name)}\b", match.group("expression"))
        } - {"parameter"}
        if len(spaces) > 1:
            diagnostics.append(
                {
                    "code": "IllegalAddressSpaceCast",
                    "severity": "error",
                    "message": "Pointer cast combines values from incompatible CUDA storage classes.",
                    "file": function.source.file,
                    "line": (function.body_line or function.source.line or 1)
                    + function.body[: match.start()].count("\n"),
                    "column": None,
                }
            )
    return diagnostics


def _validate_local_pointer_flow(function):
    if function.kind != "KERNEL":
        return []
    spaces = {
        parameter.name: "device"
        for parameter in function.parameters
        if parameter.type.pointer_depth
    }
    spaces.update(
        {item.name: item.storage for item in function.local_objects if item.type.kind == "array"}
    )
    scalar_local_spaces = {
        item.name: item.storage
        for item in function.local_objects
        if not item.type.pointer_depth and item.type.kind != "array"
    }
    diagnostics = []
    for item in function.local_objects:
        if not item.type.pointer_depth:
            continue
        semantic_alias = function.effects.get("pointer_aliases", {}).get(item.name)
        if semantic_alias in spaces:
            spaces[item.name] = spaces[semantic_alias]
            continue
        semantic_sources = [
            source
            for statement in function.statements
            for assignment in (statement.attributes or {}).get("pointer_assignments", [])
            if assignment.get("target") == item.name
            for source in assignment.get("sources", [])
        ]
        inferred = {spaces[source] for source in semantic_sources if source in spaces}
        if len(inferred) > 1:
            diagnostics.append(
                {
                    "code": "IllegalAddressSpaceCast",
                    "severity": "error",
                    "message": f"Local pointer '{item.name}' may refer to incompatible address spaces.",
                    "file": item.source.file,
                    "line": item.source.line,
                    "column": item.source.column,
                }
            )
            continue
        if len(inferred) == 1:
            spaces[item.name] = next(iter(inferred))
            continue
        expressions = [
            match.group(1)
            for match in re.finditer(rf"\b{re.escape(item.name)}\s*=\s*([^;]+)", function.body)
        ]
        inferred = {
            space
            for expression in expressions
            for name, space in spaces.items()
            if re.search(rf"\b{re.escape(name)}\b", expression)
        }
        inferred.update(
            space
            for expression in expressions
            for name, space in scalar_local_spaces.items()
            if re.search(rf"&\s*{re.escape(name)}\b", expression)
        )
        if len(inferred) > 1:
            diagnostics.append(
                {
                    "code": "IllegalAddressSpaceCast",
                    "severity": "error",
                    "message": f"Local pointer '{item.name}' may refer to incompatible address spaces.",
                    "file": item.source.file,
                    "line": item.source.line,
                    "column": item.source.column,
                }
            )
        elif len(inferred) == 1:
            spaces[item.name] = next(iter(inferred))
        elif re.search(
            rf"(?:\*\s*{re.escape(item.name)}\b|\b{re.escape(item.name)}\s*\[)", function.body
        ):
            diagnostics.append(
                {
                    "code": "AmbiguousAddressSpace",
                    "severity": "error",
                    "message": f"Cannot prove pointee address space for local pointer '{item.name}'.",
                    "file": item.source.file,
                    "line": item.source.line,
                    "column": item.source.column,
                }
            )
    return diagnostics


def _validate_device_memset(function):
    diagnostics = []
    parameters = {item.name: item.type for item in function.parameters}
    locals_by_name = {item.name: item.type for item in function.local_objects}
    for call in function.calls:
        if call.callee_name != "memset" or not call.arguments or call.callee_id:
            continue
        symbol = call.argument_symbols[0] if call.argument_symbols else None
        target_type = parameters.get(symbol) or locals_by_name.get(symbol)
        if target_type and (target_type.pointer_depth or target_type.kind == "array"):
            continue
        if call.argument_types and (
            call.argument_types[0].pointer_depth or call.argument_types[0].kind == "array"
        ):
            continue
        diagnostics.append(
            {
                "code": "UnsupportedDeviceMemset",
                "severity": "error",
                "message": "Device memset destination must be a pointer or fixed/shared array.",
                "file": call.source.file,
                "line": call.source.line,
                "column": call.source.column,
            }
        )
    return diagnostics


def _type_size(canonical):
    base = re.sub(r"\s*\[[^]]*\]\s*$", "", canonical)
    base = " ".join(base.replace("const", "").split())
    if base in {"char", "signed char", "unsigned char", "bool"}:
        return 1
    if base in {"short", "unsigned short"}:
        return 2
    if base in {"double", "long long", "unsigned long long"}:
        return 8
    return 4


def _diagnostic(code, message, item):
    return {
        "code": code,
        "severity": "error",
        "message": message,
        "file": item.source.file,
        "line": item.source.line,
        "column": item.source.column,
    }
