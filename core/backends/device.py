import re


TYPE_REPLACEMENTS = [
    (r"\bunsigned\s+long\s+long\b", "ulong"),
    (r"\blong\s+long\b", "long"),
    (r"\bunsigned\s+char\b", "uchar"),
    (r"\bunsigned\s+short\b", "ushort"),
    (r"\bunsigned\s+int\b", "uint"),
    (r"\bunsigned\s+long\b", "ulong"),
    (r"\bsize_t\b", "ulong"),
]


def render_device_components(project, plan):
    declarations = []
    declarations.extend(_render_macros(project))
    declarations.extend(_render_types(project))
    declarations.extend(_render_intrinsic_prelude(plan))

    helpers = []
    for variant in plan.ordered_variants:
        function = project.functions[variant.symbol_id]
        body = rewrite_device_calls(
            function.body,
            function,
            variant.parameter_address_spaces,
            plan,
        )
        helpers.append(_render_helper(function, variant, body))
    return declarations, helpers


def rewrite_kernel_calls(body, kernel, plan):
    spaces = tuple(
        "device" if parameter.type.pointer_depth else None
        for parameter in kernel.parameters
    )
    return rewrite_device_calls(body, kernel, spaces, plan)


def rewrite_device_calls(body, function, spaces, plan):
    targets_by_name = {}
    for call in function.calls:
        key = (
            function.symbol_id,
            spaces,
            call.callee_name,
            call.source.file,
            call.source.line,
            call.source.column,
        )
        target = plan.call_targets.get(key)
        if target:
            targets_by_name.setdefault(call.callee_name, []).append(target)

    rewritten = body
    for name, targets in targets_by_name.items():
        index = 0

        def replace(_match):
            nonlocal index
            target = targets[min(index, len(targets) - 1)]
            index += 1
            return target + "("

        rewritten = re.sub(rf"\b{re.escape(name)}\s*\(", replace, rewritten)

    for name, rule in plan.intrinsics_used.items():
        target = rule.get("target_symbol")
        if target:
            rewritten = re.sub(rf"\b{re.escape(name)}\s*\(", target + "(", rewritten)
    return rewritten


def _render_helper(function, variant, body):
    parameters = []
    for index, parameter in enumerate(function.parameters):
        space = variant.parameter_address_spaces[index]
        parameters.append(_render_parameter(parameter, space))
    params = ", ".join(parameters)
    source = _line_directive(function.source.file, function.body_line or function.source.line)
    rendered_body = _indent(body)
    return (
        f"inline {_map_type(function.return_type.canonical)} {variant.generated_name}({params}) {{\n"
        f"{source}\n{rendered_body}\n}}"
    )


def _render_parameter(parameter, address_space):
    canonical = parameter.type.canonical
    if parameter.type.pointer_depth:
        base = canonical.replace("*", "").strip()
        const_prefix = "const " if base.startswith("const ") else ""
        base = base.removeprefix("const ").strip()
        return f"{const_prefix}{address_space or 'thread'} {_map_type(base)}* {parameter.name}"
    return f"{_map_type(canonical)} {parameter.name}"


def _render_types(project):
    lines = []
    for name, declaration in sorted(project.types.items()):
        if name.startswith("__") or name in {"dim3"}:
            continue
        if declaration.fields:
            fields = "\n".join(
                "    " + _render_field(field)
                for field in declaration.fields
            )
            lines.append(f"struct {name} {{\n{fields}\n}};")
        elif declaration.kind == "alias" and declaration.underlying_type:
            underlying = declaration.underlying_type
            if "anonymous" not in underlying and name != underlying:
                lines.append(f"typedef {_map_type(underlying)} {name};")
    return lines


def _render_field(field):
    canonical = field.type.canonical
    array = re.match(r"(.+?)\s*\[([^]]+)\]$", canonical)
    if array:
        return f"{_map_type(array.group(1))} {field.name}[{array.group(2)}];"
    return f"{_map_type(canonical)} {field.name};"


def _render_macros(project):
    lines = []
    for macro in sorted(project.macros.values(), key=lambda item: item.name):
        if "##" in macro.expression or "..." in macro.expression:
            continue
        params = f"({','.join(macro.parameters)})" if macro.parameters else ""
        lines.append(f"#define {macro.name}{params} {macro.expression}")
    return lines


def _render_intrinsic_prelude(plan):
    if "memset" not in plan.intrinsics_used:
        return []
    return [
        "template <typename T>",
        "inline thread T* mcu_device_memset(thread T* destination, int value, ulong count) {",
        "    thread uchar* bytes = reinterpret_cast<thread uchar*>(destination);",
        "    for (ulong i = 0; i < count; ++i) bytes[i] = uchar(value);",
        "    return destination;",
        "}",
    ]


def _map_type(type_name):
    mapped = " ".join(str(type_name).split())
    for pattern, replacement in TYPE_REPLACEMENTS:
        mapped = re.sub(pattern, replacement, mapped)
    return mapped


def _line_directive(file, line):
    escaped = str(file).replace("\\", "\\\\").replace('"', '\\"')
    return f'#line {line or 1} "{escaped}"'


def _indent(body):
    return "\n".join(
        "    " + line.lstrip() if line.strip() else ""
        for line in body.rstrip().splitlines()
    )
