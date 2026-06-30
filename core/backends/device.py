import re

from core.frontend.regex_parser import find_matching_delimiter, split_top_level


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
        body = rewrite_address_space_syntax(
            body,
            function,
            variant.parameter_address_spaces,
        )
        helpers.append(_render_helper(
            function,
            variant,
            body,
            plan.atomic_parameters.get(function.symbol_id, {}),
        ))
    return declarations, helpers


def rewrite_kernel_calls(body, kernel, plan):
    spaces = tuple(
        "device" if parameter.type.pointer_depth else None
        for parameter in kernel.parameters
    )
    rewritten = rewrite_device_calls(body, kernel, spaces, plan)
    return rewrite_address_space_syntax(rewritten, kernel, spaces)


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
        if rule.get("lowering_handler") == "lower_typed_atomic":
            rewritten = _rewrite_atomic_calls(rewritten, name, rule["target_symbol"])
            continue
        if rule.get("lowering_handler") == "lower_warp_intrinsic":
            rewritten = _rewrite_warp_intrinsic_calls(rewritten, name, rule["target_symbol"])
            continue
        target = rule.get("target_symbol")
        if target:
            if rule.get("lowering_handler") == "lower_threadgroup_barrier":
                rewritten = re.sub(
                    rf"\b{re.escape(name)}\s*\(\s*\)",
                    target,
                    rewritten,
                )
                continue
            rewritten = re.sub(rf"\b{re.escape(name)}\s*\(", target + "(", rewritten)
    return _rewrite_atomic_loads(
        rewritten,
        plan.atomic_parameters.get(function.symbol_id, {}),
    )


def _rewrite_warp_intrinsic_calls(body, name, target):
    pattern = re.compile(rf"\b{re.escape(name)}\s*\(")
    replacements = []
    for match in pattern.finditer(body):
        opening = match.end() - 1
        closing = find_matching_delimiter(body, opening, "(", ")")
        if closing is None:
            continue
        arguments = [item.strip() for item in split_top_level(body[opening + 1:closing]) if item.strip()]
        replacement = None
        if name == "__activemask" and not arguments:
            replacement = target
        elif name == "__syncwarp" and len(arguments) <= 1:
            replacement = target
        elif name in {"__shfl_sync", "__shfl_down_sync"} and len(arguments) in {3, 4}:
            width = arguments[3] if len(arguments) == 4 else "32"
            replacement = (
                f"{target}({arguments[0]}, {arguments[1]}, {arguments[2]}, "
                f"{width}, thread_index_in_simdgroup)"
            )
        elif name in {"__any_sync", "__all_sync"} and len(arguments) == 2:
            replacement = f"{target}({arguments[0]}, {arguments[1]})"
        if replacement:
            replacements.append((match.start(), closing + 1, replacement))
    for start, end, replacement in reversed(replacements):
        body = body[:start] + replacement + body[end:]
    return body


def _rewrite_atomic_calls(body, name, target):
    pattern = re.compile(rf"\b{re.escape(name)}\s*\(")
    replacements = []
    for match in pattern.finditer(body):
        opening = match.end() - 1
        closing = find_matching_delimiter(body, opening, "(", ")")
        if closing is None:
            continue
        arguments = [item.strip() for item in split_top_level(body[opening + 1:closing])]
        if name == "atomicCAS" and len(arguments) == 3:
            replacement = (
                f"mcu_atomic_cas({arguments[0]}, {arguments[1]}, {arguments[2]})"
            )
        elif len(arguments) == 2:
            replacement = f"{target}({arguments[0]}, {arguments[1]}, memory_order_relaxed)"
        else:
            continue
        replacements.append((match.start(), closing + 1, replacement))
    for start, end, replacement in reversed(replacements):
        body = body[:start] + replacement + body[end:]
    return body


def _rewrite_atomic_loads(body, atomic_parameters):
    for name in atomic_parameters:
        body = re.sub(
            rf"\b{re.escape(name)}\s*(?P<index>\[[^]]+\])\s*=\s*(?P<value>[^;]+);",
            rf"atomic_store_explicit(&{name}\g<index>, \g<value>, memory_order_relaxed);",
            body,
        )
        body = re.sub(
            rf"\*\s*{re.escape(name)}\s*=\s*(?P<value>[^;]+);",
            rf"atomic_store_explicit({name}, \g<value>, memory_order_relaxed);",
            body,
        )
        body = re.sub(
            rf"(?<![&\w]){re.escape(name)}\s*(?P<index>\[[^]]+\])",
            rf"atomic_load_explicit(&{name}\g<index>, memory_order_relaxed)",
            body,
        )
        body = re.sub(
            rf"(?<![\w)&])\*\s*{re.escape(name)}\b",
            f"atomic_load_explicit({name}, memory_order_relaxed)",
            body,
        )
    return body


def rewrite_address_space_syntax(body, function, spaces):
    storage = {
        parameter.name: spaces[index]
        for index, parameter in enumerate(function.parameters)
        if spaces[index]
    }
    for item in function.local_objects:
        if not item.type.pointer_depth:
            storage.setdefault(item.name, item.storage)

    pointer_declarations = list(_pointer_declarations(body))
    unresolved = {item["name"] for item in pointer_declarations}
    for _pass in range(len(pointer_declarations) + 1):
        changed = False
        for declaration in pointer_declarations:
            name = declaration["name"]
            if name not in unresolved:
                continue
            expressions = [declaration["initializer"]]
            expressions.extend(
                match.group(1)
                for match in re.finditer(rf"\b{re.escape(name)}\s*=\s*([^;]+)", body)
            )
            inferred = next((
                space
                for expression in expressions if expression
                for space in [_expression_space(expression, storage)]
                if space
            ), None)
            if inferred:
                storage[name] = inferred
                unresolved.remove(name)
                changed = True
        if not changed:
            break

    declaration_pattern = re.compile(
        r"(?m)^(?P<indent>[ \t]*)(?P<const>const\s+)?"
        r"(?P<type>[A-Za-z_]\w*)\s*\*\s*(?P<name>[A-Za-z_]\w*)"
        r"(?P<tail>\s*(?:=\s*[^;]+)?;)"
    )

    def replace_declaration(match):
        space = storage.get(match.group("name"))
        if not space:
            return match.group(0)
        const = match.group("const") or ""
        return (
            f"{match.group('indent')}{const}{space} {match.group('type')}* "
            f"{match.group('name')}{match.group('tail')}"
        )

    rewritten = declaration_pattern.sub(replace_declaration, body)
    cast_pattern = re.compile(
        r"\((?P<const>const\s+)?(?P<type>[A-Za-z_]\w*)\s*\*\)\s*"
        r"(?P<expr>[A-Za-z_]\w*(?:\s*(?:->|\.)\s*[A-Za-z_]\w*)?)"
    )

    def replace_cast(match):
        expression = match.group("expr")
        space = _expression_space(expression, storage)
        if not space:
            return match.group(0)
        const = match.group("const") or ""
        return f"reinterpret_cast<{const}{space} {match.group('type')}*>({expression})"

    return cast_pattern.sub(replace_cast, rewritten)


def _pointer_declarations(body):
    pattern = re.compile(
        r"(?m)^[ \t]*(?:const\s+)?[A-Za-z_]\w*\s*\*\s*"
        r"(?P<name>[A-Za-z_]\w*)\s*(?:=\s*(?P<initializer>[^;]+))?;"
    )
    for match in pattern.finditer(body):
        yield {
            "name": match.group("name"),
            "initializer": (match.group("initializer") or "").strip(),
        }


def _expression_space(expression, storage):
    if not expression:
        return None
    for name, space in storage.items():
        if space and re.search(rf"\b{re.escape(name)}\b", expression):
            return space
    return None


def _render_helper(function, variant, body, atomic_parameters=None):
    atomic_parameters = atomic_parameters or {}
    parameters = []
    for index, parameter in enumerate(function.parameters):
        space = variant.parameter_address_spaces[index]
        atomic_type = atomic_parameters.get(parameter.name, {}).get("type")
        parameters.append(_render_parameter(parameter, space, atomic_type))
    params = ", ".join(parameters)
    source = _line_directive(function.source.file, function.body_line or function.source.line)
    rendered_body = _indent(body)
    return (
        f"inline {_map_type(function.return_type.canonical)} {variant.generated_name}({params}) {{\n"
        f"{source}\n{rendered_body}\n}}"
    )


def _render_parameter(parameter, address_space, atomic_type=None):
    canonical = parameter.type.canonical
    if parameter.type.pointer_depth:
        if atomic_type:
            return f"{address_space or 'thread'} {atomic_type}* {parameter.name}"
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
    lines = []
    if any(
        rule.get("lowering_handler") == "lower_warp_intrinsic"
        for rule in plan.intrinsics_used.values()
    ):
        lines.extend([
            "template <typename T>",
            "inline T mcu_warp_shfl_sync(uint mask, T value, uint source_lane, uint width, uint lane) {",
            "    return (mask == 0xffffffffu && width == 32u) ? simd_shuffle(value, source_lane) : value;",
            "}",
            "template <typename T>",
            "inline T mcu_warp_shfl_down_sync(uint mask, T value, uint delta, uint width, uint lane) {",
            "    return (mask == 0xffffffffu && width == 32u) ? simd_shuffle_down(value, delta) : value;",
            "}",
            "inline int mcu_warp_any_sync(uint mask, bool predicate) {",
            "    return (mask == 0xffffffffu && simd_any(predicate)) ? 1 : 0;",
            "}",
            "inline int mcu_warp_all_sync(uint mask, bool predicate) {",
            "    return (mask == 0xffffffffu && simd_all(predicate)) ? 1 : 0;",
            "}",
            "inline int mcu_warp_any_sync(uint mask, int predicate) {",
            "    return mcu_warp_any_sync(mask, predicate != 0);",
            "}",
            "inline int mcu_warp_all_sync(uint mask, int predicate) {",
            "    return mcu_warp_all_sync(mask, predicate != 0);",
            "}",
        ])
    if "memset" in plan.intrinsics_used:
        for space in ("thread", "device", "threadgroup"):
            lines.extend([
                "template <typename T>",
                f"inline {space} T* mcu_device_memset({space} T* destination, int value, ulong count) {{",
                f"    {space} uchar* bytes = reinterpret_cast<{space} uchar*>(destination);",
                "    for (ulong i = 0; i < count; ++i) bytes[i] = uchar(value);",
                "    return destination;",
                "}",
            ])
    if "atomicCAS" in plan.intrinsics_used:
        lines.extend([
            "template <typename T, typename V>",
            "inline V mcu_atomic_cas(device atomic<T>* target, V compare, V value) {",
            "    T expected = T(compare);",
            "    while (!atomic_compare_exchange_weak_explicit(target, &expected, T(value), memory_order_relaxed, memory_order_relaxed)) {",
            "        if (expected != T(compare)) break;",
            "    }",
            "    return V(expected);",
            "}",
            "template <typename T, typename V>",
            "inline V mcu_atomic_cas(threadgroup atomic<T>* target, V compare, V value) {",
            "    T expected = T(compare);",
            "    while (!atomic_compare_exchange_weak_explicit(target, &expected, T(value), memory_order_relaxed, memory_order_relaxed)) {",
            "        if (expected != T(compare)) break;",
            "    }",
            "    return V(expected);",
            "}",
        ])
    return lines


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
