import re

from core.frontend.regex_parser import find_matching_delimiter

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
        body = mark_readonly_literal_arrays(body, function)
        helpers.append(
            _render_helper(
                function,
                variant,
                body,
                plan.atomic_parameters.get(function.symbol_id, {}),
            )
        )
    return declarations, helpers


def rewrite_kernel_calls(body, kernel, plan):
    spaces = tuple(
        "device" if parameter.type.pointer_depth else None for parameter in kernel.parameters
    )
    rewritten = rewrite_device_calls(body, kernel, spaces, plan)
    rewritten = rewrite_address_space_syntax(rewritten, kernel, spaces)
    return mark_readonly_literal_arrays(rewritten, kernel)


def rewrite_device_calls(body, function, spaces, plan):
    edits = []
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
            edits.append((call, target))

    suffixes = []
    for call in function.calls:
        key = (
            function.symbol_id,
            call.callee_name,
            call.source.file,
            call.source.line,
            call.source.column,
        )
        rule = plan.intrinsic_calls.get(key)
        if not rule:
            continue
        target = rule.get("target_symbol")
        handler = rule.get("lowering_handler")
        suffix = ""
        if handler == "lower_typed_atomic":
            if call.callee_name == "atomicCAS":
                target = "mcu_atomic_cas"
            else:
                suffix = ", memory_order_relaxed"
        elif handler == "lower_threadgroup_barrier":
            suffix = "mem_flags::mem_threadgroup"
        elif handler == "lower_warp_intrinsic" and call.callee_name in {
            "__shfl_sync",
            "__shfl_down_sync",
        }:
            suffix = (
                ", 32" if call.explicit_argument_count == 3 else ""
            ) + ", thread_index_in_simdgroup"
        if target:
            edits.append((call, target))
        if suffix:
            suffixes.append((call, suffix))
    rewritten = _apply_call_edits(body, function, edits, suffixes)

    return _rewrite_atomic_loads(
        rewritten,
        plan.atomic_parameters.get(function.symbol_id, {}),
    )


def _apply_call_edits(body, function, edits, argument_suffixes=()):
    """Apply validated original-source byte ranges in one reconstruction."""
    if not edits and not argument_suffixes:
        return body
    if body != function.body or function.body_range is None:
        raise ValueError("Call edits require the unchanged original function body")
    data = body.encode("utf-8")
    ranges = []
    for call, target in edits:
        span = call.identifier_range
        if span is None or span.file != function.body_range.file or not call.callee_usr:
            raise ValueError("Call edit lacks original source identity or range")
        start, end = span.start - function.body_range.start, span.end - function.body_range.start
        if not 0 <= start < end <= len(data) or data[start:end] != call.callee_name.encode("utf-8"):
            raise ValueError("Call edit does not match the original identifier")
        ranges.append((start, end, target.encode("utf-8")))
    for call, suffix in argument_suffixes:
        span = call.call_range
        if span is None or span.file != function.body_range.file or not call.callee_usr:
            raise ValueError("Call insertion lacks original source identity or range")
        start, end = span.start - function.body_range.start, span.end - function.body_range.start
        if not 0 <= start < end <= len(data) or data[end - 1 : end] != b")":
            raise ValueError("Call insertion does not match the original closing parenthesis")
        ranges.append((end - 1, end - 1, suffix.encode("utf-8")))
    pieces = []
    copied = 0
    for start, end, replacement in sorted(ranges):
        if start < copied:
            raise ValueError("Overlapping call edits")
        pieces.extend((data[copied:start], replacement))
        copied = end
    pieces.append(data[copied:])
    return b"".join(pieces).decode("utf-8")


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

    semantic_assignments = [
        assignment
        for statement in function.statements
        for assignment in (statement.attributes or {}).get("pointer_assignments", [])
    ]
    for name, source in function.effects.get("pointer_aliases", {}).items():
        if source in storage:
            storage[name] = storage[source]
    for assignment in semantic_assignments:
        inferred = next(
            (storage[source] for source in assignment.get("sources", []) if source in storage),
            None,
        )
        if inferred:
            storage[assignment["target"]] = inferred

    pointer_declarations = list(_pointer_declarations(body))
    unresolved = {item["name"] for item in pointer_declarations}
    for _pass in range(len(pointer_declarations) + 1):
        changed = False
        for declaration in pointer_declarations:
            name = declaration["name"]
            if name not in unresolved:
                continue
            expressions = [declaration["initializer"]]
            if name not in storage:
                expressions.extend(
                    match.group(1)
                    for match in re.finditer(rf"\b{re.escape(name)}\s*=\s*([^;]+)", body)
                )
            inferred = next(
                (
                    space
                    for expression in expressions
                    if expression
                    for space in [_expression_space(expression, storage)]
                    if space
                ),
                None,
            )
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


def mark_readonly_literal_arrays(body, function=None):
    semantic_names = None
    if function is not None:
        written = {
            access.get("array")
            for statement in function.statements
            for access in (statement.attributes or {}).get("array_accesses", [])
            if access.get("mode") == "write"
        }
        semantic_names = {
            item.name
            for item in function.local_objects
            if item.type.kind == "array"
            and item.initializer
            and "{" in item.initializer
            and item.name not in written
        }
    array_pattern = re.compile(
        r"(?P<prefix>^[ \t]*)(?P<type>(?:(?:unsigned|signed)\s+)?(?:WORD|BYTE|uint|int|ulong|long|uchar|char))\s+"
        r"(?P<name>[A-Za-z_]\w*)\s*\[\s*(?P<size>\d+)\s*\]\s*=\s*\{"
        r"(?P<values>[^{}]*)"
        r"\};",
        re.MULTILINE | re.DOTALL,
    )
    replacements = []
    for match in array_pattern.finditer(body):
        name = match.group("name")
        if semantic_names is not None and name not in semantic_names:
            continue
        remaining = body[: match.start()] + body[match.end() :]
        if re.search(rf"\b{re.escape(name)}\s*\[[^]]+\]\s*=", remaining):
            continue
        if re.search(rf"\b{re.escape(name)}\s*=", remaining):
            continue
        replacement = (
            f"{match.group('prefix')}const {match.group('type')} {name}"
            f"[{match.group('size')}] = {{{match.group('values')}}};"
        )
        replacements.append((match.start(), match.end(), replacement))
    for start, end, replacement in reversed(replacements):
        body = body[:start] + replacement + body[end:]
    return body


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
    readonly = readonly_pointer_parameters(
        function.parameters,
        body,
        atomic_parameters,
        effects=function.effects,
    )
    parameters = []
    for index, parameter in enumerate(function.parameters):
        space = variant.parameter_address_spaces[index]
        atomic_type = atomic_parameters.get(parameter.name, {}).get("type")
        parameters.append(
            _render_parameter(
                parameter,
                space,
                atomic_type,
                readonly=parameter.name in readonly,
            )
        )
    params = ", ".join(parameters)
    source = _line_directive(function.source.file, function.body_line or function.source.line)
    rendered_body = _indent(body)
    return (
        f"inline {_map_type(function.return_type.canonical)} {variant.generated_name}({params}) {{\n"
        f"{source}\n{rendered_body}\n}}"
    )


def _render_parameter(parameter, address_space, atomic_type=None, readonly=False):
    canonical = parameter.type.canonical
    if parameter.type.pointer_depth:
        if atomic_type:
            return f"{address_space or 'thread'} {atomic_type}* {parameter.name}"
        base = canonical.replace("*", "").strip()
        const_prefix = "const " if readonly or base.startswith("const ") else ""
        base = base.removeprefix("const ").strip()
        return f"{const_prefix}{address_space or 'thread'} {_map_type(base)}* {parameter.name}"
    return f"{_map_type(canonical)} {parameter.name}"


def readonly_pointer_parameters(parameters, body, atomic_parameters=None, effects=None):
    atomic_parameters = atomic_parameters or {}
    effects = effects or {}
    # Semantic IR is authoritative. The textual checks remain only for callers
    # that use this low-level helper without a semantic project.
    if effects:
        written = set(effects.get("writes_parameters", []))
        unknown = set(effects.get("unknown_calls", []))
        passed_to_unknown = set(effects.get("unknown_pointer_arguments", []))
        # Keep mutable local aliases valid in generated Metal. A CUDA kernel
        # may bind a readonly parameter to a non-const local pointer; adding
        # ``const`` to the parameter would make that legal CUDA pattern fail
        # Metal's stricter qualifier checks.
        aliased = set(effects.get("pointer_aliases", {}).values())
        return {
            parameter.name
            for parameter in parameters
            if parameter.type.pointer_depth
            and parameter.name not in atomic_parameters
            and parameter.name not in written
            and parameter.name not in passed_to_unknown
            and parameter.name not in aliased
            and not (
                _has_mutable_local_pointer_alias(body, parameter.name) if not effects else False
            )
            and not (unknown and parameter.name in passed_to_unknown)
        }
    readonly = set()
    for parameter in parameters:
        if not parameter.type.pointer_depth or parameter.name in atomic_parameters:
            continue
        if _pointer_is_written(body, parameter.name):
            continue
        if _pointer_is_passed_to_call(body, parameter.name):
            continue
        readonly.add(parameter.name)
    return readonly


def _has_mutable_local_pointer_alias(body, parameter_name):
    """Conservatively detect a local pointer initialized from a parameter."""
    cursor = 0
    while True:
        position = body.find(parameter_name, cursor)
        if position < 0:
            return False
        before = body[position - 1] if position else " "
        after = (
            body[position + len(parameter_name)]
            if position + len(parameter_name) < len(body)
            else " "
        )
        if not (before.isalnum() or before == "_") and not (after.isalnum() or after == "_"):
            statement_start = max(body.rfind(";", 0, position), body.rfind("\n", 0, position))
            prefix = body[statement_start + 1 : position]
            if "*" in prefix and "=" in prefix:
                return True
        cursor = position + len(parameter_name)


def _pointer_is_written(body, name):
    escaped = re.escape(name)
    return any(
        re.search(pattern, body)
        for pattern in [
            rf"\b{escaped}\s*\[[^]]+\]\s*(?:[-+*/%&|^]?=|--|\+\+)",
            rf"\b{escaped}\s*->\s*[A-Za-z_]\w*(?:\s*\[[^]]+\])?\s*(?:[-+*/%&|^]?=|--|\+\+)",
            rf"(?:--|\+\+)\s*{escaped}\s*\[",
            rf"\*\s*{escaped}\s*(?:[-+*/%&|^]?=|--|\+\+)",
            rf"\b{escaped}\s*(?:[-+*/%&|^]?=)",
            rf"\b(?:const\s+)?[A-Za-z_]\w*\s*\*\s*[A-Za-z_]\w*\s*=\s*[^;]*\b{escaped}\b",
        ]
    )


def _pointer_is_passed_to_call(body, name):
    escaped = re.escape(name)
    keywords = {
        "if",
        "for",
        "while",
        "switch",
        "return",
        "sizeof",
        "reinterpret_cast",
        "static_cast",
    }
    pattern = re.compile(r"\b(?P<callee>[A-Za-z_]\w*)\s*\(")
    for match in pattern.finditer(body):
        if match.group("callee") in keywords:
            continue
        opening = match.end() - 1
        closing = find_matching_delimiter(body, opening, "(", ")")
        if closing is None:
            continue
        if re.search(rf"\b{escaped}\b", body[opening + 1 : closing]):
            return True
    return False


def _render_types(project):
    lines = []
    for name, declaration in sorted(project.types.items()):
        if name.startswith("__") or name in {"dim3"}:
            continue
        if declaration.fields:
            fields = "\n".join("    " + _render_field(field) for field in declaration.fields)
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
        lines.extend(
            [
                "inline uint mcu_warp_active_mask() { return 0xffffffffu; }",
                "inline void mcu_warp_sync(uint mask = 0xffffffffu) {",
                "    simdgroup_barrier(mem_flags::mem_none);",
                "}",
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
            ]
        )
    if "memset" in plan.intrinsics_used:
        for space in ("thread", "device", "threadgroup"):
            lines.extend(
                [
                    "template <typename T>",
                    f"inline {space} T* mcu_device_memset({space} T* destination, int value, ulong count) {{",
                    f"    {space} uchar* bytes = reinterpret_cast<{space} uchar*>(destination);",
                    "    for (ulong i = 0; i < count; ++i) bytes[i] = uchar(value);",
                    "    return destination;",
                    "}",
                ]
            )
    if "atomicCAS" in plan.intrinsics_used:
        lines.extend(
            [
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
            ]
        )
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
        "    " + line.lstrip() if line.strip() else "" for line in body.rstrip().splitlines()
    )
