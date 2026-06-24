from dataclasses import dataclass
from dataclasses import replace
import os
from pathlib import Path
import re
import tempfile

from core.backends.host import rewrite_triple_chevrons
from core.frontend.clang_probe import CUDA_STUBS, _configure
from core.frontend.regex_parser import iter_function_calls, read_text, split_top_level
from core.ir.symbols import (
    CallEdgeIR,
    FunctionSymbolIR,
    LocalObjectIR,
    MacroIR,
    ParameterIR,
    ProjectSymbolIR,
    SourceLocationIR,
    TypeDeclarationIR,
    TypeIR,
    canonical_symbol_id,
)


@dataclass
class SemanticFrontendResult:
    project: ProjectSymbolIR | None
    complete: bool
    diagnostics: list[dict]


def parse_semantic_project(metadata):
    device_functions = [
        function for function in metadata.get("functions", [])
        if function.get("kind") in {"KERNEL", "DEVICE_FUNC", "HOST_DEVICE_FUNC"}
    ]
    if not device_functions:
        return SemanticFrontendResult(None, True, [])

    try:
        from clang.cindex import Config, CursorKind, Index
    except Exception as exc:
        return _unavailable(f"libclang Python bindings unavailable: {exc}")
    if not _configure(Config):
        return _unavailable("Native libclang library was not found")

    source_paths = [Path(path) for path in metadata.get("sources", [])]
    semantic_source = _build_semantic_source(source_paths)
    diagnostics = []
    try:
        index = Index.create()
        with tempfile.NamedTemporaryFile("w", suffix=".cpp", delete=False, encoding="utf-8") as handle:
            virtual_path = handle.name
            handle.write(semantic_source)
        try:
            translation_unit = index.parse(virtual_path, args=_clang_args(source_paths, metadata))
            diagnostics.extend(_clang_diagnostics(translation_unit))
            project = _collect_project(
                translation_unit.cursor,
                metadata,
                source_paths,
                CursorKind,
                diagnostics,
            )
        finally:
            try:
                os.remove(virtual_path)
            except OSError:
                pass
    except Exception as exc:
        return _unavailable(f"Semantic frontend failed: {exc}")

    discovered = {
        (function.name, Path(function.source.file).name)
        for function in project.functions.values()
        if function.is_device_callable
    }
    missing = []
    for function in device_functions:
        key = (function["name"], Path(function["file"]).name)
        if key not in discovered and not any(item[0] == function["name"] for item in discovered):
            missing.append(function)
            diagnostics.append({
                "code": "IncompleteSemanticIR",
                "severity": "error",
                "message": f"Semantic frontend did not resolve device function '{function['name']}'.",
                "file": function["file"],
                "line": function["line"],
                "column": None,
            })
    project.diagnostics = diagnostics
    return SemanticFrontendResult(project, not missing, diagnostics)


def _build_semantic_source(source_paths):
    preamble = [
        "#define __CUDACC__ 1",
        CUDA_STUBS,
        "typedef unsigned char uint8_t;",
        "typedef unsigned int uint32_t;",
        "typedef unsigned long uint64_t;",
        "typedef unsigned long size_t;",
        'extern "C" void* memset(void*, int, unsigned long);',
        "",
    ]
    ordered = sorted(source_paths, key=lambda path: (path.suffix not in {".h", ".hpp"}, path.as_posix()))
    for path in ordered:
        text = rewrite_triple_chevrons(read_text(path))
        text = re.sub(r'^\s*#\s*include\s*[<"][^>"]+[>"]\s*$', "", text, flags=re.MULTILINE)
        preamble.extend([
            f'#line 1 "{path.as_posix()}"',
            text,
            "",
        ])
    return "\n".join(preamble)


def _clang_args(source_paths, metadata=None):
    args = ["-x", "c++", "-std=c++17"]
    context = (metadata or {}).get("compile_context", {})
    for directory in context.get("include_paths", []):
        args.extend(["-I", directory])
    args.extend(f"-D{definition}" for definition in context.get("definitions", []))
    args.extend(f"-U{name}" for name in context.get("undefinitions", []))
    for directory in sorted({str(path.resolve().parent) for path in source_paths}):
        args.extend(["-I", directory])
    sdk = "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk"
    resource_dir = "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib/clang/17"
    if os.path.isdir(sdk):
        args.extend(["-isysroot", sdk])
    if os.path.isdir(resource_dir):
        args.extend(["-resource-dir", resource_dir])
    return args


def _clang_diagnostics(translation_unit):
    diagnostics = []
    for item in translation_unit.diagnostics:
        if item.severity < item.Warning:
            continue
        diagnostics.append({
            "code": "SemanticFrontendDiagnostic",
            "severity": "error" if item.severity >= item.Error else "warning",
            "message": str(item),
            "file": str(item.location.file) if item.location.file else None,
            "line": item.location.line or None,
            "column": item.location.column or None,
        })
    return diagnostics


def _collect_project(root, metadata, source_paths, cursor_kind, diagnostics):
    textual = metadata.get("functions", [])
    textual_by_name = {}
    for function in textual:
        textual_by_name.setdefault(function["name"], []).append(function)

    functions = {}
    function_cursors = []
    types = {}
    allowed_names = {path.name for path in source_paths}
    declared_type_locations = _declared_type_locations(source_paths)

    for cursor in _walk(root):
        if cursor.kind == cursor_kind.FUNCTION_DECL and cursor.is_definition():
            candidates = textual_by_name.get(cursor.spelling, [])
            text_function = _match_text_function(cursor, candidates)
            if not text_function:
                continue
            parameters = [
                ParameterIR(argument.spelling or f"arg{index}", _parameter_type_ir(argument.type))
                for index, argument in enumerate(cursor.get_arguments())
            ]
            symbol_id = canonical_symbol_id(
                cursor.spelling,
                [parameter.type.canonical for parameter in parameters],
            )
            source = SourceLocationIR(
                text_function["file"],
                text_function.get("line"),
                cursor.location.column or None,
            )
            function = FunctionSymbolIR(
                symbol_id=symbol_id,
                name=cursor.spelling,
                kind=text_function["kind"],
                qualifiers=_qualifiers(text_function["kind"]),
                return_type=_type_ir(cursor.result_type),
                parameters=parameters,
                source=source,
                body_line=text_function.get("body_line"),
                body=text_function.get("body", ""),
                translation_unit=text_function["file"],
            )
            function.local_objects = _collect_local_objects(cursor, cursor_kind, function)
            previous = functions.get(symbol_id)
            if previous and previous.source != function.source:
                diagnostics.append({
                    "code": "DuplicateDeviceDefinition",
                    "severity": "error",
                    "message": f"Multiple definitions found for '{symbol_id}'.",
                    "file": function.source.file,
                    "line": function.source.line,
                    "column": function.source.column,
                })
            else:
                functions[symbol_id] = function
                function_cursors.append((cursor, function))

        if cursor.kind in {cursor_kind.STRUCT_DECL, cursor_kind.ENUM_DECL, cursor_kind.TYPEDEF_DECL}:
            location_file = str(cursor.location.file) if cursor.location.file else ""
            if cursor.spelling not in declared_type_locations and (
                location_file and Path(location_file).name not in allowed_names
            ):
                continue
            declaration = _type_declaration(cursor, cursor_kind)
            if declaration and declaration.name not in {"dim3"}:
                if declaration.name in declared_type_locations:
                    declaration = replace(
                        declaration,
                        source=declared_type_locations[declaration.name],
                    )
                types.setdefault(declaration.name, declaration)

    known_ids = set(functions)
    macros = _collect_macros(source_paths)
    for cursor, function in function_cursors:
        calls = []
        call_ordinals = {}
        for child in _walk(cursor):
            if child.kind != cursor_kind.CALL_EXPR:
                continue
            referenced = child.referenced
            callee_name = referenced.spelling if referenced else child.spelling
            if not callee_name or callee_name == function.name and child == cursor:
                continue
            callee_id = None
            if referenced and referenced.kind == cursor_kind.FUNCTION_DECL:
                callee_id = canonical_symbol_id(
                    callee_name,
                    [argument.type.get_canonical().spelling for argument in referenced.get_arguments()],
                )
                if callee_id not in known_ids:
                    callee_id = None
            argument_cursors = list(child.get_arguments())
            arguments = tuple(_tokens(argument) for argument in argument_cursors)
            ordinal = call_ordinals.get(callee_name, 0)
            call_ordinals[callee_name] = ordinal + 1
            call_source = _original_call_location(function, callee_name, ordinal)
            calls.append(CallEdgeIR(
                caller_id=function.symbol_id,
                callee_name=callee_name,
                callee_id=callee_id,
                arguments=arguments,
                source=call_source,
                argument_types=tuple(_type_ir(argument.type) for argument in argument_cursors),
                argument_symbols=tuple(
                    _argument_symbol(argument, cursor_kind)
                    for argument in argument_cursors
                ),
            ))
        function.calls = _recover_body_calls(
            function,
            calls,
            functions,
            types,
            macros,
        )

    return ProjectSymbolIR(
        frontend="libclang_semantic",
        translation_units=[path.as_posix() for path in source_paths if path.suffix in {".cu", ".cpp", ".cc", ".cxx"}],
        functions=functions,
        types=types,
        macros=macros,
        diagnostics=diagnostics,
    )


def _walk(cursor):
    yield cursor
    for child in cursor.get_children():
        yield from _walk(child)


def _match_text_function(cursor, candidates):
    if not candidates:
        return None
    location_name = Path(str(cursor.location.file)).name if cursor.location.file else ""
    same_file = [item for item in candidates if Path(item["file"]).name == location_name]
    pool = same_file or candidates
    argument_count = len(list(cursor.get_arguments()))
    same_arity = [item for item in pool if len(item.get("parameters", [])) == argument_count]
    typed_pool = same_arity or pool
    canonical_types = [
        _normalize_comparable_type(argument.type.get_canonical().spelling)
        for argument in cursor.get_arguments()
    ]
    typed_matches = [
        item for item in typed_pool
        if [
            _normalize_comparable_type(_text_parameter_type(parameter))
            for parameter in item.get("parameters", [])
        ] == canonical_types
    ]
    return (typed_matches or typed_pool)[0]


def _text_parameter_type(parameter):
    text = parameter.split("=", 1)[0].strip()
    match = re.match(r"(.+?)([A-Za-z_]\w*)\s*(?:\[[^]]*\])?$", text)
    return match.group(1).strip() if match else text


def _normalize_comparable_type(type_name):
    normalized = re.sub(r"\s*\[[^]]*\]\s*$", "*", str(type_name))
    return " ".join(normalized.replace(" *", "*").replace("* ", "*").split())


def _type_ir(clang_type):
    spelling = clang_type.spelling
    canonical = clang_type.get_canonical().spelling
    pointer_depth = canonical.count("*")
    kind = "pointer" if pointer_depth else "array" if "[" in canonical else "value"
    return TypeIR(
        spelling=spelling,
        canonical=canonical,
        kind=kind,
        is_const=canonical.strip().startswith("const "),
        pointer_depth=pointer_depth,
    )


def _parameter_type_ir(clang_type):
    result = _type_ir(clang_type)
    array = re.match(r"(?P<base>.+?)\s*\[[^]]*\]\s*$", result.canonical)
    if not array:
        return result
    base = array.group("base").strip()
    return TypeIR(
        spelling=result.spelling,
        canonical=f"{base}*",
        kind="pointer",
        is_const=base.startswith("const "),
        pointer_depth=1,
    )


def _type_declaration(cursor, cursor_kind):
    name = cursor.spelling
    if not name:
        return None
    location = SourceLocationIR(
        str(cursor.location.file) if cursor.location.file else "",
        cursor.location.line or None,
        cursor.location.column or None,
    )
    if cursor.kind == cursor_kind.TYPEDEF_DECL:
        fields = []
        for child in _walk(cursor):
            if child.kind == cursor_kind.FIELD_DECL:
                fields.append(ParameterIR(child.spelling, _type_ir(child.type)))
        return TypeDeclarationIR(
            name=name,
            kind="struct_alias" if fields else "alias",
            fields=tuple(fields),
            underlying_type=cursor.underlying_typedef_type.spelling,
            source=location,
        )
    if cursor.kind == cursor_kind.STRUCT_DECL:
        fields = tuple(
            ParameterIR(child.spelling, _type_ir(child.type))
            for child in cursor.get_children()
            if child.kind == cursor_kind.FIELD_DECL
        )
        return TypeDeclarationIR(name, "struct", fields, None, location)
    return TypeDeclarationIR(name, "enum", tuple(), None, location)


def _collect_local_objects(function_cursor, cursor_kind, function):
    objects = []
    for cursor in _walk(function_cursor):
        if cursor.kind != cursor_kind.VAR_DECL:
            continue
        kind_name = getattr(cursor.type.kind, "name", str(cursor.type.kind))
        extent = None
        if kind_name == "CONSTANTARRAY":
            try:
                extent = cursor.type.get_array_size()
            except Exception:
                extent = None
        storage = "threadgroup" if any(
            child.kind == cursor_kind.ANNOTATE_ATTR and "mcu_shared" in (_tokens(child) or child.spelling)
            for child in cursor.get_children()
        ) else "thread"
        objects.append(LocalObjectIR(
            name=cursor.spelling,
            type=_type_ir(cursor.type),
            storage=storage,
            array_extent=extent if extent is not None and extent >= 0 else None,
            variable_length=kind_name == "VARIABLEARRAY",
            incomplete_array=kind_name == "INCOMPLETEARRAY",
            source=SourceLocationIR(
                function.source.file,
                cursor.location.line or function.source.line,
                cursor.location.column or None,
            ),
        ))
    return objects


def _argument_symbol(cursor, cursor_kind):
    references = []
    for child in _walk(cursor):
        if child.kind != cursor_kind.DECL_REF_EXPR or not child.referenced:
            continue
        if child.referenced.kind in {cursor_kind.PARM_DECL, cursor_kind.VAR_DECL}:
            references.append(child.referenced.spelling)
    return references[0] if references else None


def _collect_macros(source_paths):
    macros = {}
    pattern = re.compile(
        r"^\s*#\s*define\s+(?P<name>[A-Za-z_]\w*)"
        r"(?:\((?P<params>[^)]*)\))?[ \t]+(?P<expression>[^\n\\]+)$",
        re.MULTILINE,
    )
    for path in source_paths:
        for match in pattern.finditer(read_text(path)):
            name = match.group("name")
            if name.startswith("__") or name in {"DEBUG_PRINT", "GPU_COMPATIBLE"}:
                continue
            parameters = tuple(
                item.strip() for item in (match.group("params") or "").split(",") if item.strip()
            )
            macros[name] = MacroIR(
                name,
                parameters,
                match.group("expression").strip(),
                SourceLocationIR(path.as_posix(), read_text(path)[:match.start()].count("\n") + 1),
            )
    return macros


def _declared_type_locations(source_paths):
    locations = {}
    patterns = [
        re.compile(r"\btypedef\s+struct(?:\s+[A-Za-z_]\w*)?\s*\{.*?\}\s*([A-Za-z_]\w*)\s*;", re.DOTALL),
        re.compile(r"\btypedef\s+[^;{}]+?\s+([A-Za-z_]\w*)\s*;"),
        re.compile(r"\b(?:struct|enum)\s+([A-Za-z_]\w*)\s*\{"),
    ]
    for path in source_paths:
        text = read_text(path)
        for pattern in patterns:
            for match in pattern.finditer(text):
                name = match.group(1)
                locations[name] = SourceLocationIR(
                    path.as_posix(),
                    text[:match.start()].count("\n") + 1,
                    1,
                )
    return locations


def _tokens(cursor):
    return " ".join(token.spelling for token in cursor.get_tokens()).strip()


def _original_call_location(function, callee_name, ordinal):
    matches = list(re.finditer(rf"\b{re.escape(callee_name)}\s*\(", function.body))
    if ordinal < len(matches):
        match = matches[ordinal]
        prefix = function.body[:match.start()]
        line = (function.body_line or function.source.line or 1) + prefix.count("\n")
        last_newline = prefix.rfind("\n")
        column = match.start() + 1 if last_newline < 0 else match.start() - last_newline
        return SourceLocationIR(function.source.file, line, column)
    return SourceLocationIR(function.source.file, function.source.line, function.source.column)


def _qualifiers(kind):
    return {
        "KERNEL": ("global",),
        "DEVICE_FUNC": ("device",),
        "HOST_DEVICE_FUNC": ("host", "device"),
    }.get(kind, ("host",))


def _deduplicate_calls(calls):
    unique = {}
    for call in calls:
        key = (call.callee_name, call.source.file, call.source.line, call.source.column)
        unique[key] = call
    return list(unique.values())


def _recover_body_calls(function, ast_calls, functions, types, macros):
    calls = list(ast_calls)
    existing = {
        (call.callee_name, call.source.line, call.source.column)
        for call in calls
    }
    excluded = {
        "if", "for", "while", "switch", "sizeof", "alignof", "decltype",
        "static_cast", "reinterpret_cast", "const_cast", "dynamic_cast",
        "return", "new", "delete", "throw",
        *types.keys(),
        *macros.keys(),
    }
    names = list(dict.fromkeys(
        match.group(1)
        for match in re.finditer(r"\b([A-Za-z_]\w*)\s*\(", function.body)
        if match.group(1) not in excluded
    ))
    functions_by_name = {}
    for candidate in functions.values():
        functions_by_name.setdefault(candidate.name, []).append(candidate)

    for name in names:
        for ordinal, (_offset, argument_text) in enumerate(iter_function_calls(function.body, name)):
            arguments = tuple(
                part for part in split_top_level(argument_text) if part.strip()
            )
            source = _original_call_location(function, name, ordinal)
            if (name, source.line, source.column) in existing:
                continue
            candidates = [
                candidate for candidate in functions_by_name.get(name, [])
                if len(candidate.parameters) == len(arguments)
            ]
            callee_id = candidates[0].symbol_id if len(candidates) == 1 else None
            calls.append(CallEdgeIR(
                caller_id=function.symbol_id,
                callee_name=name,
                callee_id=callee_id,
                arguments=arguments,
                source=source,
            ))
            existing.add((name, source.line, source.column))
    return _deduplicate_calls(calls)


def _unavailable(message):
    diagnostic = {
        "code": "SemanticFrontendUnavailable",
        "severity": "error",
        "message": message,
        "file": None,
        "line": None,
        "column": None,
    }
    return SemanticFrontendResult(None, False, [diagnostic])
