import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

from core.backends.host import rewrite_triple_chevrons
from core.cache import CacheStore, fingerprint
from core.frontend.clang_probe import CUDA_STUBS, _configure
from core.frontend.regex_parser import iter_function_calls, read_text, split_top_level
from core.ir.symbols import (
    CallEdgeIR,
    FunctionSymbolIR,
    LaunchIR,
    LocalObjectIR,
    MacroIR,
    ParameterIR,
    ProjectSymbolIR,
    SourceLocationIR,
    SourceRangeIR,
    StatementIR,
    TypeDeclarationIR,
    TypeIR,
    canonical_symbol_id,
)
from core.version import MCU_VERSION, SEMANTIC_IR_SCHEMA


@dataclass
class SemanticFrontendResult:
    project: ProjectSymbolIR | None
    complete: bool
    diagnostics: list[dict]
    cached: bool = False


def parse_semantic_project(metadata):
    device_functions = [
        function
        for function in metadata.get("functions", [])
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
    translation_units = [
        path for path in source_paths if path.suffix in {".cu", ".cpp", ".cc", ".cxx"}
    ]
    if len(translation_units) > 1 and not metadata.get("_semantic_monolithic"):
        unit_result = _parse_semantic_units(metadata, translation_units)
        if unit_result is not None:
            return unit_result
    cache = CacheStore()
    cache_key = fingerprint(
        {
            "schema": SEMANTIC_IR_SCHEMA,
            "mcu_version": MCU_VERSION,
            "compile_context": metadata.get("compile_context", {}),
            "sources": [
                (str(path.resolve()), hashlib.sha256(path.read_bytes()).hexdigest())
                for path in source_paths
            ],
        }
    )
    cached = _restore_semantic_cache(cache, cache_key)
    if cached is not None:
        return cached
    semantic_source = _build_semantic_source(source_paths)
    diagnostics = []
    try:
        index = Index.create()
        with tempfile.NamedTemporaryFile(
            "w", suffix=".cpp", delete=False, encoding="utf-8"
        ) as handle:
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
            diagnostics.append(
                {
                    "code": "IncompleteSemanticIR",
                    "severity": "error",
                    "message": f"Semantic frontend did not resolve device function '{function['name']}'.",
                    "file": function["file"],
                    "line": function["line"],
                    "column": None,
                }
            )
    project.diagnostics = diagnostics
    result = SemanticFrontendResult(project, not missing, diagnostics)
    _publish_semantic_cache(cache, cache_key, result)
    return result


def _parse_semantic_units(metadata, translation_units):
    """Parse and cache translation units independently before relinking them."""
    all_sources = [Path(path) for path in metadata.get("sources", [])]
    headers = [path for path in all_sources if path.suffix in {".h", ".hpp"}]
    projects = []
    cached = True
    for unit in translation_units:
        unit_metadata = dict(metadata)
        unit_metadata["_semantic_monolithic"] = True
        unit_metadata["sources"] = [str(path) for path in [unit, *headers]]
        unit_metadata["functions"] = [
            item
            for item in metadata.get("functions", [])
            if Path(item.get("file", "")).resolve() == unit.resolve()
        ]
        result = parse_semantic_project(unit_metadata)
        if not result.complete or result.project is None:
            return None
        projects.append(result.project)
        cached = cached and result.cached

    merged = ProjectSymbolIR(
        frontend="libclang_semantic",
        translation_units=[path.as_posix() for path in translation_units],
        functions={},
        diagnostics=[],
    )
    for project in projects:
        merged.functions.update(project.functions)
        merged.types.update(project.types)
        merged.macros.update(project.macros)
        merged.launches.extend(project.launches)
        merged.translation_units.extend(
            item for item in project.translation_units if item not in merged.translation_units
        )
        merged.diagnostics.extend(project.diagnostics)
    merged, unresolved = relink_semantic_project(merged)
    if unresolved:
        return None
    return SemanticFrontendResult(merged, True, merged.diagnostics, cached=cached)


def relink_semantic_project(project):
    """Resolve calls after combining semantic IR from multiple objects.

    Object-level Clang parses can see declarations from headers while the
    definition lives in another translation unit.  Their call edge therefore
    has a stable USR but no local ``callee_id``.  Relinking by USR restores the
    cross-unit edge without guessing by spelling; effects are then propagated
    over the reunited graph.
    """
    usr_to_id = {
        function.clang_usr: symbol_id
        for symbol_id, function in project.functions.items()
        if function.clang_usr
    }
    unresolved = []
    for function in project.functions.values():
        for index, call in enumerate(function.calls):
            if call.callee_id and call.callee_id in project.functions:
                continue
            callee_id = usr_to_id.get(call.callee_usr)
            if callee_id:
                function.calls[index] = replace(call, callee_id=callee_id)
            elif call.callee_id:
                unresolved.append(call)
    _propagate_effects(project.functions)
    return project, unresolved


def _restore_semantic_cache(cache, key):
    with tempfile.TemporaryDirectory(prefix="mcu-semantic-restore-") as directory:
        target = Path(directory) / "semantic.json"
        if not cache.restore("semantic", key, "semantic.json", target):
            return None
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
            project = ProjectSymbolIR.from_dict(payload["project"])
            return SemanticFrontendResult(
                project, bool(payload["complete"]), payload["diagnostics"], cached=True
            )
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
            return None


def _publish_semantic_cache(cache, key, result):
    with tempfile.TemporaryDirectory(prefix="mcu-semantic-publish-") as directory:
        source = Path(directory) / "semantic.json"
        source.write_text(
            json.dumps(
                {
                    "complete": result.complete,
                    "diagnostics": result.diagnostics,
                    "project": result.project.to_dict(),
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        cache.publish("semantic", key, "semantic.json", source)


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
    ordered = sorted(
        source_paths, key=lambda path: (path.suffix not in {".h", ".hpp"}, path.as_posix())
    )
    for path in ordered:
        text = rewrite_triple_chevrons(read_text(path))
        text = re.sub(r'^\s*#\s*include\s*[<"][^>"]+[>"]\s*$', "", text, flags=re.MULTILINE)
        preamble.extend(
            [
                f'#line 1 "{path.as_posix()}"',
                text,
                "",
            ]
        )
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
        diagnostics.append(
            {
                "code": "SemanticFrontendDiagnostic",
                "severity": "error" if item.severity >= item.Error else "warning",
                "message": str(item),
                "file": str(item.location.file) if item.location.file else None,
                "line": item.location.line or None,
                "column": item.location.column or None,
            }
        )
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
            function.statements = _collect_statements(cursor, cursor_kind, function)
            function.effects = _collect_effects(cursor, cursor_kind)
            previous = functions.get(symbol_id)
            if previous and previous.source != function.source:
                diagnostics.append(
                    {
                        "code": "DuplicateDeviceDefinition",
                        "severity": "error",
                        "message": f"Multiple definitions found for '{symbol_id}'.",
                        "file": function.source.file,
                        "line": function.source.line,
                        "column": function.source.column,
                    }
                )
            else:
                functions[symbol_id] = function
                function_cursors.append((cursor, function))

        if cursor.kind in {
            cursor_kind.STRUCT_DECL,
            cursor_kind.ENUM_DECL,
            cursor_kind.TYPEDEF_DECL,
        }:
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
    from core.frontend.source_calls import (
        collect_original_macros,
        collect_original_type_locations,
    )

    macros = _collect_macros(source_paths)
    original_macros = collect_original_macros(source_paths, _clang_args(source_paths, metadata))
    macros.update(original_macros)
    original_type_locations = collect_original_type_locations(
        source_paths, _clang_args(source_paths, metadata)
    )
    for name, declaration in list(types.items()):
        location = original_type_locations.get(name)
        if location:
            types[name] = replace(declaration, source=location)
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
                    [
                        argument.type.get_canonical().spelling
                        for argument in referenced.get_arguments()
                    ],
                )
                if callee_id not in known_ids:
                    callee_id = None
            argument_cursors = list(child.get_arguments())
            arguments = tuple(_tokens(argument) for argument in argument_cursors)
            ordinal = call_ordinals.get(callee_name, 0)
            call_ordinals[callee_name] = ordinal + 1
            call_source = _original_call_location(function, callee_name, ordinal)
            calls.append(
                CallEdgeIR(
                    caller_id=function.symbol_id,
                    callee_name=callee_name,
                    callee_id=callee_id,
                    arguments=arguments,
                    source=call_source,
                    argument_types=tuple(_type_ir(argument.type) for argument in argument_cursors),
                    argument_symbols=tuple(
                        _argument_symbol(argument, cursor_kind) for argument in argument_cursors
                    ),
                )
            )
        function.calls = _recover_body_calls(
            function,
            calls,
            functions,
            types,
            macros,
        )

    _attach_original_calls(functions, source_paths, metadata)
    _propagate_effects(functions)

    kernel_index = {}
    for symbol_id, function in functions.items():
        if function.kind == "KERNEL":
            kernel_index.setdefault(function.name, []).append(symbol_id)
    launch_irs = []
    for item in metadata.get("launches", []):
        candidates = kernel_index.get(item["kernel"], [])
        referenced_usr = item.get("kernel_usr")
        if referenced_usr:
            usr_candidates = [
                symbol_id
                for symbol_id in candidates
                if functions[symbol_id].clang_usr == referenced_usr
            ]
            if usr_candidates:
                candidates = usr_candidates
        kernel_id = candidates[0] if len(candidates) == 1 else None
        if not candidates:
            diagnostics.append(
                {
                    "code": "UnresolvedKernelLaunch",
                    "severity": "error",
                    "message": f"Launch target '{item['kernel']}' has no semantic kernel definition.",
                    "file": item.get("file"),
                    "line": item.get("line"),
                    "column": None,
                }
            )
        elif len(candidates) > 1:
            diagnostics.append(
                {
                    "code": "AmbiguousKernelLaunch",
                    "severity": "error",
                    "message": f"Launch target '{item['kernel']}' matches multiple semantic kernels.",
                    "file": item.get("file"),
                    "line": item.get("line"),
                    "column": None,
                }
            )
        launch_irs.append(
            LaunchIR(
                kernel=item["kernel"],
                kernel_id=kernel_id,
                file=item["file"],
                line=item["line"],
                grid_expr=item["grid_expr"],
                block_expr=item["block_expr"],
                shared_expr=item.get("shared_expr", "0"),
                stream_expr=item.get("stream_expr", "0"),
                args=tuple(item.get("args", [])),
                syntax=item.get("syntax", "cudaLaunch"),
                source_range=(
                    SourceRangeIR(**item["source_range"]) if item.get("source_range") else None
                ),
            )
        )

    return ProjectSymbolIR(
        frontend="libclang_semantic",
        translation_units=[
            path.as_posix()
            for path in source_paths
            if path.suffix in {".cu", ".cpp", ".cc", ".cxx"}
        ],
        functions=functions,
        types=types,
        macros=macros,
        launches=launch_irs,
        diagnostics=diagnostics,
    )


def _attach_original_calls(functions, source_paths, metadata):
    from clang.cindex import CursorKind

    from core.frontend.source_calls import (
        collect_original_functions,
        direct_call_range,
        is_intrinsic_declaration,
        original_call_extent,
    )

    originals = collect_original_functions(
        source_paths, _clang_args(source_paths, metadata), _parameter_type_ir
    )
    matched = {}
    usr_to_id = {}
    for symbol_id, function in functions.items():
        record = originals.get((str(Path(function.source.file).resolve()), symbol_id))
        if record is None:
            continue
        function.clang_usr = record.cursor.get_usr()
        function.body_range = record.body_range
        function.body = record.body
        function.body_line = record.body_cursor.extent.start.line
        function.source = replace(
            function.source,
            file=record.body_range.file,
            line=record.cursor.location.line or function.source.line,
            column=record.cursor.location.column or function.source.column,
        )
        function.translation_unit = record.body_range.file
        # Rebuild statement/control-flow metadata from the original unit. The
        # synthetic translation unit remains available for legacy analyses, but
        # validation must reason about the source the backend will rewrite.
        function.statements = _collect_statements(record.body_cursor, CursorKind, function)
        matched[symbol_id] = record
        usr_to_id[function.clang_usr] = symbol_id

    for symbol_id, record in matched.items():
        function = functions[symbol_id]
        if not function.is_device_callable:
            continue
        calls = []
        data = function.body.encode("utf-8")
        for cursor in record.body_cursor.walk_preorder():
            if cursor.kind != CursorKind.CALL_EXPR:
                continue
            referenced = cursor.referenced
            name = referenced.spelling if referenced else cursor.spelling
            if not name:
                continue
            usr = referenced.get_usr() if referenced else None
            identifier_range, issue = direct_call_range(cursor, record.body_range, data)
            arguments = list(cursor.get_arguments())
            location = cursor.location
            calls.append(
                CallEdgeIR(
                    caller_id=symbol_id,
                    callee_name=name,
                    callee_id=usr_to_id.get(usr),
                    arguments=tuple(_tokens(a) for a in arguments),
                    source=SourceLocationIR(function.source.file, location.line, location.column),
                    argument_types=tuple(_type_ir(a.type) for a in arguments),
                    argument_symbols=tuple(_argument_symbol(a, CursorKind) for a in arguments),
                    callee_usr=usr,
                    identifier_range=identifier_range,
                    rewrite_issue=issue,
                    call_range=original_call_extent(cursor, record.body_range, data),
                    intrinsic_declaration=is_intrinsic_declaration(
                        referenced, record.compatibility_header
                    ),
                    explicit_argument_count=sum(a.extent.start.file is not None for a in arguments),
                )
            )
        # Preserve unresolved legacy diagnostics/intrinsics during migration.
        # Resolved helper calls must only come from the original AST, never a
        # recovered name/ordinal (which can refer to a comment or another scope).
        names = {c.callee_name for c in calls}
        for call in function.calls:
            if call.callee_name not in names and record.has_body_errors:
                if call.callee_id:
                    calls.append(
                        replace(
                            call,
                            rewrite_issue="Original Clang call identity could not be established.",
                        )
                    )
                else:
                    calls.append(call)
        function.calls = calls

    # Preserve the original reference behind cudaLaunch when Clang exposes it.
    # The textual extractor still supplies launch expressions and arguments.
    for item in metadata.get("launches", []):
        for record in originals.values():
            if record is None:
                continue
            for cursor in record.body_cursor.walk_preorder():
                if cursor.kind != CursorKind.CALL_EXPR or cursor.spelling != "cudaLaunch":
                    continue
                if cursor.location.line != item.get("line"):
                    continue
                location_file = str(Path(str(cursor.location.file)).resolve())
                if location_file != str(Path(item.get("file", "")).resolve()):
                    continue
                launch_range = original_call_extent(
                    cursor,
                    record.body_range,
                    Path(record.body_range.file).read_bytes()[
                        record.body_range.start : record.body_range.end
                    ],
                )
                if launch_range:
                    item["source_range"] = launch_range.to_dict()
                arguments = list(cursor.get_arguments())
                if arguments and arguments[0].referenced is not None:
                    item["kernel_usr"] = arguments[0].referenced.get_usr()
                    break


def _propagate_effects(functions):
    """Propagate observable effects through resolved device-call edges."""
    changed = True
    while changed:
        changed = False
        for function in functions.values():
            effects = function.effects
            propagated = []
            effect_changed = False
            for call in function.calls:
                call_changed, impure_name = _propagate_call_effect(function, call, functions)
                effect_changed = call_changed or effect_changed
                if impure_name:
                    propagated.append(impure_name)
            if propagated:
                for name in propagated:
                    if name not in effects.setdefault("propagated_impure_calls", []):
                        effects["propagated_impure_calls"].append(name)
                        effect_changed = True
                if effects.get("pure"):
                    effects["pure"] = False
                    effect_changed = True
            if (
                effects.get("writes_memory")
                or effects.get("uses_atomic")
                or effects.get("uses_barrier")
            ):
                if effects.get("pure"):
                    effects["pure"] = False
                    effect_changed = True
            elif not effects.get("unknown_calls") and not effects.get("propagated_impure_calls"):
                if not effects.get("pure"):
                    effects["pure"] = True
                    effect_changed = True
            if effect_changed:
                changed = True


def _propagate_call_effect(function, call, functions):
    effects = function.effects
    callee = functions.get(call.callee_id)
    if callee is None:
        parameter_names = {item.name for item in function.parameters}
        unknown = effects.setdefault("unknown_pointer_arguments", [])
        before = len(unknown)
        unknown.extend(
            argument
            for argument in call.argument_symbols
            if argument in parameter_names and argument not in unknown
        )
        return len(unknown) != before, None
    changed = False
    unknown_calls = effects.get("unknown_calls", [])
    if call.callee_name in unknown_calls:
        effects["unknown_calls"] = [name for name in unknown_calls if name != call.callee_name]
        changed = True
    callee_effects = callee.effects
    for key in ("writes_memory", "reads_memory", "uses_atomic", "uses_barrier"):
        if callee_effects.get(key) and not effects.get(key):
            effects[key] = True
            changed = True
    parameter_names = {item.name for item in function.parameters}
    written = set(callee_effects.get("writes_parameters", []))
    for index, parameter in enumerate(callee.parameters):
        if parameter.name not in written or index >= len(call.argument_symbols):
            continue
        argument = _resolve_pointer_alias(effects, call.argument_symbols[index])
        if argument and argument not in effects.setdefault("writes_argument_symbols", []):
            effects["writes_argument_symbols"].append(argument)
            changed = True
        if argument in parameter_names and argument not in effects.setdefault(
            "writes_parameters", []
        ):
            effects["writes_parameters"].append(argument)
            changed = True
    return changed, None if callee_effects.get("pure", False) else callee.name


def _collect_effects(cursor, cursor_kind):
    """Summarize observable effects using AST cursors, conservatively."""
    pointer_aliases = {}
    parameter_names = {item.spelling for item in cursor.get_arguments() if item.spelling}
    pointer_aliases = _collect_pointer_aliases(cursor, cursor_kind)
    calls, writes_memory, reads_memory, has_return, writes_parameters, reads_parameters = (
        _collect_effect_events(cursor, cursor_kind, parameter_names)
    )
    calls = sorted(set(item for item in calls if item))
    atomics = [name for name in calls if name.startswith("atomic")]
    barriers = [name for name in calls if name.startswith("__syncthreads")]
    unknown_calls = [name for name in calls if name not in atomics + barriers]
    pure = not writes_memory and not atomics and not barriers and not unknown_calls
    return {
        "reads_memory": reads_memory,
        "writes_memory": writes_memory,
        "uses_atomic": bool(atomics),
        "uses_barrier": bool(barriers),
        "uses_geometry": bool(
            re.search(r"\b(?:threadIdx|blockIdx|blockDim|gridDim)\b", _tokens(cursor))
        ),
        "has_return": has_return,
        "calls": calls,
        "unknown_calls": unknown_calls,
        "writes_parameters": sorted(writes_parameters),
        "reads_parameters": sorted(reads_parameters),
        "writes_argument_symbols": [],
        "pointer_aliases": pointer_aliases,
        "pure": pure,
    }


def _collect_pointer_aliases(cursor, cursor_kind):
    aliases = {}
    for child in _walk(cursor):
        if child.kind != cursor_kind.VAR_DECL or not child.spelling:
            continue
        if child.type.get_canonical().spelling.count("*") == 0:
            continue
        references = [
            item.referenced.spelling
            for item in _walk(child)
            if item.kind == cursor_kind.DECL_REF_EXPR
            and item.referenced is not None
            and item.referenced.spelling
        ]
        if references:
            aliases[child.spelling] = references[-1]
    return aliases


def _collect_effect_events(cursor, cursor_kind, parameter_names):
    calls = []
    writes_memory = False
    reads_memory = False
    has_return = False
    writes_parameters = set()
    reads_parameters = set()
    for child in _walk(cursor):
        if child.kind == cursor_kind.CALL_EXPR:
            name = child.spelling or (child.referenced.spelling if child.referenced else "")
            calls.append(name)
            if name and (name.startswith("atomic") or name.startswith("__syncthreads")):
                writes_memory = True
        elif child.kind == cursor_kind.RETURN_STMT:
            has_return = True
        elif child.kind in {cursor_kind.BINARY_OPERATOR, cursor_kind.COMPOUND_ASSIGNMENT_OPERATOR}:
            tokens = [token.spelling for token in child.get_tokens()]
            if "=" in tokens or child.kind == cursor_kind.COMPOUND_ASSIGNMENT_OPERATOR:
                operands = list(child.get_children())
                lhs = operands[0] if operands else child
                for reference in _walk(lhs):
                    if reference.kind != cursor_kind.DECL_REF_EXPR or reference.referenced is None:
                        continue
                    type_spelling = reference.referenced.type.get_canonical().spelling
                    if "*" in type_spelling or "[" in type_spelling:
                        writes_memory = True
                        if reference.referenced.spelling in parameter_names:
                            writes_parameters.add(reference.referenced.spelling)
            if any("[" in token for token in tokens):
                reads_memory = True
                for reference in _walk(child):
                    if reference.kind == cursor_kind.DECL_REF_EXPR and reference.referenced:
                        if reference.referenced.spelling in parameter_names:
                            reads_parameters.add(reference.referenced.spelling)
    return calls, writes_memory, reads_memory, has_return, writes_parameters, reads_parameters


def _resolve_pointer_alias(effects, symbol):
    """Resolve a local pointer alias to its stable root, with cycle protection."""
    aliases = effects.get("pointer_aliases", {})
    seen = set()
    current = symbol
    while current and current in aliases and current not in seen:
        seen.add(current)
        current = aliases[current]
    return current


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
        item
        for item in typed_pool
        if [
            _normalize_comparable_type(_text_parameter_type(parameter))
            for parameter in item.get("parameters", [])
        ]
        == canonical_types
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
        storage = (
            "threadgroup"
            if any(
                child.kind == cursor_kind.ANNOTATE_ATTR
                and "mcu_shared" in (_tokens(child) or child.spelling)
                for child in cursor.get_children()
            )
            else "thread"
        )
        objects.append(
            LocalObjectIR(
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
                initializer=_local_initializer(cursor),
            )
        )
    return objects


def _local_initializer(cursor):
    tokens = _tokens(cursor)
    if "=" not in tokens:
        return ""
    return tokens.split("=", 1)[1].strip()


def _collect_statements(function_cursor, cursor_kind, function):
    """Retain statement-level AST structure for typed optimization passes."""
    statement_kinds = {
        cursor_kind.COMPOUND_STMT,
        cursor_kind.DECL_STMT,
        cursor_kind.RETURN_STMT,
        cursor_kind.IF_STMT,
        cursor_kind.FOR_STMT,
        cursor_kind.WHILE_STMT,
        cursor_kind.DO_STMT,
        cursor_kind.SWITCH_STMT,
        cursor_kind.CASE_STMT,
        cursor_kind.DEFAULT_STMT,
        cursor_kind.BREAK_STMT,
        cursor_kind.CONTINUE_STMT,
        cursor_kind.CALL_EXPR,
    }
    statements = []

    def visit(cursor, parent_index=None):
        statement_index = parent_index
        child_indices = []
        if cursor.kind in statement_kinds:
            statement_index = len(statements)
            statements.append(None)
            attributes = {
                "array_accesses": _statement_array_accesses(cursor, cursor_kind),
                "pointer_assignments": _statement_pointer_assignments(cursor, cursor_kind),
                "pointer_casts": _statement_pointer_casts(cursor, cursor_kind),
                "non_atomic_targets": _statement_non_atomic_targets(cursor, cursor_kind),
                **_statement_control_attributes(cursor, cursor_kind, function, function_cursor),
                "parent_index": parent_index,
                "child_indices": child_indices,
            }
            statements[statement_index] = StatementIR(
                kind=getattr(cursor.kind, "name", str(cursor.kind)),
                source=SourceLocationIR(
                    function.source.file,
                    cursor.location.line or function.source.line,
                    cursor.location.column or None,
                ),
                text=_tokens(cursor),
                children=tuple(
                    getattr(child.kind, "name", str(child.kind))
                    for child in cursor.get_children()
                    if child.kind in statement_kinds
                ),
                attributes=attributes,
            )
        for child in cursor.get_children():
            before = len(statements)
            visit(child, statement_index)
            if statement_index is not None and len(statements) > before:
                child_indices.extend(
                    index
                    for index in range(before, len(statements))
                    if statements[index] is not None
                    and statements[index].attributes.get("parent_index") == statement_index
                )

    visit(function_cursor)
    _attach_cfg_links(statements)
    return statements


def _attach_cfg_links(statements):
    """Attach conservative lexical CFG edges to the statement tree."""
    for index, statement in enumerate(statements):
        attributes = statement.attributes
        children = list(attributes.get("child_indices", []))
        descendants = [
            child
            for child in range(index + 1, len(statements))
            if _is_descendant(statements, child, index)
        ]
        after = (max(descendants) + 1) if descendants else index + 1
        successors = []
        if statement.kind in {"IF_STMT", "SWITCH_STMT"}:
            successors.extend(children)
            if after < len(statements):
                successors.append(after)
        elif statement.kind in {"FOR_STMT", "WHILE_STMT", "DO_STMT"}:
            if children:
                successors.append(children[0])
            if after < len(statements):
                successors.append(after)
            if descendants:
                attributes["cfg_back_edges"] = [descendants[-1], index]
        elif index + 1 < len(statements):
            successors.append(index + 1)
        attributes["cfg_successors"] = list(dict.fromkeys(successors))


def _is_descendant(statements, candidate, ancestor):
    parent = statements[candidate].attributes.get("parent_index")
    while parent is not None:
        if parent == ancestor:
            return True
        parent = statements[parent].attributes.get("parent_index")
    return False


def _statement_control_attributes(cursor, cursor_kind, function, function_cursor):
    """Collect conservative control-flow facts used by legality validation."""
    calls = []
    has_early_exit = False
    if cursor.kind == cursor_kind.CALL_EXPR:
        name = cursor.spelling or (cursor.referenced.spelling if cursor.referenced else "")
        if name:
            calls.append(name)
    for child in _walk(cursor):
        if child.kind == cursor_kind.CALL_EXPR:
            name = child.spelling or (child.referenced.spelling if child.referenced else "")
            if name:
                calls.append(name)
        elif child.kind in {
            cursor_kind.RETURN_STMT,
            cursor_kind.BREAK_STMT,
            cursor_kind.CONTINUE_STMT,
        }:
            has_early_exit = True

    condition = ""
    loop_init = loop_update = ""
    if cursor.kind in {
        cursor_kind.IF_STMT,
        cursor_kind.WHILE_STMT,
        cursor_kind.DO_STMT,
        cursor_kind.FOR_STMT,
        cursor_kind.SWITCH_STMT,
    }:
        children = list(cursor.get_children())
        if children:
            if cursor.kind == cursor_kind.FOR_STMT:
                # libclang exposes the three header clauses as the first
                # children (declaration, condition, increment), even though
                # punctuation is absent from the cursor token stream.
                if len(children) >= 3:
                    loop_init = _tokens(children[0])
                    condition = _tokens(children[1])
                    loop_update = _tokens(children[2])
            else:
                condition = _tokens(children[0])
    pointer_parameters = {item.name for item in function.parameters if item.type.pointer_depth > 0}
    thread_variables = set()
    for declaration in _walk(function_cursor):
        if declaration.kind != cursor_kind.VAR_DECL or not declaration.spelling:
            continue
        expression = _tokens(declaration)
        if re.search(r"\b(?:threadIdx|blockIdx)\b", expression):
            thread_variables.add(declaration.spelling)
    thread_dependent = bool(
        re.search(r"\b(?:threadIdx|blockIdx)\b", condition)
        or any(re.search(rf"\b{re.escape(name)}\b", condition) for name in thread_variables)
        or any(
            re.search(rf"(?:\*\s*{re.escape(name)}\b|\b{re.escape(name)}\s*\[)", condition)
            for name in pointer_parameters
        )
    )
    return {
        "condition": condition,
        "loop_init": loop_init,
        "loop_condition": condition if cursor.kind == cursor_kind.FOR_STMT else "",
        "loop_update": loop_update,
        "thread_dependent": thread_dependent,
        "contains_barrier": any(name.startswith("__syncthreads") for name in calls),
        "has_early_exit": has_early_exit,
    }


def _statement_array_accesses(cursor, cursor_kind):
    accesses = []
    writes = set()
    for operation in _walk(cursor):
        if operation.kind not in {
            cursor_kind.BINARY_OPERATOR,
            cursor_kind.COMPOUND_ASSIGNMENT_OPERATOR,
        }:
            continue
        operands = list(operation.get_children())
        if not operands:
            continue
        lhs = operands[0]
        if lhs.kind != cursor_kind.ARRAY_SUBSCRIPT_EXPR:
            continue
        lhs_operands = list(lhs.get_children())
        if lhs_operands and (lhs_operands[0].spelling or _tokens(lhs_operands[0])):
            writes.add(lhs_operands[0].spelling or _tokens(lhs_operands[0]))
    for child in _walk(cursor):
        if child.kind != cursor_kind.ARRAY_SUBSCRIPT_EXPR:
            continue
        operands = list(child.get_children())
        if len(operands) < 2:
            continue
        base = operands[0].spelling or _tokens(operands[0])
        index = _tokens(operands[1])
        if base:
            accesses.append(
                {"array": base, "index": index, "mode": "write" if base in writes else "read"}
            )
    return accesses


def _statement_pointer_assignments(cursor, cursor_kind):
    assignments = []
    for operation in _walk(cursor):
        if operation.kind not in {
            cursor_kind.BINARY_OPERATOR,
            cursor_kind.COMPOUND_ASSIGNMENT_OPERATOR,
        }:
            continue
        tokens = [token.spelling for token in operation.get_tokens()]
        if "=" not in tokens:
            continue
        operands = list(operation.get_children())
        if len(operands) < 2:
            continue
        lhs = operands[0]
        target = lhs.spelling if lhs.kind == cursor_kind.DECL_REF_EXPR else ""
        if not target:
            continue
        sources = tuple(
            reference.referenced.spelling
            for reference in _walk(operands[1])
            if reference.kind == cursor_kind.DECL_REF_EXPR
            and reference.referenced is not None
            and reference.referenced.spelling
        )
        assignments.append({"target": target, "sources": list(dict.fromkeys(sources))})
    return assignments


def _statement_pointer_casts(cursor, cursor_kind):
    casts = []
    cast_kinds = {
        getattr(cursor_kind, "CSTYLE_CAST_EXPR", None),
        getattr(cursor_kind, "CXX_STATIC_CAST_EXPR", None),
        getattr(cursor_kind, "CXX_REINTERPRET_CAST_EXPR", None),
        getattr(cursor_kind, "CXX_CONST_CAST_EXPR", None),
    }
    for operation in _walk(cursor):
        if operation.kind not in cast_kinds or "*" not in operation.type.get_canonical().spelling:
            continue
        sources = tuple(
            reference.referenced.spelling
            for reference in _walk(operation)
            if reference.kind == cursor_kind.DECL_REF_EXPR
            and reference.referenced is not None
            and reference.referenced.spelling
        )
        casts.append(
            {
                "target_type": operation.type.get_canonical().spelling,
                "sources": list(dict.fromkeys(sources)),
            }
        )
    return casts


def _statement_non_atomic_targets(cursor, cursor_kind):
    targets = []
    for operation in _walk(cursor):
        if operation.kind not in {
            cursor_kind.UNARY_OPERATOR,
            cursor_kind.COMPOUND_ASSIGNMENT_OPERATOR,
        }:
            continue
        tokens = [token.spelling for token in operation.get_tokens()]
        if operation.kind == cursor_kind.COMPOUND_ASSIGNMENT_OPERATOR or any(
            token in {"++", "--"} for token in tokens
        ):
            for reference in _walk(operation):
                if (
                    reference.kind == cursor_kind.DECL_REF_EXPR
                    and reference.referenced is not None
                    and reference.referenced.spelling
                ):
                    targets.append(reference.referenced.spelling)
    return list(dict.fromkeys(targets))


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
        text = read_text(path)
        for match in pattern.finditer(text):
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
                SourceLocationIR(path.as_posix(), text.count("\n", 0, match.start()) + 1),
            )
    return macros


def _declared_type_locations(source_paths):
    locations = {}
    patterns = [
        re.compile(
            r"\btypedef\s+struct(?:\s+[A-Za-z_]\w*)?\s*\{.*?\}\s*([A-Za-z_]\w*)\s*;", re.DOTALL
        ),
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
                    text[: match.start()].count("\n") + 1,
                    1,
                )
    return locations


def _tokens(cursor):
    return " ".join(token.spelling for token in cursor.get_tokens()).strip()


def _original_call_location(function, callee_name, ordinal):
    matches = list(re.finditer(rf"\b{re.escape(callee_name)}\s*\(", function.body))
    if ordinal < len(matches):
        match = matches[ordinal]
        prefix = function.body[: match.start()]
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
    existing = {(call.callee_name, call.source.line, call.source.column) for call in calls}
    excluded = {
        "if",
        "for",
        "while",
        "switch",
        "sizeof",
        "alignof",
        "decltype",
        "static_cast",
        "reinterpret_cast",
        "const_cast",
        "dynamic_cast",
        "return",
        "new",
        "delete",
        "throw",
        *types.keys(),
        *macros.keys(),
    }
    names = list(
        dict.fromkeys(
            match.group(1)
            for match in re.finditer(r"\b([A-Za-z_]\w*)\s*\(", function.body)
            if match.group(1) not in excluded
        )
    )
    functions_by_name = {}
    for candidate in functions.values():
        functions_by_name.setdefault(candidate.name, []).append(candidate)

    for name in names:
        for ordinal, (_offset, argument_text) in enumerate(
            iter_function_calls(function.body, name)
        ):
            arguments = tuple(part for part in split_top_level(argument_text) if part.strip())
            source = _original_call_location(function, name, ordinal)
            if (name, source.line, source.column) in existing:
                continue
            candidates = [
                candidate
                for candidate in functions_by_name.get(name, [])
                if len(candidate.parameters) == len(arguments)
            ]
            callee_id = candidates[0].symbol_id if len(candidates) == 1 else None
            calls.append(
                CallEdgeIR(
                    caller_id=function.symbol_id,
                    callee_name=name,
                    callee_id=callee_id,
                    arguments=arguments,
                    source=source,
                )
            )
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
