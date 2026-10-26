"""Collect call locations from original translation units during frontend migration.

Compatibility declarations deliberately retain the current frontend's erased CUDA
attributes. This pass supplies source identity/ranges, not CUDA legality checks.
"""

import tempfile
from dataclasses import dataclass
from pathlib import Path

from core.frontend.clang_probe import CUDA_STUBS
from core.ir.symbols import SourceRangeIR, canonical_symbol_id


@dataclass
class OriginalFunction:
    cursor: object
    body_cursor: object
    body_range: SourceRangeIR
    body: str
    translation_unit: object  # Retain ownership of cursors and their tokens.
    has_body_errors: bool
    compatibility_header: str


def collect_original_functions(source_paths, clang_args, parameter_type):
    from clang.cindex import CursorKind, Index, TranslationUnit

    originals = {}
    allowed = {str(path.resolve()) for path in source_paths}
    contents = {str(path.resolve()): path.read_bytes() for path in source_paths}
    with tempfile.TemporaryDirectory(prefix="mcu-call-locations-") as directory:
        root = Path(directory)
        header = root / "compat.h"
        header.write_text(
            "#pragma once\n#include <stdint.h>\n#include <stddef.h>\n#include <string.h>\n"
            + CUDA_STUBS
            + "\nint cudaConfigureCall(dim3, dim3, unsigned long = 0, void* = nullptr);\n",
            encoding="utf-8",
        )
        for name in ("cuda_runtime.h", "cuda.h", "cublas_v2.h"):
            (root / name).write_text('#include "compat.h"\n', encoding="utf-8")
        args = list(clang_args)
        args[1] = "cuda"
        args.extend(
            [
                "--cuda-host-only",
                "-nocudainc",
                "-nocudalib",
                "-ferror-limit=0",
                "-I",
                directory,
                "-include",
                str(header),
            ]
        )
        index = Index.create()
        units = [p for p in source_paths if p.suffix in {".cu", ".cpp", ".cc", ".cxx"}]
        for path in units:
            tu = index.parse(
                str(path.resolve()),
                args=args,
                options=TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD,
            )
            pending = list(tu.cursor.get_children())
            while pending:
                cursor = pending.pop()
                file = (
                    str(Path(str(cursor.location.file)).resolve()) if cursor.location.file else ""
                )
                if file not in allowed:
                    continue
                if cursor.kind != CursorKind.FUNCTION_DECL:
                    if cursor.kind in {CursorKind.NAMESPACE, CursorKind.LINKAGE_SPEC}:
                        pending.extend(cursor.get_children())
                    continue
                if not cursor.is_definition():
                    continue
                bodies = [c for c in cursor.get_children() if c.kind == CursorKind.COMPOUND_STMT]
                if len(bodies) != 1:
                    continue
                body = bodies[0]
                start, end = body.extent.start.offset + 1, body.extent.end.offset - 1
                data = contents[file]
                if data[start - 1 : start] != b"{" or data[end : end + 1] != b"}":
                    continue
                symbol = canonical_symbol_id(
                    cursor.spelling,
                    [parameter_type(a.type).canonical for a in cursor.get_arguments()],
                )
                key = (file, symbol)
                record = OriginalFunction(
                    cursor,
                    body,
                    SourceRangeIR(file, start, end),
                    data[start:end].decode("utf-8"),
                    tu,
                    any(
                        d.severity >= d.Error
                        and d.location.file
                        and str(Path(str(d.location.file)).resolve()) == file
                        and start <= d.location.offset <= end
                        for d in tu.diagnostics
                    ),
                    str(header),
                )
                # Definitions in headers may occur in several units. Never silently
                # select differing preprocessor interpretations of the same function.
                if key in originals:
                    previous = originals[key]
                    if (
                        previous is None
                        or previous.cursor.get_usr() != cursor.get_usr()
                        or previous.body_range != record.body_range
                        or _call_signature(previous.body_cursor) != _call_signature(body)
                    ):
                        originals[key] = None
                else:
                    originals[key] = record
    return originals


def collect_original_macros(source_paths, clang_args):
    """Collect macro definitions from each original preprocessing context."""
    from clang.cindex import CursorKind, Index, TranslationUnit
    from core.ir.symbols import MacroIR, SourceLocationIR

    allowed = {str(path.resolve()) for path in source_paths}
    macros = {}
    args = list(clang_args)
    args[1] = "cuda"
    with tempfile.TemporaryDirectory(prefix="mcu-macro-locations-") as directory:
        header = Path(directory) / "compat.h"
        header.write_text(
            "#pragma once\n#include <stdint.h>\n#include <stddef.h>\n" + CUDA_STUBS,
            encoding="utf-8",
        )
        args.extend(
            [
                "--cuda-host-only",
                "-nocudainc",
                "-nocudalib",
                "-ferror-limit=0",
                "-I",
                directory,
                "-include",
                str(header),
            ]
        )
        index = Index.create()
        for path in source_paths:
            if path.suffix not in {".cu", ".cpp", ".cc", ".cxx", ".h", ".hpp"}:
                continue
            tu = index.parse(
                str(path.resolve()),
                args=args,
                options=TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD,
            )
            for cursor in tu.cursor.walk_preorder():
                if cursor.kind != CursorKind.MACRO_DEFINITION or not cursor.location.file:
                    continue
                file = str(Path(str(cursor.location.file)).resolve())
                if file not in allowed:
                    continue
                tokens = [token.spelling for token in cursor.get_tokens()]
                if len(tokens) < 3 or tokens[0] != "#" or tokens[1] != "define":
                    continue
                name = cursor.spelling
                if name.startswith("__") or name in {"DEBUG_PRINT", "GPU_COMPATIBLE"}:
                    continue
                index_name = 2
                parameters = []
                if len(tokens) > 3 and tokens[3] == "(":
                    close = next((i for i in range(4, len(tokens)) if tokens[i] == ")"), None)
                    if close is None:
                        continue
                    parameters = [item for item in tokens[4:close] if item != ","]
                    index_name = close + 1
                expression = " ".join(tokens[index_name:]).strip()
                macros[name] = MacroIR(
                    name,
                    tuple(parameters),
                    expression,
                    SourceLocationIR(file, cursor.location.line, cursor.location.column),
                )
    return macros


def collect_original_type_locations(source_paths, clang_args):
    """Index type declaration locations from original translation units."""
    from clang.cindex import CursorKind, Index, TranslationUnit
    from core.ir.symbols import SourceLocationIR

    allowed = {str(path.resolve()) for path in source_paths}
    locations = {}
    args = list(clang_args)
    args[1] = "cuda"
    with tempfile.TemporaryDirectory(prefix="mcu-type-locations-") as directory:
        header = Path(directory) / "compat.h"
        header.write_text(
            "#pragma once\n#include <stdint.h>\n#include <stddef.h>\n" + CUDA_STUBS,
            encoding="utf-8",
        )
        args.extend(
            [
                "--cuda-host-only",
                "-nocudainc",
                "-nocudalib",
                "-ferror-limit=0",
                "-I",
                directory,
                "-include",
                str(header),
            ]
        )
        index = Index.create()
        for path in source_paths:
            if path.suffix not in {".cu", ".cpp", ".cc", ".cxx", ".h", ".hpp"}:
                continue
            tu = index.parse(
                str(path.resolve()),
                args=args,
                options=TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD,
            )
            for cursor in tu.cursor.walk_preorder():
                if (
                    cursor.kind
                    not in {CursorKind.STRUCT_DECL, CursorKind.ENUM_DECL, CursorKind.TYPEDEF_DECL}
                    or not cursor.spelling
                    or not cursor.location.file
                ):
                    continue
                file = str(Path(str(cursor.location.file)).resolve())
                if file not in allowed or cursor.spelling == "dim3":
                    continue
                locations[cursor.spelling] = SourceLocationIR(
                    file, cursor.location.line or None, cursor.location.column or None
                )
    return locations


def _call_signature(body):
    from clang.cindex import CursorKind

    return [
        (c.location.offset, c.referenced.get_usr() if c.referenced else None)
        for c in body.walk_preorder()
        if c.kind == CursorKind.CALL_EXPR
    ]


def direct_call_range(call, body_range, body):
    """Reject macro expansions, member calls and qualified references for now."""
    from clang.cindex import CursorKind

    referenced = call.referenced
    if referenced is None or referenced.kind != CursorKind.FUNCTION_DECL:
        return None, "The call does not reference a free function declaration."
    # Only the first child describes the callee; nested argument calls must not
    # be mistaken for the outer call's identifier.
    children = list(call.get_children())
    references = (
        []
        if not children
        else [
            c
            for c in children[0].walk_preorder()
            if c.kind == CursorKind.DECL_REF_EXPR
            and c.referenced
            and c.referenced.get_usr() == referenced.get_usr()
        ]
    )
    if len(references) != 1:
        return None, "No unique direct callee reference exists."
    ref = references[0]
    start, end = ref.extent.start, ref.extent.end
    if (
        not start.file
        or not end.file
        or (
            str(Path(str(start.file)).resolve()) != body_range.file
            or str(Path(str(end.file)).resolve()) != body_range.file
        )
    ):
        return None, "The callee reference is outside the original source file."
    begin, finish = start.offset - body_range.start, end.offset - body_range.start
    expected = referenced.spelling.encode("utf-8")
    if not (0 <= begin < finish <= len(body)) or body[begin:finish] != expected:
        return None, "Macro or qualified callee has no directly editable identifier range."
    # A macro may expand to the same spelling. Token origin must match as well.
    tokens = list(ref.get_tokens())
    if len(tokens) != 1 or tokens[0].extent.start.offset != start.offset:
        return None, "The callee token comes from a macro expansion."
    return SourceRangeIR(body_range.file, start.offset, end.offset), None


def original_call_extent(call, body_range, body):
    """Validate AST call boundaries without rediscovering parentheses in text."""
    start, end = call.extent.start, call.extent.end
    if not start.file or not end.file:
        return None
    if any(str(Path(str(p.file)).resolve()) != body_range.file for p in (start, end)):
        return None
    begin, finish = start.offset - body_range.start, end.offset - body_range.start
    if not 0 <= begin < finish <= len(body) or body[finish - 1 : finish] != b")":
        return None
    return SourceRangeIR(body_range.file, start.offset, end.offset)


def is_intrinsic_declaration(cursor, compatibility_header):
    """Only compatibility declarations and system memset can authorize lowering."""
    from clang.cindex import CursorKind

    if cursor is None or cursor.kind != CursorKind.FUNCTION_DECL:
        return False
    declaration = cursor.canonical
    location = declaration.location
    if not location.file:
        return False
    if str(Path(str(location.file)).resolve()) == str(Path(compatibility_header).resolve()):
        return True
    parent = declaration.semantic_parent
    while parent.kind == CursorKind.LINKAGE_SPEC:
        parent = parent.semantic_parent
    return (
        cursor.spelling == "memset"
        and location.is_in_system_header
        and parent.kind == CursorKind.TRANSLATION_UNIT
    )
