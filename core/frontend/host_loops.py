"""Report host loops from original Clang units, without authorizing transforms."""

import hashlib
import tempfile
from pathlib import Path

from core.frontend.clang_probe import _configure
from core.frontend.semantic import _clang_args
from core.ir.host_loops import HOST_LOOP_SCHEMA, HostLoopIR, HostNodeIR


def _operator(cursor, children):
    if (
        cursor.kind.name in {"BINARY_OPERATOR", "COMPOUND_ASSIGNMENT_OPERATOR"}
        and len(children) == 2
    ):
        between = [
            t.spelling
            for t in cursor.get_tokens()
            if children[0].extent.end.offset <= t.extent.start.offset
            and t.extent.end.offset <= children[1].extent.start.offset
        ]
        return between[0] if len(between) == 1 else None
    if cursor.kind.name == "UNARY_OPERATOR" and len(children) == 1:
        outside = [
            t.spelling
            for t in cursor.get_tokens()
            if t.extent.end.offset <= children[0].extent.start.offset
            or t.extent.start.offset >= children[0].extent.end.offset
        ]
        return outside[0] if len(outside) == 1 else None
    return None


def _node(cursor, cache=None):
    cache = {} if cache is None else cache
    if cursor in cache:
        return cache[cursor]
    start, end = cursor.extent.start, cursor.extent.end
    referenced = cursor.referenced
    children = list(cursor.get_children())
    value = None
    if cursor.kind.name == "INTEGER_LITERAL":
        tokens = [t.spelling for t in cursor.get_tokens()]
        if len(tokens) == 1:
            number = tokens[0].rstrip("uUlL")
            if number.isdecimal() and (number == "0" or not number.startswith("0")):
                value = int(number)
    extent = cursor.type.get_array_size()

    result = HostNodeIR(
        kind=cursor.kind.name,
        type=cursor.type.get_canonical().spelling,
        file=str(start.file) if start.file else "",
        start=start.offset,
        end=end.offset,
        spelling=cursor.spelling,
        symbol_usr=referenced.get_usr() if referenced else cursor.get_usr() or None,
        operator=_operator(cursor, children),
        integer_value=value,
        array_extent=extent if extent >= 0 else None,
        storage=cursor.storage_class.name,
        initialized=(
            any(child.kind.name == "INIT_LIST_EXPR" for child in children)
            if cursor.kind.name == "VAR_DECL"
            else False
        ),
        tokens=[token.spelling for token in cursor.get_tokens()],
        children=[_node(child, cache) for child in children],
    )
    cache[cursor] = result
    return result


def analyze_host_loops(metadata):
    from clang.cindex import Config, CursorKind, Index, TranslationUnit

    report = {
        "schema_version": HOST_LOOP_SCHEMA,
        "mode": "report",
        "complete": True,
        "loops": [],
        "diagnostics": [],
        "transformed_loop_count": 0,
        "cached": False,
    }
    if not _configure(Config):
        report["complete"] = False
        report["diagnostics"].append(
            {
                "severity": "error",
                "code": "HostFrontendUnavailable",
                "message": "Native libclang was not found.",
            }
        )
        return report
    paths = [Path(p).resolve() for p in metadata.get("sources", [])]
    allowed = {str(p) for p in paths}
    source_contents = {str(p): p.read_bytes() for p in paths}
    snapshots = [(name, data.decode("utf-8")) for name, data in source_contents.items()]
    with tempfile.TemporaryDirectory(prefix="mcu-host-analysis-") as directory:
        header = Path(directory) / "attributes.h"
        header.write_text(
            "#define __host__ __attribute__((host))\n"
            "#define __device__ __attribute__((device))\n"
            "#define __global__ __attribute__((global))\n"
        )
        for path in paths:
            if path.suffix not in {".cpp", ".cc", ".cxx", ".cu", ".c"}:
                continue
            args = _clang_args(paths, metadata)
            if path.suffix == ".c":
                args[:3] = ["-x", "c", "-std=c11"]
            elif path.suffix == ".cu":
                args[1] = "cuda"
                args += ["--cuda-host-only", "-nocudainc", "-nocudalib", "-include", str(header)]
            unit = Index.create().parse(
                str(path),
                args=args,
                unsaved_files=snapshots,
                options=TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD,
            )
            errors = [d for d in unit.diagnostics if d.severity >= d.Error]
            for error in errors:
                report["diagnostics"].append(
                    {
                        "severity": "error",
                        "code": "HostFrontendDiagnostic",
                        "message": str(error),
                        "file": str(error.location.file),
                        "line": error.location.line,
                        "column": error.location.column,
                    }
                )
            if errors:
                report["complete"] = False
            macros = [
                c for c in unit.cursor.get_children() if c.kind == CursorKind.MACRO_INSTANTIATION
            ]
            for function in unit.cursor.walk_preorder():
                if (
                    function.kind not in {CursorKind.FUNCTION_DECL, CursorKind.CXX_METHOD}
                    or not function.is_definition()
                ):
                    continue
                if not function.location.file or str(function.location.file) not in allowed:
                    continue
                attrs = {c.kind.name for c in function.get_children()}
                if "CUDAGLOBAL_ATTR" in attrs or (
                    "CUDADEVICE_ATTR" in attrs and "CUDAHOST_ATTR" not in attrs
                ):
                    continue
                node_cache = {}
                variable_cursors = [
                    c
                    for c in function.walk_preorder()
                    if c.kind in {CursorKind.VAR_DECL, CursorKind.PARM_DECL}
                ]
                for loop in function.walk_preorder():
                    if loop.kind not in {
                        CursorKind.FOR_STMT,
                        CursorKind.WHILE_STMT,
                        CursorKind.DO_STMT,
                    }:
                        continue
                    children = list(loop.get_children())
                    reasons = [
                        "DependenceProofMissing",
                        "BufferExtentProofMissing",
                        "ProfitabilityNotMeasured",
                    ]
                    if errors:
                        reasons.append("TranslationUnitHasErrors")
                    macro = any(
                        str(c.location.file) == str(loop.location.file)
                        and loop.extent.start.offset <= c.location.offset < loop.extent.end.offset
                        for c in macros
                    )
                    if macro:
                        reasons.append("MacroRangeRequiresMapping")
                    # Only a complete four-part original for can expose header roles here.
                    full = loop.kind == CursorKind.FOR_STMT and len(children) == 4 and not macro
                    if not full:
                        reasons.append("LoopHeaderRolesUnavailable")
                    if any(c.kind == CursorKind.CALL_EXPR for c in loop.walk_preorder()):
                        reasons.append("CallEffectProofMissing")
                    node = _node(loop, node_cache)
                    parts = node.children if full else [None] * 4
                    record = HostLoopIR(
                        function.get_usr(),
                        function.spelling,
                        str(path),
                        loop.location.line,
                        loop.location.column,
                        node,
                        *parts,
                        reasons,
                    )
                    from core.passes.host_parallel import prove_loop

                    declarations = {
                        c.get_usr(): _node(c, node_cache)
                        for c in variable_cursors
                        if c.extent.end.offset <= loop.extent.start.offset
                    }
                    record.source_sha256 = hashlib.sha256(source_contents[node.file]).hexdigest()
                    proof = prove_loop(record, declarations)
                    record.reasons = list(
                        dict.fromkeys(proof["reasons"] + ["ProfitabilityNotMeasured"])
                    )
                    item = record.to_dict()
                    item.update(
                        {
                            "status": (
                                "eligible"
                                if proof["eligible"]
                                else "conditional" if proof["conditional"] else "unproven"
                            ),
                            "proof": proof,
                            "execution": "cpu",
                            "estimated_iterations": proof["iterations"],
                            "transfer_bytes": None,
                        }
                    )
                    report["loops"].append(item)
    return report
