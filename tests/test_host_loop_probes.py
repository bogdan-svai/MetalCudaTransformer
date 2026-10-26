"""Feasibility evidence, not an auto-parallelization legality checker."""

from pathlib import Path

import pytest
from clang.cindex import Config, CursorKind, Index, TranslationUnit

from core.frontend.clang_probe import _configure
from core.frontend.semantic import parse_semantic_project

CORPUS = Path(__file__).parent / "fixtures" / "host_loops" / "loops.cpp"


@pytest.fixture
def host_unit():
    assert _configure(Config), "Host probes require libclang"
    unit = Index.create().parse(
        str(CORPUS.resolve()),
        args=["-x", "c++", "-std=c++17"],
        options=TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD,
    )
    assert not [str(d) for d in unit.diagnostics if d.severity >= d.Error]
    return unit


@pytest.mark.parametrize(
    "name,accesses,calls",
    [
        ("map", 2, 0),
        ("combine", 3, 0),
        ("overlap", 2, 0),
        ("prefix", 2, 0),
        ("effects", 0, 1),
        ("wrap", 1, 0),
        ("empty", 1, 0),
        ("macro_loop", 1, 0),
    ],
)
def test_original_host_loop_structure(host_unit, name, accesses, calls):
    function = next(c for c in host_unit.cursor.get_children() if c.spelling == name)
    assert function.get_usr()
    nodes = list(function.walk_preorder())
    (loop,) = [c for c in nodes if c.kind == CursorKind.FOR_STMT]
    children = list(loop.get_children())
    assert children[0].kind == CursorKind.DECL_STMT
    assert children[1].kind == CursorKind.BINARY_OPERATOR
    assert children[2].kind == CursorKind.UNARY_OPERATOR
    assert sum(c.kind == CursorKind.ARRAY_SUBSCRIPT_EXPR for c in nodes) == accesses
    assert sum(c.kind == CursorKind.CALL_EXPR for c in nodes) == calls
    (iterator,) = [c for c in children[0].get_children() if c.kind == CursorKind.VAR_DECL]
    assert iterator.type.get_canonical().spelling == "unsigned int"
    assert iterator.get_usr()
    references = [c for c in nodes if c.kind == CursorKind.DECL_REF_EXPR and c.spelling == "i"]
    assert references
    assert all(c.referenced.get_usr() == iterator.get_usr() for c in references)


def test_direct_loop_range_uses_original_utf8_bytes(host_unit):
    function = next(c for c in host_unit.cursor.get_children() if c.spelling == "map")
    loop = next(c for c in function.walk_preorder() if c.kind == CursorKind.FOR_STMT)
    data = CORPUS.read_bytes()
    start, end = loop.extent.start, loop.extent.end
    assert str(start.file) == str(CORPUS.resolve())
    assert (
        data[start.offset : end.offset]
        == b"for (unsigned i = 0; i < n; ++i) out[i] = in[i] * 3u + 1u"
    )
    assert start.offset != CORPUS.read_text().index("for (")


def test_macro_loop_cannot_be_rewritten_as_direct_for(host_unit):
    function = next(c for c in host_unit.cursor.get_children() if c.spelling == "macro_loop")
    loop = next(c for c in function.walk_preorder() if c.kind == CursorKind.FOR_STMT)
    source = CORPUS.read_bytes()[loop.extent.start.offset : loop.extent.end.offset]
    assert source.startswith(b"LOOP(n)")  # Extent is invocation, not expanded syntax.
    assert any(
        c.kind == CursorKind.MACRO_INSTANTIATION and c.spelling == "LOOP"
        for c in host_unit.cursor.get_children()
    )


def test_external_host_call_identity_survives_separate_units(tmp_path):
    assert _configure(Config)
    header = tmp_path / "helper.h"
    header.write_text("unsigned helper(unsigned);\n")
    definition = tmp_path / "helper.cpp"
    definition.write_text('#include "helper.h"\nunsigned helper(unsigned x) { return x + 1u; }\n')
    caller = tmp_path / "caller.cpp"
    caller.write_text(
        '#include "helper.h"\nvoid map(unsigned* a, unsigned n) { for(unsigned i=0;i<n;++i) a[i]=helper(a[i]); }\n'
    )
    units = [Index.create().parse(str(p), args=["-std=c++17"]) for p in (definition, caller)]
    for unit in units:
        assert not [str(d) for d in unit.diagnostics if d.severity >= d.Error]
    helper = next(
        c for c in units[0].cursor.get_children() if c.spelling == "helper" and c.is_definition()
    )
    call = next(c for c in units[1].cursor.walk_preorder() if c.kind == CursorKind.CALL_EXPR)
    assert call.referenced.get_usr() == helper.get_usr()
    assert str(call.referenced.location.file) == str(header)
    assert str(call.location.file) == str(caller)


def test_existing_semantic_entry_point_skips_host_only_metadata():
    result = parse_semantic_project(
        {
            "sources": [str(CORPUS)],
            "functions": [{"name": "map", "kind": "HOST_FUNC", "file": str(CORPUS)}],
        }
    )
    assert result.complete
    assert result.project is None  # Completeness here means no device work, not host proof.
