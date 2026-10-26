import pytest

from types import SimpleNamespace

from core.backends.device import _apply_call_edits
from core.ir.symbols import CallEdgeIR, SourceLocationIR, SourceRangeIR


def _edit_selected_calls(source, offsets, targets):
    """Supply explicit IR ranges; symbol discovery is tested in integration tests."""
    function = SimpleNamespace(body=source, body_range=SourceRangeIR("case.cu", 0, len(source)))
    edits = []
    for index, offset in enumerate(offsets):
        call = CallEdgeIR(
            "caller",
            "helper",
            "helper(int)",
            (),
            SourceLocationIR("case.cu"),
            callee_usr="helper-usr",
            identifier_range=SourceRangeIR("case.cu", offset, offset + 6),
        )
        edits.append((call, targets[min(index, len(targets) - 1)]))
    return _apply_call_edits(source, function, edits)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("int helper_count = 0; helper(x);", "int helper_count = 0; renamed(x);"),
        ("otherhelper(x); helper(x);", "otherhelper(x); renamed(x);"),
        ("helper2(x); helper(x);", "helper2(x); renamed(x);"),
        ("auto ptr = helper; helper(x);", "auto ptr = helper; renamed(x);"),
        ("helper_count += 1;", "helper_count += 1;"),
        ("helper(x); helper_count += 1;", "renamed(x); helper_count += 1;"),
        ("helper(helper(x));", "renamed(renamed(x));"),
        ("{ int helper_count = 1; helper(x); }", "{ int helper_count = 1; renamed(x); }"),
    ],
)
def test_call_rewrite_preserves_unrelated_source(source, expected):
    offsets = (
        [0, 7]
        if source == "helper(helper(x));"
        else ([source.rindex("helper(")] if "helper(" in source else [])
    )
    assert _edit_selected_calls(source, offsets, ["renamed"]) == expected


def test_skipped_identifiers_do_not_consume_call_targets():
    source = "helper_count++; helper(a); auto f = helper; helper(b);"
    assert _edit_selected_calls(
        source, [source.index("helper(a)"), source.index("helper(b)")], ["first", "second"]
    ) == ("helper_count++; first(a); auto f = helper; second(b);")


@pytest.mark.parametrize(
    "source",
    [
        "// helper(x)\nhelper(x);",
        'const char* text = "helper(x)"; helper(x);',
        "object.helper(x); helper(x);",
        "object->helper(x); helper(x);",
        "namespace::helper(x); helper(x);",
    ],
)
def test_call_rewrite_ignores_comments_literals_and_member_calls(source):
    rewritten = _edit_selected_calls(source, [source.rindex("helper(")], ["renamed"])
    assert rewritten.count("renamed(") == 1
    assert (
        "object.helper(x)" in rewritten
        or "object->helper(x)" in rewritten
        or "namespace::helper(x)" in rewritten
        or "// helper(x)" in rewritten
        or '"helper(x)"' in rewritten
    )
