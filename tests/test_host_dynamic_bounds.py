import pytest
from core.frontend.host_loops import analyze_host_loops
from core.passes.host_guards import evaluate_guards


def loop_report(tmp_path, parameter="unsigned n", body="b[i]=a[i]+1u;"):
    source = tmp_path / "map.cpp"
    source.write_text(
        f"void f({parameter}) {{ unsigned a[8]={{}}; unsigned b[6]; "
        f"for(unsigned i=0;i<n;++i) {{ {body} }} }}"
    )
    report = analyze_host_loops({"sources": [str(source)]})
    assert report["complete"], report["diagnostics"]
    return report["loops"][0]


def test_dynamic_bound_is_conditional_not_unconditionally_eligible(tmp_path):
    loop = loop_report(tmp_path)
    assert loop["status"] == "conditional", loop["reasons"]
    proof = loop["proof"]
    assert not proof["eligible"]
    assert proof["conditional"]
    assert {g["maximum"] for g in proof["runtime_guards"]} == {6}
    assert loop["execution"] == "cpu"
    assert loop["estimated_iterations"] is None
    symbol = proof["runtime_guards"][0]["symbol_usr"]
    for size in (0, 1, 6):
        assert evaluate_guards(proof["runtime_guards"], {symbol: size})
    for size in (-1, 7, 8, 4294967295, 4294967296):
        assert not evaluate_guards(proof["runtime_guards"], {symbol: size})
    assert not evaluate_guards(proof["runtime_guards"], {})


@pytest.mark.parametrize("parameter", ["int n", "volatile unsigned n", "unsigned long n"])
def test_nonmatching_parameter_types_are_refused(tmp_path, parameter):
    loop = loop_report(tmp_path, parameter=parameter)
    assert loop["status"] == "unproven"
    assert "LoopBoundsUnsupported" in loop["reasons"]


@pytest.mark.parametrize("body", ["b[i]=a[i]; ++n;", "b[i]=a[i-1];", "b[0]=i;"])
def test_dynamic_bound_does_not_bypass_dependence_checks(tmp_path, body):
    loop = loop_report(tmp_path, body=body)
    assert loop["status"] == "unproven"
    assert not loop["proof"]["conditional"]


@pytest.mark.parametrize("value", [None, True, 3.0, "3"])
def test_guard_invalid_capture_is_refused(value):
    assert not evaluate_guards(
        [{"kind": "unsigned_le_constant", "symbol_usr": "n", "maximum": 8}], {"n": value}
    )


def test_unknown_guard_is_refused():
    assert not evaluate_guards([{"kind": "unknown"}], {})


def test_pointer_parameters_still_need_extents_and_alias_proof(tmp_path):
    source = tmp_path / "pointer.cpp"
    source.write_text(
        "void f(unsigned* a, unsigned* b, unsigned n) { for(unsigned i=0;i<n;++i) b[i]=a[i]; }"
    )
    (loop,) = analyze_host_loops({"sources": [str(source)]})["loops"]
    assert loop["status"] == "unproven"
    assert "BufferExtentOrAliasProofMissing" in loop["reasons"]
