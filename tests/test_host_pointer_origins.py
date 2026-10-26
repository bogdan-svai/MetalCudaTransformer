import pytest
from core.frontend.host_loops import analyze_host_loops
from core.passes.host_guards import evaluate_guards


def analyze(tmp_path, declarations, body, bound="4", parameter=""):
    source = tmp_path / "map.cpp"
    source.write_text(
        f"void f({parameter}) {{ unsigned a[16]={{}}; unsigned b[16]; "
        f"{declarations} for(unsigned i=0;i<{bound};++i) {{ {body} }} }}"
    )
    report = analyze_host_loops({"sources": [str(source)]})
    assert report["complete"], report["diagnostics"]
    return report["loops"][0]


def test_array_decay_between_disjoint_objects(tmp_path):
    loop = analyze(tmp_path, "unsigned* const p=a; unsigned* const q=b;", "q[i]=p[i]+1u;")
    assert loop["status"] == "eligible", loop["reasons"]
    accesses = loop["proof"]["accesses"]
    assert accesses[0]["origin_usr"] != accesses[1]["origin_usr"]


def test_pointer_chain_retains_origin_offset_and_available_extent(tmp_path):
    loop = analyze(tmp_path, "unsigned* const p=a+2; unsigned* const q=p+3;", "q[i]=i;")
    assert loop["status"] == "eligible"
    (access,) = loop["proof"]["accesses"]
    assert access["offset_elements"] == 5
    assert access["extent"] == 11


def test_same_mapping_alias_is_per_iteration(tmp_path):
    loop = analyze(tmp_path, "unsigned* const p=a; unsigned* const q=a;", "q[i]=p[i]+1u;")
    assert loop["status"] == "eligible"


def test_disjoint_regions_on_same_object(tmp_path):
    loop = analyze(tmp_path, "unsigned* const p=a; unsigned* const q=a+8;", "q[i]=p[i]+1u;")
    assert loop["status"] == "eligible"


def test_shifted_overlap_has_cross_iteration_dependence(tmp_path):
    loop = analyze(tmp_path, "unsigned* const p=a; unsigned* const q=a+1;", "q[i]=p[i];")
    assert loop["status"] == "unproven"
    assert "CrossIterationAliasOverlap" in loop["reasons"]


def test_dynamic_alias_guard_limits_extent_and_overlap(tmp_path):
    loop = analyze(
        tmp_path, "unsigned* const p=a; unsigned* const q=a+3;", "q[i]=p[i];", "n", "unsigned n"
    )
    assert loop["status"] == "conditional"
    guards = loop["proof"]["runtime_guards"]
    (guard,) = guards
    assert guard["maximum"] == 3
    assert evaluate_guards(guards, {guard["symbol_usr"]: 3})
    assert not evaluate_guards(guards, {guard["symbol_usr"]: 4})


@pytest.mark.parametrize(
    "declarations,reason",
    [
        ("unsigned* p=a;", "BufferExtentOrAliasProofMissing"),
        ("unsigned* p=b; p=a;", "BufferExtentOrAliasProofMissing"),
        ("unsigned* const p=a+17;", "BufferExtentOrAliasProofMissing"),
        ("unsigned* const p=a+16;", "BufferBoundsExceeded"),
        ("unsigned* const p=a-1;", "BufferExtentOrAliasProofMissing"),
    ],
)
def test_unsupported_or_out_of_bounds_origins_stay_cpu(tmp_path, declarations, reason):
    loop = analyze(tmp_path, declarations, "p[i]=i;")
    assert loop["status"] == "unproven"
    assert reason in loop["reasons"]


def test_readonly_shifted_aliases_can_overlap(tmp_path):
    loop = analyze(tmp_path, "unsigned* const p=a; unsigned* const q=a+1;", "b[i]=p[i]+q[i];")
    assert loop["status"] == "eligible"


def test_one_past_pointer_is_valid_only_for_empty_loop(tmp_path):
    loop = analyze(tmp_path, "unsigned* const p=a+16;", "p[i]=i;", bound="0")
    assert loop["status"] == "eligible"
    assert loop["proof"]["accesses"][0]["extent"] == 0


def test_const_pointee_read_retains_origin(tmp_path):
    loop = analyze(tmp_path, "const unsigned* const p=a+2;", "b[i]=p[i];")
    assert loop["status"] == "eligible"


def test_pointer_read_requires_initialized_origin(tmp_path):
    loop = analyze(tmp_path, "unsigned* const p=b;", "a[i]=p[i];")
    assert loop["status"] == "unproven"
    assert "BufferInitializationProofMissing" in loop["reasons"]
