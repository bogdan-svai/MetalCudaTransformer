import pytest
from core.frontend.host_loops import analyze_host_loops


def analyze(tmp_path, declarations, body, header="unsigned i=0;i<8;++i"):
    source = tmp_path / "loops.cpp"
    source.write_text(f"void f() {{ {declarations} for({header}) {{ {body} }} }}")
    report = analyze_host_loops({"sources": [str(source)]})
    assert report["complete"], report["diagnostics"]
    return report["loops"][0]


def test_local_unsigned_map_has_complete_static_memory_proof(tmp_path):
    loop = analyze(tmp_path, "unsigned a[8]={}; unsigned b[8];", "b[i]=a[i]*3u+1u;")
    assert loop["status"] == "eligible", loop["reasons"]
    assert loop["proof"]["eligible"]
    assert loop["estimated_iterations"] == 8
    assert loop["execution"] == "cpu"
    assert loop["reasons"] == ["ProfitabilityNotMeasured"]
    assert len(loop["proof"]["accesses"]) == 2


def test_same_array_same_index_is_independent(tmp_path):
    loop = analyze(tmp_path, "unsigned a[8]={};", "a[i]=a[i]+1u;")
    assert loop["proof"]["eligible"]


@pytest.mark.parametrize(
    "declarations,body,header,reason",
    [
        (
            "unsigned a[8]; unsigned b[8];",
            "b[i]=a[i];",
            "unsigned i=0;i<8;++i",
            "BufferInitializationProofMissing",
        ),
        ("unsigned a[4]={};", "a[i]=i;", "unsigned i=0;i<8;++i", "BufferBoundsExceeded"),
        (
            "unsigned a[8]={};",
            "a[i]=a[i-1];",
            "unsigned i=0;i<8;++i",
            "CrossIterationAccessUnsupported",
        ),
        (
            "unsigned a[8]={}; unsigned* p=a;",
            "p[i]=i;",
            "unsigned i=0;i<8;++i",
            "BufferExtentOrAliasProofMissing",
        ),
        ("unsigned a[8]={};", "a[0]=i;", "unsigned i=0;i<8;++i", "CrossIterationAccessUnsupported"),
        (
            "static unsigned a[8]={};",
            "a[i]=i;",
            "unsigned i=0;i<8;++i",
            "BufferExtentOrAliasProofMissing",
        ),
        (
            "volatile unsigned a[8]={};",
            "a[i]=i;",
            "unsigned i=0;i<8;++i",
            "BufferExtentOrAliasProofMissing",
        ),
        (
            "unsigned a[8]={};",
            "a[i]=i;",
            "unsigned i=0;i<=4294967295u;++i",
            "LoopBoundsUnsupported",
        ),
        ("unsigned a[8]={};", "a[i]=i;", "int i=0;i<8;++i", "IteratorTypeUnsupported"),
        ("unsigned a[8]={};", "a[i]=i;", "unsigned i=0;i<8;i+=2", "IteratorUpdateUnsupported"),
        (
            "unsigned a[8]={};",
            "a[i]=i; ++i;",
            "unsigned i=0;i<8;++i",
            "StatementOrEffectUnsupported",
        ),
        (
            "unsigned a[8]={}; unsigned n=8;",
            "a[i]=i;",
            "unsigned i=0;i<n;++i",
            "LoopBoundsUnsupported",
        ),
        (
            "unsigned a[8]={};",
            "a[i]=i/0u;",
            "unsigned i=0;i<8;++i",
            "ExpressionOrEffectUnsupported",
        ),
        (
            "unsigned a[8]={};",
            "a[i]=2147483647+1;",
            "unsigned i=0;i<8;++i",
            "ExpressionOrEffectUnsupported",
        ),
        (
            "unsigned a[8]={};",
            "if(i) a[i]=i;",
            "unsigned i=0;i<8;++i",
            "StatementOrEffectUnsupported",
        ),
    ],
)
def test_unsafe_or_unproven_loops_stay_on_cpu(tmp_path, declarations, body, header, reason):
    loop = analyze(tmp_path, declarations, body, header)
    assert not loop["proof"]["eligible"]
    assert loop["status"] == "unproven"
    assert loop["execution"] == "cpu"
    assert reason in loop["reasons"]
