import json
import subprocess
from pathlib import Path

from core.frontend.host_loops import analyze_host_loops
from core.ir.host_loops import HostLoopIR
from core.tooling.analyze import analyze_path

CORPUS = Path(__file__).parent / "fixtures/host_loops/loops.cpp"


def test_host_only_report_and_roundtrip():
    report = analyze_host_loops({"sources": [str(CORPUS)]})
    assert report["complete"]
    assert len(report["loops"]) == 8
    for item in report["loops"]:
        fields = {
            k: v
            for k, v in item.items()
            if k not in {"status", "execution", "estimated_iterations", "transfer_bytes", "proof"}
        }
        record = HostLoopIR.from_dict(json.loads(json.dumps(fields)))
        assert record.to_dict() == fields
        assert item["status"] == "unproven"
        assert item["execution"] == "cpu"
    loop = next(item for item in report["loops"] if item["function"] == "map")
    assert loop["init"]["children"][0]["type"] == "unsigned int"
    assert "<" in loop["condition"]["tokens"]
    assert "++" in loop["update"]["tokens"]
    assert loop["node"]["file"] == str(CORPUS.resolve())
    macro = next(item for item in report["loops"] if item["function"] == "macro_loop")
    assert "MacroRangeRequiresMapping" in macro["reasons"]
    assert macro["init"] is None
    effects = next(item for item in report["loops"] if item["function"] == "effects")
    assert "CallEffectProofMissing" in effects["reasons"]


def test_analysis_is_opt_in_and_does_not_modify_input():
    before = CORPUS.read_bytes()
    off = analyze_path(CORPUS)
    on = analyze_path(CORPUS, auto_parallel="report")
    assert "auto_parallel" not in off.payload
    assert on.payload["auto_parallel"]["transformed_loop_count"] == 0
    assert off.payload["metadata"] == on.payload["metadata"]
    assert CORPUS.read_bytes() == before


def test_cli_report_json():
    root = Path(__file__).resolve().parents[1]
    run = subprocess.run(
        [str(root / "mcu"), "analyze", str(CORPUS), "--auto-parallel=report", "--json"],
        text=True,
        capture_output=True,
    )
    assert run.returncode == 0, run.stderr + run.stdout
    assert len(json.loads(run.stdout)["auto_parallel"]["loops"]) == 8


def test_errors_cannot_produce_complete_host_report(tmp_path):
    source = tmp_path / "invalid.cpp"
    source.write_text("void f() { for(unsigned i=0;i<3;++i) unknown(i); }")
    report = analyze_host_loops({"sources": [str(source)]})
    assert not report["complete"]
    assert report["diagnostics"]
    assert all("TranslationUnitHasErrors" in item["reasons"] for item in report["loops"])


def test_cuda_report_excludes_device_loops(tmp_path):
    source = tmp_path / "mixed.cu"
    source.write_text(
        "__device__ void device(unsigned* a) { for(unsigned i=0;i<4;++i) a[i]=i; }\n"
        "void host(unsigned* a) { for(unsigned i=0;i<4;++i) a[i]=i; }\n"
    )
    report = analyze_host_loops({"sources": [str(source)]})
    assert report["complete"], report["diagnostics"]
    assert [item["function"] for item in report["loops"]] == ["host"]


def test_missing_header_roles_remain_unavailable(tmp_path):
    source = tmp_path / "incomplete.cpp"
    source.write_text("void f(unsigned* a) { unsigned i=0; for(;i<4;) a[i++]=1u; }")
    report = analyze_host_loops({"sources": [str(source)]})
    (loop,) = report["loops"]
    assert loop["init"] is None
    assert "LoopHeaderRolesUnavailable" in loop["reasons"]


def test_header_loop_keeps_original_location_and_unit_identity(tmp_path):
    header = tmp_path / "map.h"
    header.write_text("inline void map(unsigned* a) { for(unsigned i=0;i<3;++i) a[i]=i; }\n")
    units = []
    for name in ("first.cpp", "second.cpp"):
        unit = tmp_path / name
        unit.write_text('#include "map.h"\n')
        units.append(unit)
    report = analyze_host_loops({"sources": [str(header), *map(str, units)]})
    assert report["complete"]
    first, second = report["loops"]
    assert first["function_usr"] == second["function_usr"]
    assert first["node"]["file"] == str(header)
    assert {first["translation_unit"], second["translation_unit"]} == set(map(str, units))


def test_original_c_loop_is_reported(tmp_path):
    source = tmp_path / "map.c"
    source.write_text("void map(unsigned* a) { for(unsigned i=0;i<3;++i) a[i]=i; }\n")
    report = analyze_host_loops({"sources": [str(source)]})
    assert report["complete"]
    assert len(report["loops"]) == 1
