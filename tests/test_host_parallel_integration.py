from pathlib import Path
from core.application.pipeline import CompilerPipeline
from core.application.host_parallel import integrate_host_loops
from core.frontend.host_loops import analyze_host_loops

ROOT = Path(__file__).resolve().parents[1]


def test_safe_requires_native(tmp_path):
    result = CompilerPipeline(ROOT).build(tmp_path, auto_parallel="safe")
    assert result.exit_code == 2
    assert result.payload["diagnostics"][0]["code"] == "HostParallelNativeRequired"


def test_multiple_edits_preserve_source_and_unproven_loops(tmp_path):
    source = tmp_path / "map.cpp"
    text = "void f(){unsigned a[8]={}; for(unsigned i=0;i<4;++i)a[i]=i; for(unsigned j=0;j<4;++j)a[j]=a[j]+1u; while(a[0])a[0]--; }"
    source.write_text(text)
    report = analyze_host_loops({"sources": [str(source)]})
    generated = {"metal_source": "", "routing_table": {"translated_kernels": {}}}
    integrate_host_loops(report, generated, "safe")
    assert report["transformed_loop_count"] == 2
    replacement = generated["host_source_overrides"][str(source)]
    assert "while(a[0])a[0]--;" in replacement
    assert source.read_text() == text
    assert len(generated["routing_table"]["translated_kernels"]) == 2
    assert generated["host_parallel_cache"]["mode"] == "safe"


def test_report_build_does_not_enable_offload(tmp_path, monkeypatch):
    monkeypatch.setenv("MCU_CACHE_DIR", str(tmp_path / "cache"))
    source = tmp_path / "map.cpp"
    source.write_text("int main(){unsigned a[8]={}; for(unsigned i=0;i<4;++i)a[i]=i; return 0;}")
    result = CompilerPipeline(ROOT, artifact_path=str(tmp_path / "artifacts.zip")).build(
        source, auto_parallel="report"
    )
    assert result.exit_code == 0
    assert result.payload["auto_parallel"]["transformed_loop_count"] == 0
    assert result.payload["build"]["translated_kernel_count"] == 0


def test_native_output_marker_excludes_generated_sources(tmp_path):
    from core.frontend.regex_parser import discover_sources

    original = tmp_path / "main.cpp"
    original.write_text("int main(){return 0;}")
    output = tmp_path / "custom-output"
    output.mkdir()
    (output / ".mcu-native-build").write_text("MCU native build directory\n")
    (output / "main.cpp").write_text("int main(){return 1;}")
    assert discover_sources(original) == [original]


def test_cli_rejects_safe_without_native(tmp_path):
    import subprocess
    import json

    result = subprocess.run(
        [str(ROOT / "mcu"), "build", str(tmp_path), "--auto-parallel=safe", "--json"],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["diagnostics"][0]["code"] == "HostParallelNativeRequired"


def test_presenter_distinguishes_generated_gpu_from_report_cpu(capsys):
    from core.diagnostics.presenter import emit

    emit(
        {
            "command": "build",
            "ok": True,
            "diagnostics": [],
            "auto_parallel": {
                "loops": [
                    {
                        "node": {"file": "map.cpp"},
                        "line": 1,
                        "column": 1,
                        "function": "map",
                        "status": "conditional",
                        "execution": "gpu_with_cpu_guard_fallback",
                        "reasons": [],
                    }
                ]
            },
        },
        False,
    )
    assert "GPU with CPU fallback" in capsys.readouterr().out
