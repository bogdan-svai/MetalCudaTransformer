from pathlib import Path
import pytest
from core.frontend.host_loops import analyze_host_loops
from core.backends.host_parallel import generate_loop


def generated(tmp_path, code):
    source = tmp_path / "map.cpp"
    source.write_text(code)
    (item,) = analyze_host_loops({"sources": [str(source)]})["loops"]
    return generate_loop(item, source.read_bytes())


def test_codegen_groups_aliases_and_copies_only_written_regions(tmp_path):
    plan = generated(
        tmp_path,
        "void f(){unsigned a[16]={}; unsigned* const p=a; unsigned* const q=a+8; for(unsigned i=0;i<4;++i) q[i]=p[i]+1u;}",
    )
    assert len(plan["captures"]) == 1
    assert plan["captures"][0]["writes"] == [8]
    assert "b0[tid + 8u]" in plan["metal_source"]
    assert "cudaDeviceSynchronize" in plan["host_replacement"]
    assert "a + 8u" in plan["host_replacement"]
    assert "cudaMemcpyHostToDevice" in plan["host_replacement"]


def test_write_only_buffer_does_not_upload_uninitialized_values(tmp_path):
    plan = generated(tmp_path, "void f(){unsigned a[16]; for(unsigned i=0;i<4;++i) a[i]=i;}")
    assert "cudaMemcpyHostToDevice" not in plan["host_replacement"]
    assert "cudaMemcpyDeviceToHost" in plan["host_replacement"]


def test_dynamic_guard_preserves_original_cpu_loop(tmp_path):
    plan = generated(
        tmp_path, "void f(unsigned n){unsigned a[16]={}; for(unsigned i=0;i<n;++i) a[i]=a[i]+1u;}"
    )
    assert "count <= 16u" in plan["host_replacement"]
    assert "for(unsigned i=0;i<n;++i) a[i]=a[i]+1u;" in plan["host_replacement"]


def test_unproven_loop_has_no_codegen(tmp_path):
    with pytest.raises(ValueError, match="no complete"):
        generated(tmp_path, "void f(unsigned* a){for(unsigned i=0;i<4;++i) a[i]=i;}")


def test_stale_source_cannot_authorize_generation(tmp_path):
    source = tmp_path / "map.cpp"
    source.write_text("void f(){unsigned a[8]; for(unsigned i=0;i<4;++i) a[i]=i;}")
    (item,) = analyze_host_loops({"sources": [str(source)]})["loops"]
    with pytest.raises(ValueError, match="changed"):
        generate_loop(item, source.read_bytes() + b"// changed")
