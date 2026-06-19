import json
import platform
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MCU = ROOT / "mcu"


def test_routing_and_msl_retain_original_kernel_location(tmp_path):
    target = tmp_path / "mapped.cu"
    target.write_text(
        """__global__ void mapped(float* values) {
    values[threadIdx.x] = 1.0f;
}
void host() { cudaLaunch(mapped, 1, 32, 0); }
""",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(MCU), "build", str(target), "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    with zipfile.ZipFile(ROOT / "mcu_artifacts.zip") as archive:
        metal = archive.read("generated.metal").decode("utf-8")
        routing = json.loads(archive.read("routing_table.txt"))
    assert f'#line 1 "{target.as_posix()}"' in metal
    assert routing["translated_kernels"]["mapped"]["source"] == {
        "file": target.as_posix(),
        "line": 1,
    }


@pytest.mark.skipif(platform.system() != "Darwin", reason="Metal diagnostics require Xcode")
def test_invalid_msl_diagnostic_points_to_cuda_source(tmp_path):
    target = tmp_path / "broken.cu"
    target.write_text(
        """__global__ void broken(float* values) {
    int index = threadIdx.x;
    values[index] = missing_cuda_value;
}
int main() {
    broken<<<1, 2200>>>(nullptr);
    return 0;
}
""",
        encoding="utf-8",
    )
    output = tmp_path / "native"

    result = subprocess.run(
        [
            sys.executable,
            str(MCU),
            "build",
            str(target),
            "--native",
            "--output",
            str(output),
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    mapped = [item for item in payload["diagnostics"] if item["code"] == "MetalCompileError"]
    assert mapped
    assert mapped[0]["file"] == target.as_posix()
    assert mapped[0]["line"] == 3
    assert "missing_cuda_value" in mapped[0]["message"]
