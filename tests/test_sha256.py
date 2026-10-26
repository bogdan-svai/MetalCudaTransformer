import json
import platform
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MCU = ROOT / "mcu"
SOURCE = ROOT / "examples" / "sha256" / "app" / "main.cpp"
EXPECTED = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


@pytest.mark.skipif(platform.system() != "Darwin", reason="native test requires Metal")
def test_multifile_device_helpers_compute_sha256_on_metal(tmp_path):
    output = tmp_path / "native"
    build = subprocess.run(
        [
            sys.executable,
            str(MCU),
            "build",
            str(SOURCE),
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

    assert build.returncode == 0, build.stderr or build.stdout
    payload = json.loads(build.stdout)
    executable = Path(payload["build"]["native"]["executable"])
    manifest = json.loads((output / "symbol_manifest.json").read_text())
    symbol_names = {symbol["name"] for symbol in manifest["symbols"].values()}

    assert executable.is_file()
    assert "sha256_compress" in symbol_names
    assert "rotate_right" in symbol_names
    assert manifest["frontend"] == "libclang_semantic"
    assert "device_functions" in manifest["capabilities"]

    run = subprocess.run([str(executable)], text=True, capture_output=True, check=False)

    assert run.returncode == 0, run.stderr or run.stdout
    assert f"sha256_abc={EXPECTED}" in run.stdout
