import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
MCU = ROOT / "mcu"


def _run(*arguments):
    result = subprocess.run(
        [sys.executable, str(MCU), *arguments],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result


def _json(*arguments):
    result = _run(*arguments, "--json")
    return result, json.loads(result.stdout)


def test_capabilities_json_contract_is_stable():
    result, payload = _json("capabilities")

    assert result.returncode == 0, result.stderr or result.stdout
    assert result.stdout.count("\n") == 1
    assert payload["schema_version"] == 2
    assert payload["contract"] == "mcu.tooling.v2"
    assert payload["mcu_version"] == "0.14.0"
    assert payload["command"] == "capabilities"
    assert payload["exit_status"] == {"class": "success", "code": 0}
    assert payload["target"]["backend"] == "metal"
    assert payload["runtime_abi"] >= 5


def test_capabilities_report_runtime_intrinsics_flags_and_libraries():
    _, payload = _json("capabilities")

    runtime = {item["name"]: item for item in payload["runtime_apis"]}
    assert runtime["cudaEventRecord"]["status"] == "supported"
    assert runtime["cudaGetDeviceProperties"]["status"] == "supported"
    assert runtime["cudaOccupancyMaxPotentialBlockSize"]["signatures"]

    syntax = {item["name"]: item for item in payload["syntax"]}
    assert syntax["cuda_graphs"]["status"] == "partial"
    assert syntax["warp_intrinsics"]["status"] == "partial"

    intrinsics = {item["name"]: item for item in payload["intrinsics"]}
    assert intrinsics["atomicAdd"]["capability"] == "typed_integer_atomics"
    assert intrinsics["__syncthreads"]["status"] == "supported"

    flags = {item["name"]: item for item in payload["nvcc_flags"]}
    assert flags["-dc"]["status"] == "supported"
    assert flags["-Xcompiler"]["status"] == "partial"

    libraries = {(item["library"], item["symbol"]): item for item in payload["library_adapters"]}
    assert libraries[("cublas_v2.h", "cublasSgemm")]["status"] == "supported"
    assert libraries[("curand.h", "*")]["status"] == "planned"


def test_analyze_json_returns_metadata_diagnostics_and_capability_summary(tmp_path):
    target = tmp_path / "case.cu"
    target.write_text(
        "__global__ void work(int* values) { values[threadIdx.x] = 7; }\n",
        encoding="ascii",
    )

    result, payload = _json("analyze", str(target))

    assert result.returncode == 0, result.stderr or result.stdout
    assert payload["schema_version"] == 2
    assert payload["contract"] == "mcu.tooling.v2"
    assert payload["command"] == "analyze"
    assert payload["metadata"]["kernels"][0]["name"] == "work"
    assert payload["diagnostics"] == []
    assert payload["capabilities"]["contract"] == "mcu.tooling.v2"
    assert payload["capabilities"]["supported_runtime_api_count"] >= 20
    assert payload["timings"]["phases_seconds"]["frontend"] >= 0


def test_analyze_invalid_path_uses_infrastructure_exit_class(tmp_path):
    missing = tmp_path / "missing.cu"
    result, payload = _json("analyze", str(missing))

    assert result.returncode == 2
    assert payload["ok"] is False
    assert payload["diagnostics"][0]["code"] == "InvalidTargetPath"
    assert payload["exit_status"] == {"class": "infrastructure_failure", "code": 2}


def test_existing_check_json_receives_tooling_envelope(tmp_path):
    target = tmp_path / "case.cu"
    target.write_text("__global__ void work(int* values) { values[0] = 1; }\n", encoding="ascii")

    result, payload = _json("check", str(target))

    assert result.returncode == 0
    assert payload["schema_version"] == 2
    assert payload["contract"] == "mcu.tooling.v2"
    assert payload["mcu_version"] == "0.14.0"
    assert payload["exit_status"]["class"] == "success"
