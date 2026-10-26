import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from core.build.native import _stable_routing_for_cache
from core.application.toolchain import _merge_object_semantic_projects
from core.objects import load_mcu_object

ROOT = Path(__file__).resolve().parents[1]
MCU = ROOT / "mcu"
NUMERIC = ROOT / "examples" / "numeric" / "main.cu"


def test_native_cache_key_ignores_volatile_optimizer_timings():
    first = {
        "routing_table": {
            "schema_version": 2,
            "metal_optimization": {
                "passes": [{"name": "unroll", "details": {"timing_ms": 0.1}}],
                "analysis": {"optimization_total_ms": 0.2, "register_pressure": 4},
            },
        }
    }
    second = {
        "routing_table": {
            "schema_version": 2,
            "metal_optimization": {
                "passes": [{"name": "unroll", "details": {"timing_ms": 9.9}}],
                "analysis": {"optimization_total_ms": 12.0, "register_pressure": 4},
            },
        }
    }

    assert _stable_routing_for_cache(first) == _stable_routing_for_cache(second)


def _run(arguments, cwd, cache):
    environment = os.environ.copy()
    environment["MCU_CACHE_DIR"] = str(cache)
    result = subprocess.run(
        [sys.executable, str(MCU), *arguments, "--json"],
        cwd=cwd,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    return result, json.loads(result.stdout)


def test_compile_only_emits_mcu_object_and_depfile(tmp_path):
    source = tmp_path / "kernel.cu"
    header = tmp_path / "value.h"
    header.write_text("#define VALUE 4\n", encoding="ascii")
    source.write_text(
        '#include "value.h"\n__global__ void kernel(int* out) { out[0] = VALUE; }\n',
        encoding="ascii",
    )
    output = tmp_path / "kernel.o"
    depfile = tmp_path / "kernel.d"

    result, payload = _run(
        ["compile", "-dc", str(source), "-MD", "-MF", str(depfile), "-o", str(output)],
        ROOT,
        tmp_path / "cache",
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert output.read_bytes().startswith(b"MCUOBJ\n")
    assert str(source) in depfile.read_text(encoding="utf-8")
    assert str(header) in depfile.read_text(encoding="utf-8")
    assert payload["object"]["schema_version"] == 1
    object_payload = json.loads(output.read_text(encoding="utf-8").split("\n", 1)[1])
    assert object_payload["semantic_ir"]["frontend"] == "libclang_semantic"
    assert object_payload["target"]["architecture"]


def test_object_semantic_relink_resolves_cross_file_usr(tmp_path):
    header = tmp_path / "helpers.h"
    header.write_text("__device__ int helper(int value);\n", encoding="ascii")
    kernel = tmp_path / "kernel.cu"
    kernel.write_text(
        '#include "helpers.h"\n' "__global__ void kernel(int* output) { output[0] = helper(7); }\n",
        encoding="ascii",
    )
    helper = tmp_path / "helper.cu"
    helper.write_text(
        '#include "helpers.h"\n' "__device__ int helper(int value) { return value + 1; }\n",
        encoding="ascii",
    )
    objects = []
    for source in (kernel, helper):
        output = tmp_path / f"{source.stem}.o"
        result, _payload = _run(
            ["compile", "-dc", str(source), "-I", str(tmp_path), "-o", str(output)],
            ROOT,
            tmp_path / "cache",
        )
        assert result.returncode == 0, result.stderr or result.stdout
        objects.append(load_mcu_object(output))

    project = _merge_object_semantic_projects(objects)
    assert project is not None
    kernel_function = next(item for item in project.functions.values() if item.name == "kernel")
    helper_function = next(item for item in project.functions.values() if item.name == "helper")
    call = next(item for item in kernel_function.calls if item.callee_name == "helper")
    assert call.callee_id == helper_function.symbol_id


def test_multifile_link_rejects_stale_common_header(tmp_path):
    header = tmp_path / "shared.h"
    header.write_text("#define VALUE 3\n", encoding="ascii")
    sources = []
    for name, body in (
        ("kernel", "__global__ void kernel(int* output) { output[0] = VALUE; }\n"),
        ("helper", "__device__ int helper() { return VALUE; }\n"),
    ):
        source = tmp_path / f"{name}.cu"
        source.write_text(f'#include "shared.h"\n{body}', encoding="ascii")
        sources.append(source)
    objects = []
    for source in sources:
        output = tmp_path / f"{source.stem}.o"
        result, _payload = _run(
            ["compile", "-dc", str(source), "-I", str(tmp_path), "-o", str(output)],
            ROOT,
            tmp_path / "cache",
        )
        assert result.returncode == 0, result.stderr or result.stdout
        objects.append(output)

    header.write_text("#define VALUE 4\n", encoding="ascii")
    result, payload = _run(
        ["compile", *(str(item) for item in objects), "-o", str(tmp_path / "linked")],
        ROOT,
        tmp_path / "cache",
    )
    assert result.returncode != 0
    assert any(
        item["code"] == "IncompatibleMcuObject" and "dependency changed" in item["message"]
        for item in payload["diagnostics"]
    )


@pytest.mark.skipif(platform.system() != "Darwin", reason="run requires the native toolchain")
def test_mcu_run_forwards_program_arguments_and_exit_status(tmp_path):
    source = tmp_path / "arguments.cu"
    source.write_text(
        """#include <cstdio>
#ifndef TOOLCHAIN_VALUE
#error TOOLCHAIN_VALUE must be supplied by the compiler driver
#endif
__global__ void unused_kernel(int* output) { output[0] = TOOLCHAIN_VALUE; }
int main(int argc, char** argv) {
    if (argc != 2) return 9;
    std::printf("argument=%s\\n", argv[1]);
    return argv[1][0] == 'h' ? 0 : 7;
}
""",
        encoding="ascii",
    )
    executable = tmp_path / "arguments"

    result, payload = _run(
        ["run", str(source), "-DTOOLCHAIN_VALUE=7", "-o", str(executable), "--", "hello-toolchain"],
        ROOT,
        tmp_path / "cache",
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert payload["program"]["arguments"] == ["hello-toolchain"]
    assert payload["program"]["stdout"] == "argument=hello-toolchain\n"

    failed, failed_payload = _run(
        ["run", str(source), "-DTOOLCHAIN_VALUE=7", "-o", str(executable), "--", "bad"],
        ROOT,
        tmp_path / "cache",
    )
    assert failed.returncode == 7, failed.stdout
    assert failed_payload["program"]["exit_code"] == 7


@pytest.mark.skipif(
    platform.system() != "Darwin", reason="native toolchain requires Xcode and Metal"
)
def test_compile_links_metallib_runs_without_msl_and_hits_noop_cache(tmp_path):
    cache = tmp_path / "cache"
    object_path = tmp_path / "numeric.o"
    executable = tmp_path / "numeric"

    compiled, _ = _run(
        ["compile", "-dc", str(NUMERIC), "-o", str(object_path)],
        ROOT,
        cache,
    )
    assert compiled.returncode == 0, compiled.stderr or compiled.stdout

    linked, payload = _run(
        ["compile", str(object_path), "-o", str(executable)],
        ROOT,
        cache,
    )
    assert linked.returncode == 0, linked.stderr or linked.stdout
    sidecar = Path(f"{executable}.mcu")
    assert (sidecar / "generated.metallib").is_file()
    repeated, warm_payload = _run(
        ["compile", str(object_path), "-o", str(executable)],
        ROOT,
        cache,
    )
    assert repeated.returncode == 0
    native = warm_payload["build"]["native"]
    assert native["cache_hit"] is True
    assert warm_payload["timings"]["phases_seconds"]["total"] < 0.20

    (sidecar / "generated.metal").unlink()
    relocated = tmp_path / "relocated"
    relocated.mkdir()
    relocated_executable = Path(shutil.move(executable, relocated / executable.name))
    shutil.move(sidecar, Path(f"{relocated_executable}.mcu"))
    executed = subprocess.run(
        [str(relocated_executable)], text=True, capture_output=True, check=False
    )
    assert executed.returncode == 0, executed.stderr
    assert "numeric_count=100000 max_error=0" in executed.stdout


@pytest.mark.skipif(platform.system() != "Darwin", reason="Make integration requires Xcode")
def test_mcu_make_intercepts_nested_nvcc_compile_and_link(tmp_path):
    project = tmp_path / "project with spaces"
    application = project / "app"
    application.mkdir(parents=True)
    (application / "main.cu").write_text(
        """__global__ void kernel(int* output) { output[threadIdx.x] = 1; }
int main() { return 0; }
""",
        encoding="ascii",
    )
    (application / "Makefile").write_text(
        """NVCC ?= nvcc

all: app

main.o: main.cu
	$(NVCC) -dc main.cu -MMD -MF main.d -o main.o

app: main.o
	$(NVCC) main.o -o app
""",
        encoding="ascii",
    )
    (project / "Makefile").write_text(
        "all:\n\t$(MAKE) -C app\n",
        encoding="ascii",
    )

    result, payload = _run(
        ["make", "-j2", "-C", str(project)],
        ROOT,
        tmp_path / "cache",
    )

    assert result.returncode == 0, payload.get("make", {}).get("stderr") or result.stdout
    assert (application / "main.o").read_bytes().startswith(b"MCUOBJ\n")
    assert (application / "app").is_file()
    assert (application / "app.mcu" / "generated.metallib").is_file()
    assert payload["make"]["exit_code"] == 0
