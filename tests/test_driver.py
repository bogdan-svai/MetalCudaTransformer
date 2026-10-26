from pathlib import Path

import pytest

from core.cache import CacheStore, fingerprint
from core.driver import NvccInvocationError, parse_nvcc_arguments
from core.objects import McuObjectError, create_mcu_object, load_mcu_object, write_mcu_object
from core.frontend.regex_parser import discover_sources


def test_nvcc_driver_parses_device_compile_context(tmp_path):
    invocation = parse_nvcc_arguments(
        ["-dc", "src/kernel.cu", "-Iinclude", "-DCOUNT=32", "-O3", "-o", "obj/kernel.o"],
        working_directory=tmp_path,
    )

    assert invocation.mode == "device_compile"
    assert invocation.sources == [str(tmp_path / "src/kernel.cu")]
    assert invocation.include_paths == [str(tmp_path / "include")]
    assert invocation.definitions == ["COUNT=32"]
    assert invocation.optimization == "-O3"
    assert invocation.optimization_explicit is True
    assert invocation.output == str(tmp_path / "obj/kernel.o")


def test_nvcc_driver_rejects_semantic_unknown_flag(tmp_path):
    with pytest.raises(NvccInvocationError) as captured:
        parse_nvcc_arguments(["kernel.cu", "--use_fast_math"], working_directory=tmp_path)

    assert captured.value.code == "UnsupportedNvccFlag"
    assert captured.value.argument == "--use_fast_math"


def test_mcu_object_round_trip_and_dependency_invalidation(tmp_path):
    include = tmp_path / "include"
    include.mkdir()
    header = include / "value.h"
    header.write_text("#define VALUE 7\n", encoding="ascii")
    source = tmp_path / "kernel.cu"
    source.write_text('#include "value.h"\n__global__ void k(int* out) { out[0] = VALUE; }\n', encoding="ascii")
    invocation = parse_nvcc_arguments(
        ["-dc", str(source), "-I", str(include), "-o", str(tmp_path / "kernel.o")],
        working_directory=tmp_path,
    )

    output = write_mcu_object(invocation.output, create_mcu_object(source, invocation))
    loaded = load_mcu_object(output)
    assert loaded["source"] == str(source)
    assert {Path(item["path"]).name for item in loaded["dependencies"]} == {"kernel.cu", "value.h"}

    header.write_text("#define VALUE 8\n", encoding="ascii")
    with pytest.raises(McuObjectError) as captured:
        load_mcu_object(output)
    assert captured.value.code == "IncompatibleMcuObject"


def test_cache_store_restores_immutable_entry(tmp_path):
    store = CacheStore(tmp_path / "cache")
    source = tmp_path / "source.bin"
    source.write_bytes(b"metal-library")
    key = fingerprint({"source": "abc", "flags": ["-O2"]})

    store.publish("metal", key, "generated.metallib", source)
    destination = tmp_path / "restored.metallib"
    assert store.restore("metal", key, "generated.metallib", destination)
    assert destination.read_bytes() == b"metal-library"

    store.path("metal", key, "generated.metallib").write_bytes(b"corrupt")
    assert not store.restore("metal", key, "generated.metallib", tmp_path / "rejected.metallib")


def test_source_discovery_ignores_generated_sidecar_directories(tmp_path):
    source = tmp_path / "main.cu"
    source.write_text("int main() { return 0; }\n", encoding="ascii")
    generated = tmp_path / "app.mcu" / "host"
    generated.mkdir(parents=True)
    (generated / "main.cpp").write_text("int main() { return 0; }\n", encoding="ascii")

    assert discover_sources(source) == [source]
