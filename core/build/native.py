import shutil
import subprocess
import os
from pathlib import Path

from core.backends.host import has_host_code, transform_host_source
from core.diagnostics.compiler import parse_compiler_diagnostics
from core.frontend.regex_parser import read_text


HOST_SOURCE_EXTENSIONS = {".c", ".cc", ".cpp", ".cxx", ".cu"}
HEADER_EXTENSIONS = {".h", ".hpp"}


def build_native(metadata, build_output, output_directory, project_root):
    output = Path(output_directory).resolve()
    host_directory = output / "host"
    marker = output / ".mcu-native-build"
    output.mkdir(parents=True, exist_ok=True)
    if host_directory.exists():
        legacy_build = (output / "generated.metal").is_file() and (output / "routing_table.txt").is_file()
        if not marker.is_file() and not legacy_build:
            return {
                "ok": False,
                "diagnostic": f"Refusing to replace unowned host directory: {host_directory}",
                "command": [],
            }
        shutil.rmtree(host_directory)
    host_directory.mkdir(parents=True)
    marker.write_text("MCU native build directory\n", encoding="ascii")

    (output / "generated.metal").write_text(build_output["metal_source"], encoding="utf-8")
    (output / "routing_table.txt").write_text(build_output["routing_table_text"], encoding="utf-8")
    if build_output.get("symbol_manifest_text"):
        (output / "symbol_manifest.json").write_text(
            build_output["symbol_manifest_text"],
            encoding="utf-8",
        )

    metal_command = [
        "xcrun",
        "-sdk",
        "macosx",
        "metal",
        f"-fmodules-cache-path={output / 'module-cache'}",
        str(output / "generated.metal"),
        "-o",
        str(output / "generated.air"),
    ]
    try:
        metal_result = subprocess.run(metal_command, text=True, capture_output=True, check=False)
    except OSError as exc:
        return {
            "ok": False,
            "phase": "metal",
            "diagnostic": f"Unable to start Metal compiler: {exc}",
            "diagnostics": [],
            "command": metal_command,
        }
    if metal_result.returncode != 0:
        return {
            "ok": False,
            "phase": "metal",
            "diagnostic": metal_result.stderr.strip(),
            "diagnostics": parse_compiler_diagnostics(
                metal_result.stderr,
                code="MetalCompileError",
            ),
            "command": metal_command,
            "stderr": metal_result.stderr,
        }

    source_paths = [Path(name).resolve() for name in metadata.get("sources", [])]
    source_root = Path(os.path.commonpath([str(path.parent) for path in source_paths]))
    host_sources = []
    for source in source_paths:
        if source.suffix not in HOST_SOURCE_EXTENSIONS | HEADER_EXTENSIONS:
            continue
        transformed = transform_host_source(read_text(source))
        relative = source.relative_to(source_root)
        if source.suffix in HEADER_EXTENSIONS:
            destination = host_directory / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(transformed, encoding="utf-8")
            continue
        if not has_host_code(transformed):
            continue
        destination = host_directory / relative
        if destination.suffix != ".cpp":
            destination = destination.with_name(destination.name + ".cpp")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(transformed, encoding="utf-8")
        host_sources.append(destination)

    if not host_sources:
        return {
            "ok": False,
            "diagnostic": "No host compilation units remained after CUDA kernel extraction.",
            "command": [],
        }

    executable_name = Path(metadata.get("entry", "mcu_app")).stem or "mcu_app"
    executable = output / executable_name
    root = Path(project_root).resolve()
    runtime_include = root / "runtime" / "include"
    runtime_sources = sorted((root / "runtime" / "src").glob("*.mm"))
    command = [
        "xcrun",
        "clang++",
        "-std=c++17",
        "-fobjc-arc",
        "-O2",
        "-I",
        str(runtime_include),
        "-I",
        str(host_directory),
        *[str(path) for path in host_sources],
        *[str(path) for path in runtime_sources],
        "-framework",
        "Foundation",
        "-framework",
        "Metal",
        "-o",
        str(executable),
    ]
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=False)
    except OSError as exc:
        return {
            "ok": False,
            "diagnostic": f"Unable to start native compiler: {exc}",
            "command": command,
        }
    return {
        "ok": result.returncode == 0,
        "phase": "host",
        "executable": str(executable),
        "build_directory": str(output),
        "host_sources": [str(path) for path in host_sources],
        "source_root": str(source_root),
        "command": command,
        "metal_command": metal_command,
        "diagnostics": parse_compiler_diagnostics(
            result.stderr,
            code="HostCompileError",
        ) if result.returncode else [],
        "stdout": result.stdout,
        "stderr": result.stderr,
        "diagnostic": result.stderr.strip() if result.returncode else "",
    }
