import shutil
import subprocess
from pathlib import Path

from core.fallback import read_text
from core.host_transpiler import has_host_code, transform_host_source


HOST_SOURCE_EXTENSIONS = {".c", ".cc", ".cpp", ".cxx", ".cu"}


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

    host_sources = []
    for index, source_name in enumerate(metadata.get("sources", [])):
        source = Path(source_name)
        if source.suffix not in HOST_SOURCE_EXTENSIONS:
            continue
        transformed = transform_host_source(read_text(source))
        if not has_host_code(transformed):
            continue
        destination = host_directory / f"{index:03d}_{source.stem}.cpp"
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
    runtime_source = root / "runtime" / "src" / "runtime_core.mm"
    command = [
        "xcrun",
        "clang++",
        "-std=c++17",
        "-fobjc-arc",
        "-O2",
        "-I",
        str(runtime_include),
        *[str(path) for path in host_sources],
        str(runtime_source),
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
        "executable": str(executable),
        "build_directory": str(output),
        "host_sources": [str(path) for path in host_sources],
        "command": command,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "diagnostic": result.stderr.strip() if result.returncode else "",
    }
