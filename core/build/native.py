import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

from core.backends.host import has_host_code, transform_host_source
from core.cache import CacheStore, fingerprint
from core.diagnostics.compiler import parse_compiler_diagnostics
from core.frontend.regex_parser import read_text
from core.version import MCU_VERSION, RUNTIME_ABI


HOST_SOURCE_EXTENSIONS = {".c", ".cc", ".cpp", ".cxx", ".cu"}
HEADER_EXTENSIONS = {".h", ".hpp"}


def build_native(
    metadata,
    build_output,
    output_directory,
    project_root,
    executable_path=None,
    optimization="-O2",
    timings=None,
    native_objects=None,
    host_flags=None,
    native_link_flags=None,
):
    output = Path(output_directory).resolve()
    host_directory = output / "host"
    object_directory = output / "objects"
    marker = output / ".mcu-native-build"
    manifest_path = output / "build_manifest.json"
    executable = Path(executable_path).resolve() if executable_path else output / _executable_name(metadata)
    cache = CacheStore()
    root = Path(project_root).resolve()

    output.mkdir(parents=True, exist_ok=True)
    host_directory.mkdir(parents=True, exist_ok=True)
    object_directory.mkdir(parents=True, exist_ok=True)
    marker.write_text("MCU native build directory\n", encoding="ascii")

    source_paths = [Path(name).resolve() for name in metadata.get("sources", [])]
    native_objects = [Path(path).resolve() for path in (native_objects or [])]
    host_flags = list(host_flags or [])
    native_link_flags = list(native_link_flags or [])
    compile_context = metadata.get("compile_context", {})
    build_key = _build_key(
        source_paths, native_objects, build_output, optimization, executable, root,
        compile_context, host_flags,
        native_link_flags,
    )
    if _is_noop_build(manifest_path, build_key, executable, output):
        if timings:
            timings.cache("native_executable", True, build_key)
        return _result(
            True,
            executable,
            output,
            [],
            [],
            cache_hit=True,
            metallib=output / "generated.metallib",
        )
    if timings:
        timings.cache("native_executable", False, build_key)

    _write_if_changed(output / "generated.metal", build_output["metal_source"])
    _write_if_changed(output / "routing_table.txt", build_output["routing_table_text"])
    if build_output.get("symbol_manifest_text"):
        _write_if_changed(output / "symbol_manifest.json", build_output["symbol_manifest_text"])

    metal_result = _build_metallib(
        build_output["metal_source"], output, cache, timings, compile_context,
    )
    if not metal_result["ok"]:
        return metal_result

    transformed = _transform_host_sources(source_paths, host_directory)
    if not transformed["sources"] and not native_objects:
        return _failure("host", "No host compilation units remained after CUDA kernel extraction.")

    runtime_result = _build_runtime_library(root, cache, output, optimization, timings)
    if not runtime_result["ok"]:
        return runtime_result

    host_result = _build_host_objects(
        transformed["sources"],
        transformed["headers"],
        object_directory,
        root / "runtime" / "include",
        host_directory,
        cache,
        optimization,
        timings,
        compile_context,
        host_flags,
        transformed.get("include_paths", []),
    )
    if not host_result["ok"]:
        return host_result

    executable.parent.mkdir(parents=True, exist_ok=True)
    link_command = [
        "xcrun",
        "clang++",
        *[str(path) for path in host_result["objects"]],
        *[str(path) for path in native_objects],
        str(runtime_result["library"]),
        *native_link_flags,
        "-framework",
        "Foundation",
        "-framework",
        "Metal",
        "-framework",
        "MetalPerformanceShaders",
        "-o",
        str(executable),
    ]
    with _phase(timings, "native_link"):
        linked = _run(link_command)
    if linked["exception"]:
        return _failure("native_link", linked["exception"], link_command)
    if linked["returncode"] != 0:
        return _failure(
            "native_link",
            linked["stderr"].strip(),
            link_command,
            parse_compiler_diagnostics(linked["stderr"], code="NativeLinkFailed"),
            linked["stderr"],
        )

    manifest_path.write_text(
        json.dumps({"build_key": build_key, "mcu_version": MCU_VERSION}, sort_keys=True),
        encoding="utf-8",
    )
    return _result(
        True,
        executable,
        output,
        transformed["sources"],
        link_command,
        cache_hit=False,
        metallib=metal_result["metallib"],
        metal_command=metal_result["commands"],
    )


def _build_metallib(metal_source, output, cache, timings, compile_context):
    metal_file = output / "generated.metal"
    air_file = output / "generated.air"
    metallib = output / "generated.metallib"
    key = fingerprint({
        "source": metal_source,
        "architecture": platform.machine(),
        "mcu_version": MCU_VERSION,
        "metal_pipeline": 2,
        "compile_context": compile_context,
        "toolchain": _toolchain_identity(),
    })
    if cache.restore("metal", key, "generated.metallib", metallib):
        if timings:
            timings.cache("metal", True, key)
        return {"ok": True, "metallib": metallib, "commands": [], "cache_hit": True}
    if timings:
        timings.cache("metal", False, key)
    metal_command = [
        "xcrun", "-sdk", "macosx", "metal", "-c",
        *[f"-D{value}" for value in compile_context.get("definitions", [])],
        *[f"-U{value}" for value in compile_context.get("undefinitions", [])],
        *[item for path in compile_context.get("include_paths", []) for item in ("-I", path)],
        f"-fmodules-cache-path={output / 'module-cache'}",
        str(metal_file), "-o", str(air_file),
    ]
    metallib_command = [
        "xcrun", "-sdk", "macosx", "metal",
        str(air_file), "-o", str(metallib),
    ]
    with _phase(timings, "metal_compile"):
        compiled = _run(metal_command)
    if compiled["exception"]:
        return _failure("metal", compiled["exception"], metal_command)
    if compiled["returncode"] != 0:
        return _failure(
            "metal", compiled["stderr"].strip(), metal_command,
            parse_compiler_diagnostics(compiled["stderr"], code="MetalCompileError"),
            compiled["stderr"],
        )
    with _phase(timings, "metal_link"):
        linked = _run(metallib_command)
    if linked["exception"]:
        return _failure("metal_link", linked["exception"], metallib_command)
    if linked["returncode"] != 0:
        return _failure(
            "metal_link", linked["stderr"].strip(), metallib_command,
            parse_compiler_diagnostics(linked["stderr"], code="MetalLibraryLinkFailed"),
            linked["stderr"],
        )
    cache.publish("metal", key, "generated.metallib", metallib)
    return {
        "ok": True,
        "metallib": metallib,
        "commands": [metal_command, metallib_command],
        "cache_hit": False,
    }


def _build_runtime_library(root, cache, output, optimization, timings):
    runtime_include = root / "runtime" / "include"
    runtime_sources = sorted((root / "runtime" / "src").glob("*.mm"))
    runtime_inputs = [*runtime_sources, *sorted(runtime_include.glob("*")), root / "runtime" / "src" / "runtime_internal.h"]
    key = fingerprint({
        "files": [(str(path.relative_to(root)), _file_hash(path)) for path in runtime_inputs if path.is_file()],
        "optimization": optimization,
        "architecture": platform.machine(),
        "runtime_abi": RUNTIME_ABI,
        "native_pipeline": 2,
        "toolchain": _toolchain_identity(),
    })
    destination = output / "libmcu_runtime.a"
    if cache.restore("runtime", key, "libmcu_runtime.a", destination):
        if timings:
            timings.cache("runtime", True, key)
        return {"ok": True, "library": destination, "cache_hit": True}
    if timings:
        timings.cache("runtime", False, key)

    with tempfile.TemporaryDirectory(prefix="mcu-runtime-") as temporary:
        temporary_path = Path(temporary)
        objects = []
        compile_jobs = []
        for index, source in enumerate(runtime_sources):
            object_path = temporary_path / f"runtime_{index}.o"
            command = [
                "xcrun", "clang++", "-std=c++17", "-fobjc-arc", optimization,
                "-I", str(runtime_include), "-I", str(root / "runtime" / "src"),
                "-c", str(source), "-o", str(object_path),
            ]
            compile_jobs.append((object_path, command))
        with _phase(timings, "runtime_compile"):
            with ThreadPoolExecutor(max_workers=min(4, len(compile_jobs))) as executor:
                compile_results = list(executor.map(lambda job: _run(job[1]), compile_jobs))
        for (object_path, command), compiled in zip(compile_jobs, compile_results):
            if compiled["exception"]:
                return _failure("runtime_compile", compiled["exception"], command)
            if compiled["returncode"] != 0:
                return _failure(
                    "runtime_compile", compiled["stderr"].strip(), command,
                    parse_compiler_diagnostics(compiled["stderr"], code="RuntimeCompileFailed"),
                    compiled["stderr"],
                )
            objects.append(object_path)
        archive = temporary_path / "libmcu_runtime.a"
        archive_command = ["xcrun", "ar", "rcs", str(archive), *[str(path) for path in objects]]
        with _phase(timings, "runtime_archive"):
            archived = _run(archive_command)
        if archived["exception"] or archived["returncode"] != 0:
            return _failure(
                "runtime_archive",
                archived["exception"] or archived["stderr"].strip(),
                archive_command,
            )
        cache.publish("runtime", key, "libmcu_runtime.a", archive)
    cache.restore("runtime", key, "libmcu_runtime.a", destination)
    return {"ok": True, "library": destination, "cache_hit": False}


def _build_host_objects(
    sources, headers, object_directory, runtime_include, host_directory, cache,
    optimization, timings, compile_context, host_flags, transformed_include_paths,
):
    objects = []
    header_hashes = [(str(path), _file_hash(path)) for path in headers]
    runtime_header_hashes = [
        (str(path.name), _file_hash(path))
        for path in sorted(Path(runtime_include).glob("*.h"))
    ]
    for index, source in enumerate(sources):
        object_path = object_directory / f"host_{index}.o"
        key = fingerprint({
            "source": source.read_text(encoding="utf-8"),
            "headers": header_hashes,
            "runtime_headers": runtime_header_hashes,
            "optimization": optimization,
            "architecture": platform.machine(),
            "runtime_abi": RUNTIME_ABI,
            "compile_context": compile_context,
            "host_flags": host_flags,
            "toolchain": _toolchain_identity(),
        })
        if cache.restore("host", key, "host.o", object_path):
            if timings:
                timings.cache("host", True, key)
            objects.append(object_path)
            continue
        if timings:
            timings.cache("host", False, key)
        command = [
            "xcrun", "clang++", "-std=c++17", "-fobjc-arc", optimization,
            "-I", str(runtime_include), "-I", str(host_directory),
            *[item for path in transformed_include_paths for item in ("-I", path)],
            *[item for path in compile_context.get("include_paths", []) for item in ("-I", path)],
            *[f"-D{value}" for value in compile_context.get("definitions", [])],
            *[f"-U{value}" for value in compile_context.get("undefinitions", [])],
            *host_flags,
            "-c", str(source), "-o", str(object_path),
        ]
        with _phase(timings, "host_compile"):
            compiled = _run(command)
        if compiled["exception"]:
            return _failure("host_compile", compiled["exception"], command)
        if compiled["returncode"] != 0:
            return _failure(
                "host_compile", compiled["stderr"].strip(), command,
                parse_compiler_diagnostics(compiled["stderr"], code="HostCompileError"),
                compiled["stderr"],
            )
        cache.publish("host", key, "host.o", object_path)
        objects.append(object_path)
    return {"ok": True, "objects": objects}


def _transform_host_sources(source_paths, host_directory):
    if not source_paths:
        return {"sources": [], "headers": []}
    source_root = Path(os.path.commonpath([str(path.parent) for path in source_paths]))
    host_sources = []
    headers = []
    for source in source_paths:
        if source.suffix not in HOST_SOURCE_EXTENSIONS | HEADER_EXTENSIONS:
            continue
        transformed = transform_host_source(read_text(source))
        relative = source.relative_to(source_root)
        destination = host_directory / relative
        if source.suffix in HEADER_EXTENSIONS:
            destination.parent.mkdir(parents=True, exist_ok=True)
            _write_if_changed(destination, transformed)
            headers.append(destination)
            continue
        if not has_host_code(transformed):
            continue
        if destination.suffix != ".cpp":
            destination = destination.with_name(destination.name + ".cpp")
        destination.parent.mkdir(parents=True, exist_ok=True)
        _write_if_changed(destination, transformed)
        host_sources.append(destination)
    include_paths = []
    for source in source_paths:
        if source.suffix not in HEADER_EXTENSIONS:
            continue
        candidate = host_directory / source.parent.relative_to(source_root)
        value = str(candidate)
        if value not in include_paths:
            include_paths.append(value)
    return {
        "sources": host_sources,
        "headers": headers,
        "source_root": source_root,
        "include_paths": include_paths,
    }


def _build_key(
    source_paths, native_objects, build_output, optimization, executable, root,
    compile_context, host_flags, native_link_flags,
):
    runtime_files = sorted((root / "runtime").rglob("*"))
    return fingerprint({
        "sources": [(str(path), _file_hash(path)) for path in source_paths],
        "native_objects": [(str(path), _file_hash(path)) for path in native_objects],
        "metal": build_output["metal_source"],
        "routing": build_output["routing_table_text"],
        "optimization": optimization,
        "executable": str(executable),
        "mcu_version": MCU_VERSION,
        "runtime_abi": RUNTIME_ABI,
        "native_pipeline": 2,
        "compile_context": compile_context,
        "host_flags": host_flags,
        "native_link_flags": native_link_flags,
        "toolchain": _toolchain_identity(),
        "runtime_sources": [
            (str(path.relative_to(root)), _file_hash(path))
            for path in runtime_files if path.is_file()
        ],
    })


def _is_noop_build(manifest_path, build_key, executable, output):
    if not manifest_path.is_file() or not executable.is_file():
        return False
    if not (output / "generated.metallib").is_file() or not (output / "routing_table.txt").is_file():
        return False
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8")).get("build_key") == build_key
    except (OSError, json.JSONDecodeError):
        return False


def _write_if_changed(path, content):
    target = Path(path)
    if target.is_file() and target.read_text(encoding="utf-8") == content:
        return
    target.write_text(content, encoding="utf-8")


def _file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def _toolchain_identity():
    commands = {
        "sdk": ["xcrun", "--sdk", "macosx", "--show-sdk-path"],
        "clang": ["xcrun", "clang++", "--version"],
        "metal": ["xcrun", "metal", "--version"],
    }
    identity = {}
    for name, command in commands.items():
        result = _run(command)
        output = result["stdout"] or result["stderr"]
        identity[name] = output.splitlines()[0] if output else "unavailable"
    return identity


def _executable_name(metadata):
    return Path(metadata.get("entry", "mcu_app")).stem or "mcu_app"


def _run(command):
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exception": None,
        }
    except OSError as exc:
        return {"returncode": 2, "stdout": "", "stderr": "", "exception": str(exc)}


def _result(ok, executable, output, host_sources, command, cache_hit, metallib, metal_command=None):
    return {
        "ok": ok,
        "phase": "complete",
        "executable": str(executable),
        "build_directory": str(output),
        "host_sources": [str(path) for path in host_sources],
        "command": command,
        "metal_command": metal_command or [],
        "metallib": str(metallib),
        "cache_hit": cache_hit,
        "diagnostics": [],
        "stdout": "",
        "stderr": "",
        "diagnostic": "",
    }


def _failure(phase, diagnostic, command=None, diagnostics=None, stderr=""):
    return {
        "ok": False,
        "phase": phase,
        "diagnostic": diagnostic,
        "diagnostics": diagnostics or [],
        "command": command or [],
        "stderr": stderr,
    }


class _phase:
    def __init__(self, timings, name):
        self.context = timings.phase(name) if timings else None

    def __enter__(self):
        if self.context:
            return self.context.__enter__()
        return None

    def __exit__(self, *args):
        if self.context:
            return self.context.__exit__(*args)
        return False
