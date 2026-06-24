import os
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

from core.application.pipeline import CommandResult, CompilerPipeline
from core.cache import fingerprint
from core.driver import NvccInvocationError, parse_nvcc_arguments
from core.frontend.project_parser import parse_project
from core.frontend.semantic import parse_semantic_project
from core.metrics import BuildTimings
from core.objects import McuObjectError, create_mcu_object, load_mcu_object, write_mcu_object
from core.passes.coarsening import plan_coarsening
from core.passes.validation import validate
from core.version import MCU_VERSION


def execute_compile(project_root, arguments, artifact_path="mcu_artifacts.zip"):
    try:
        invocation = parse_nvcc_arguments(arguments)
    except NvccInvocationError as exc:
        return _error_result("compile", exc.diagnostic(), 2)
    return execute_invocation(project_root, invocation, command="compile", artifact_path=artifact_path)


def execute_invocation(project_root, invocation, command="compile", artifact_path="mcu_artifacts.zip"):
    timings = BuildTimings()
    if invocation.mode in {"compile", "device_compile"}:
        source = invocation.sources[0]
        if not Path(source).is_file():
            return _error_result(command, _diagnostic("InvalidTargetPath", f"Source does not exist: {source}", source), 2)
        payload = create_mcu_object(source, invocation)
        dependency_paths = [item["path"] for item in payload["dependencies"]]
        with timings.phase("frontend"):
            metadata = parse_project(
                dependency_paths,
                include_paths=invocation.include_paths,
                definitions=invocation.definitions,
                undefinitions=invocation.undefinitions,
            )
            diagnostics = validate(metadata)
            coarsening = plan_coarsening(metadata)
            diagnostics.extend(coarsening["diagnostics"])
            if coarsening["recovered_hardware_limit"]:
                diagnostics = [
                    item for item in diagnostics
                    if item.get("code") != "HardwareLimitExceeded"
                ]
            semantic = parse_semantic_project(metadata)
            if not semantic.complete:
                diagnostics.extend(
                    item for item in semantic.diagnostics
                    if item.get("severity") == "error"
                )
        if any(item.get("severity") == "error" for item in diagnostics):
            return CommandResult(1, {
                "ok": False,
                "command": command,
                "metadata": metadata,
                "diagnostics": diagnostics,
                "timings": timings.to_dict(),
            })
        payload["semantic_ir"] = semantic.project.to_dict() if semantic.project else None
        payload["coarsening"] = coarsening["plans"]
        payload["translation_unit"] = {
            "functions": metadata.get("functions", []),
            "includes": metadata.get("includes", []),
        }
        with timings.phase("object_write"):
            write_mcu_object(invocation.output, payload)
            _write_dependency_file(invocation, payload)
        return CommandResult(0, {
            "ok": True,
            "command": command,
            "metadata": _driver_metadata(source),
            "diagnostics": [],
            "object": {"path": invocation.output, "schema_version": payload["schema_version"]},
            "timings": timings.to_dict(),
        })

    try:
        sources, native_objects, contexts, object_payloads = _link_inputs(invocation)
    except McuObjectError as exc:
        return _error_result(command, exc.diagnostic(), 1)
    if not sources:
        return _error_result(
            command,
            _diagnostic("InvalidNvccInvocation", "Link invocation contains no MCU source contribution."),
            2,
        )
    conflict = _compile_context_conflict(contexts, invocation)
    if conflict:
        return _error_result(command, _diagnostic("IncompatibleMcuObject", conflict), 1)
    output = Path(invocation.output).resolve()
    sidecar = Path(f"{output}.mcu")
    fast_key = _fast_link_key(invocation, native_objects)
    if not invocation.sources and object_payloads and _fast_link_hit(sidecar, output, fast_key):
        timings.cache("driver_link", True, fast_key)
        return CommandResult(0, {
            "ok": True,
            "command": command,
            "metadata": _driver_metadata(sources[0]),
            "diagnostics": [],
            "artifact": artifact_path,
            "build": {
                "metal_file": "generated.metal",
                "routing_table": "routing_table.txt",
                "symbol_manifest": "symbol_manifest.json",
                "device_function_count": sum(
                    function.get("kind") in {"DEVICE_FUNC", "HOST_DEVICE_FUNC"}
                    for payload in object_payloads
                    for function in payload.get("translation_unit", {}).get("functions", [])
                ),
                "translated_kernel_count": sum(
                    sum(
                        function.get("kind") == "KERNEL"
                        for function in payload.get("translation_unit", {}).get("functions", [])
                    )
                    for payload in object_payloads
                ),
                "coarsening": {"requested_factor": None, "plans": {}},
                "native": {
                    "ok": True,
                    "phase": "complete",
                    "executable": str(output),
                    "build_directory": str(sidecar),
                    "metallib": str(sidecar / "generated.metallib"),
                    "cache_hit": True,
                    "diagnostics": [],
                },
                "objects": invocation.objects,
                "sidecar": str(sidecar),
            },
            "timings": timings.to_dict(),
        })
    timings.cache("driver_link", False, fast_key)
    pipeline = CompilerPipeline(project_root=project_root, artifact_path=artifact_path)
    result = pipeline.build(
        sources,
        native=True,
        output=sidecar,
        executable=output,
        include_paths=_stable_unique([*invocation.include_paths, *contexts["include_paths"]]),
        definitions=_stable_unique([*invocation.definitions, *contexts["definitions"]]),
        undefinitions=_stable_unique([*invocation.undefinitions, *contexts["undefinitions"]]),
        optimization=(
            invocation.optimization
            if invocation.optimization_explicit or not contexts["optimizations"]
            else contexts["optimizations"][-1]
        ),
        host_flags=_stable_unique([*invocation.host_flags, *contexts["host_flags"]]),
        command=command,
        native_objects=native_objects,
        native_link_flags=[
            *[f"-L{path}" for path in invocation.library_paths],
            *[f"-l{name}" for name in invocation.libraries],
        ],
    )
    if result.payload.get("build"):
        result.payload["build"]["objects"] = invocation.objects
        result.payload["build"]["sidecar"] = str(sidecar)
    if result.exit_code == 0:
        sidecar.mkdir(parents=True, exist_ok=True)
        (sidecar / "driver_manifest.json").write_text(
            json.dumps({"key": fast_key, "mcu_version": MCU_VERSION}, sort_keys=True),
            encoding="utf-8",
        )
    return result


def execute_run(project_root, compile_arguments, program_arguments):
    result = execute_compile(project_root, compile_arguments)
    if result.exit_code != 0:
        return result
    executable = result.payload.get("build", {}).get("native", {}).get("executable")
    if not executable:
        return _error_result("run", _diagnostic("InvalidRunTarget", "Compile did not produce an executable."), 2)
    process = subprocess.run(
        [executable, *program_arguments],
        text=True,
        capture_output=True,
        check=False,
    )
    result.payload["command"] = "run"
    result.payload["program"] = {
        "arguments": program_arguments,
        "exit_code": process.returncode,
        "stdout": process.stdout,
        "stderr": process.stderr,
    }
    result.payload["ok"] = process.returncode == 0
    return CommandResult(process.returncode, result.payload)


def execute_make(project_root, make_arguments, artifact_path="mcu_artifacts.zip"):
    timings = BuildTimings()
    root = Path(project_root).resolve()
    with tempfile.TemporaryDirectory(prefix="mcu-nvcc-") as temporary:
        shim = Path(temporary) / "nvcc"
        shim.write_text(
            "#!/usr/bin/env bash\n"
            f"exec {shlex.quote(sys.executable)} {shlex.quote(str(root / 'mcu'))} nvcc-driver -- \"$@\"\n",
            encoding="utf-8",
        )
        shim.chmod(0o755)
        environment = os.environ.copy()
        environment["PATH"] = temporary + os.pathsep + environment.get("PATH", "")
        command = ["make", *make_arguments]
        with timings.phase("make"):
            try:
                process = subprocess.run(
                    command,
                    text=True,
                    capture_output=True,
                    check=False,
                    env=environment,
                )
            except OSError as exc:
                return _error_result("make", _diagnostic("MakeInvocationFailed", str(exc)), 2)
    ok = process.returncode == 0
    diagnostics = [] if ok else [_diagnostic(
        "MakeInvocationFailed",
        f"make exited with status {process.returncode}",
    )]
    return CommandResult(process.returncode, {
        "ok": ok,
        "command": "make",
        "metadata": _driver_metadata(str(Path.cwd())),
        "diagnostics": diagnostics,
        "make": {
            "arguments": make_arguments,
            "exit_code": process.returncode,
            "stdout": process.stdout,
            "stderr": process.stderr,
        },
        "timings": timings.to_dict(),
    })


def _link_inputs(invocation):
    sources = list(invocation.sources)
    native_objects = []
    contexts = {
        "include_paths": [],
        "definitions": [],
        "undefinitions": [],
        "host_flags": [],
        "optimizations": [],
    }
    object_payloads = []
    for path in invocation.objects:
        target = Path(path)
        if target.suffix == ".a":
            native_objects.append(str(target))
            continue
        try:
            payload = load_mcu_object(target)
        except McuObjectError as exc:
            if exc.code == "IncompatibleMcuObject" and target.is_file() and not target.read_bytes().startswith(b"MCUOBJ\n"):
                native_objects.append(str(target.resolve()))
                continue
            raise
        sources.append(payload["source"])
        object_payloads.append(payload)
        context = payload.get("invocation", {})
        contexts["include_paths"].extend(context.get("include_paths", []))
        contexts["definitions"].extend(context.get("definitions", []))
        contexts["undefinitions"].extend(context.get("undefinitions", []))
        contexts["host_flags"].extend(context.get("host_flags", []))
        contexts["optimizations"].append(context.get("optimization", "-O2"))
    return _stable_unique(sources), native_objects, contexts, object_payloads


def _fast_link_key(invocation, native_objects):
    files = []
    for path in [*invocation.objects, *native_objects]:
        target = Path(path)
        if target.is_file():
            files.append((str(target.resolve()), hashlib.sha256(target.read_bytes()).hexdigest()))
    return fingerprint({
        "files": files,
        "output": invocation.output,
        "include_paths": invocation.include_paths,
        "definitions": invocation.definitions,
        "undefinitions": invocation.undefinitions,
        "optimization": invocation.optimization,
        "host_flags": invocation.host_flags,
        "library_paths": invocation.library_paths,
        "libraries": invocation.libraries,
        "mcu_version": MCU_VERSION,
    })


def _fast_link_hit(sidecar, executable, key):
    manifest = sidecar / "driver_manifest.json"
    if not manifest.is_file() or not executable.is_file():
        return False
    if not (sidecar / "generated.metallib").is_file() or not (sidecar / "routing_table.txt").is_file():
        return False
    try:
        return json.loads(manifest.read_text(encoding="utf-8")).get("key") == key
    except (OSError, json.JSONDecodeError):
        return False


def _compile_context_conflict(contexts, invocation):
    definitions = {}
    for definition in contexts["definitions"]:
        name, _, value = definition.partition("=")
        normalized = value or "1"
        if name in definitions and definitions[name] != normalized:
            return f"MCU objects define '{name}' with incompatible values."
        definitions[name] = normalized
    undefined = set(contexts["undefinitions"])
    overlap = undefined.intersection(definitions)
    if overlap:
        return f"MCU objects both define and undefine '{sorted(overlap)[0]}'."
    optimizations = set(contexts["optimizations"])
    if len(optimizations) > 1 and not invocation.optimization_explicit:
        return "MCU objects use different optimization levels; pass an explicit link optimization."
    return None


def _write_dependency_file(invocation, payload):
    if not invocation.dependency_mode:
        return
    target = invocation.dependency_target or invocation.output
    dependency_file = invocation.dependency_file or str(Path(invocation.output).with_suffix(".d"))
    dependencies = " ".join(_make_escape(item["path"]) for item in payload["dependencies"])
    path = Path(dependency_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{_make_escape(target)}: {dependencies}\n", encoding="utf-8")


def _make_escape(value):
    return str(value).replace(" ", "\\ ")


def _stable_unique(values):
    return list(dict.fromkeys(values))


def _driver_metadata(entry):
    return {
        "entry": str(entry),
        "parser": "nvcc_driver",
        "sources": [str(entry)] if Path(entry).is_file() else [],
        "kernels": [],
        "launches": [],
        "parser_diagnostics": [],
    }


def _diagnostic(code, message, file=None):
    return {
        "code": code,
        "severity": "error",
        "message": message,
        "file": file,
        "line": None,
        "column": None,
    }


def _error_result(command, diagnostic, exit_code):
    return CommandResult(exit_code, {
        "ok": False,
        "command": command,
        "metadata": _driver_metadata("-"),
        "diagnostics": [diagnostic],
    })
