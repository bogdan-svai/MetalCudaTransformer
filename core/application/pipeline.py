from dataclasses import dataclass
from pathlib import Path

from core.backends.metal import transpile
from core.build.artifacts import write_artifact
from core.build.native import build_native
from core.frontend.project_parser import parse_project
from core.frontend.semantic import parse_semantic_project
from core.metrics import BuildTimings
from core.passes.coarsening import plan_coarsening
from core.passes.device_linker import build_device_link_plan
from core.passes.validation import validate


@dataclass
class CommandResult:
    exit_code: int
    payload: dict


class CompilerPipeline:
    def __init__(self, project_root, artifact_path="mcu_artifacts.zip"):
        self.project_root = Path(project_root)
        self.artifact_path = artifact_path

    def check(self, path, verbose=False):
        target = Path(path)
        if not target.exists():
            return CommandResult(2, invalid_path_payload("check", path))

        timings = BuildTimings()
        with timings.phase("frontend"):
            metadata = parse_project(target, verbose=verbose)
            diagnostics = validate(metadata)
        has_errors = _has_errors(diagnostics)
        payload = self._base_payload("check", metadata, diagnostics, not has_errors)
        payload["timings"] = timings.to_dict()
        self._write(payload)
        return CommandResult(1 if has_errors else 0, payload)

    def build(
        self,
        path,
        verbose=False,
        coarsening=None,
        native=False,
        output=".mcu/build",
        executable=None,
        include_paths=None,
        definitions=None,
        undefinitions=None,
        host_flags=None,
        optimization="-O2",
        command="build",
        native_objects=None,
        native_link_flags=None,
    ):
        inputs = list(path) if isinstance(path, (list, tuple)) else [path]
        missing = next((item for item in inputs if not Path(item).exists()), None)
        if missing is not None:
            return CommandResult(2, invalid_path_payload(command, missing))

        timings = BuildTimings()
        parse_target = inputs if len(inputs) > 1 else inputs[0]
        with timings.phase("frontend"):
            metadata = parse_project(
                parse_target,
                verbose=verbose,
                include_paths=include_paths,
                definitions=definitions,
                undefinitions=undefinitions,
            )
            diagnostics = validate(metadata)
            semantic = parse_semantic_project(metadata)
        semantic_errors = [
            item for item in semantic.diagnostics
            if item.get("severity") == "error"
        ]
        requires_semantic = native or any(
            function.get("kind") in {"DEVICE_FUNC", "HOST_DEVICE_FUNC"}
            for function in metadata.get("functions", [])
        )
        if requires_semantic and not semantic.complete:
            diagnostics.extend(
                item for item in semantic_errors
                if item not in diagnostics
            )

        link_plan = None
        if semantic.project and semantic.complete:
            with timings.phase("device_link"):
                link_plan = build_device_link_plan(semantic.project)
                diagnostics.extend(link_plan.diagnostics)
        with timings.phase("optimization"):
            coarsening_result = plan_coarsening(metadata, requested_factor=coarsening)
        diagnostics.extend(coarsening_result["diagnostics"])
        if coarsening_result["recovered_hardware_limit"]:
            diagnostics = [
                item for item in diagnostics
                if item["code"] != "HardwareLimitExceeded"
            ]

        payload = self._base_payload(command, metadata, diagnostics, not _has_errors(diagnostics))
        if _has_errors(diagnostics):
            payload["timings"] = timings.to_dict()
            self._write(payload)
            return CommandResult(1, payload)

        with timings.phase("codegen"):
            generated = transpile(
                metadata,
                coarsening_plans=coarsening_result["plans"],
                semantic_project=semantic.project if semantic.complete else None,
                link_plan=link_plan,
            )
        payload["build"] = {
            "metal_file": "generated.metal",
            "routing_table": "routing_table.txt",
            "symbol_manifest": "symbol_manifest.json" if generated.get("symbol_manifest_text") else None,
            "device_function_count": sum(
                function.get("kind") in {"DEVICE_FUNC", "HOST_DEVICE_FUNC"}
                for function in metadata.get("functions", [])
            ),
            "translated_kernel_count": len(generated["routing_table"]["translated_kernels"]),
            "coarsening": {
                "requested_factor": coarsening,
                "plans": coarsening_result["plans"],
            },
        }
        extras = {
            "generated.metal": generated["metal_source"],
            "routing_table.txt": generated["routing_table_text"],
        }
        if generated.get("symbol_manifest_text"):
            extras["symbol_manifest.json"] = generated["symbol_manifest_text"]

        if native:
            with timings.phase("native_build"):
                native_result = build_native(
                    metadata,
                    generated,
                    output_directory=output,
                    project_root=self.project_root,
                    executable_path=executable,
                    optimization=optimization,
                    timings=timings,
                    native_objects=native_objects,
                    host_flags=host_flags,
                    native_link_flags=native_link_flags,
                )
            payload["build"]["native"] = native_result
            if not native_result["ok"]:
                payload["ok"] = False
                native_diagnostics = native_result.get("diagnostics", [])
                if native_diagnostics:
                    payload["diagnostics"].extend(native_diagnostics)
                else:
                    payload["diagnostics"].append({
                        "code": "NativeCompileFailed",
                        "severity": "error",
                        "message": native_result["diagnostic"],
                        "file": None,
                        "line": None,
                        "column": None,
                    })
                payload["timings"] = timings.to_dict()
                self._write(payload, extras)
                return CommandResult(1, payload)

        payload["timings"] = timings.to_dict()
        self._write(payload, extras)
        return CommandResult(0, payload)

    def _base_payload(self, command, metadata, diagnostics, ok):
        return {
            "ok": ok,
            "command": command,
            "metadata": metadata,
            "diagnostics": diagnostics,
            "artifact": self.artifact_path,
        }

    def _write(self, payload, extras=None):
        write_artifact(payload, artifact_path=self.artifact_path, extra_files=extras)


def invalid_path_payload(command, path):
    return {
        "ok": False,
        "command": command,
        "metadata": {},
        "diagnostics": [{
            "code": "InvalidTargetPath",
            "severity": "error",
            "message": f"Target path does not exist: {path}",
            "file": str(path),
            "line": None,
            "column": None,
        }],
    }


def _has_errors(diagnostics):
    return any(item["severity"] == "error" for item in diagnostics)
