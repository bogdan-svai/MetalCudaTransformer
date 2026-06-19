from dataclasses import dataclass
from pathlib import Path

from core.backends.metal import transpile
from core.build.artifacts import write_artifact
from core.build.native import build_native
from core.frontend.project_parser import parse_project
from core.passes.coarsening import plan_coarsening
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

        metadata = parse_project(target, verbose=verbose)
        diagnostics = validate(metadata)
        has_errors = _has_errors(diagnostics)
        payload = self._base_payload("check", metadata, diagnostics, not has_errors)
        self._write(payload)
        return CommandResult(1 if has_errors else 0, payload)

    def build(self, path, verbose=False, coarsening=None, native=False, output=".mcu/build"):
        target = Path(path)
        if not target.exists():
            return CommandResult(2, invalid_path_payload("build", path))

        metadata = parse_project(target, verbose=verbose)
        diagnostics = validate(metadata)
        optimization = plan_coarsening(metadata, requested_factor=coarsening)
        diagnostics.extend(optimization["diagnostics"])
        if optimization["recovered_hardware_limit"]:
            diagnostics = [
                item for item in diagnostics
                if item["code"] != "HardwareLimitExceeded"
            ]

        payload = self._base_payload("build", metadata, diagnostics, not _has_errors(diagnostics))
        if _has_errors(diagnostics):
            self._write(payload)
            return CommandResult(1, payload)

        generated = transpile(metadata, coarsening_plans=optimization["plans"])
        payload["build"] = {
            "metal_file": "generated.metal",
            "routing_table": "routing_table.txt",
            "translated_kernel_count": len(generated["routing_table"]["translated_kernels"]),
            "coarsening": {
                "requested_factor": coarsening,
                "plans": optimization["plans"],
            },
        }
        extras = {
            "generated.metal": generated["metal_source"],
            "routing_table.txt": generated["routing_table_text"],
        }
        self._write(payload, extras)

        if native:
            native_result = build_native(
                metadata,
                generated,
                output_directory=output,
                project_root=self.project_root,
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
                self._write(payload, extras)
                return CommandResult(1, payload)
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
