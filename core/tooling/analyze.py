from pathlib import Path

from core.application.pipeline import CommandResult, invalid_path_payload
from core.frontend.project_parser import parse_project
from core.metrics import BuildTimings
from core.passes.validation import validate
from core.tooling.capabilities import capability_payload


def analyze_path(path, verbose=False):
    target = Path(path)
    if not target.exists():
        return CommandResult(2, invalid_path_payload("analyze", path))

    timings = BuildTimings()
    with timings.phase("frontend"):
        metadata = parse_project(target, verbose=verbose)
        diagnostics = validate(metadata)
    ok = not any(item.get("severity") == "error" for item in diagnostics)
    capabilities = capability_payload()
    payload = {
        "ok": ok,
        "command": "analyze",
        "metadata": metadata,
        "diagnostics": diagnostics,
        "capabilities": {
            "contract": capabilities["contract"],
            "mcu_version": capabilities["mcu_version"],
            "runtime_abi": capabilities["runtime_abi"],
            "supported_runtime_api_count": len(capabilities["runtime_apis"]),
            "supported_library_adapters": [
                item
                for item in capabilities["library_adapters"]
                if item["status"] == "supported"
            ],
        },
        "timings": timings.to_dict(),
    }
    return CommandResult(0 if ok else 1, payload)
