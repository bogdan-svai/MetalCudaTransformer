from pathlib import Path

from core.application.pipeline import CommandResult, invalid_path_payload
from core.frontend.project_parser import parse_project
from core.frontend.semantic import parse_semantic_project
from core.metrics import BuildTimings
from core.passes.validation import validate, validate_semantic_project
from core.tooling.capabilities import capability_payload


def analyze_path(path, verbose=False, auto_parallel="off"):
    if auto_parallel not in {"off", "report"}:
        raise ValueError("auto_parallel must be off or report")
    target = Path(path)
    if not target.exists():
        return CommandResult(2, invalid_path_payload("analyze", path))

    timings = BuildTimings()
    with timings.phase("frontend"):
        metadata = parse_project(target, verbose=verbose)
        requires_semantic = any(
            function.get("kind") in {"KERNEL", "DEVICE_FUNC", "HOST_DEVICE_FUNC"}
            for function in metadata.get("functions", [])
        )
        diagnostics = validate(
            metadata,
            validate_barriers=not requires_semantic,
            validate_shared=not requires_semantic,
        )
        semantic = parse_semantic_project(metadata)
        timings.cache("semantic", semantic.cached)
    if semantic.project and semantic.complete:
        diagnostics.extend(validate_semantic_project(semantic.project, metadata))
    elif requires_semantic:
        diagnostics.extend(item for item in semantic.diagnostics if item.get("severity") == "error")
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
                item for item in capabilities["library_adapters"] if item["status"] == "supported"
            ],
        },
        "timings": timings.to_dict(),
    }
    if auto_parallel == "report":
        from core.frontend.host_loops import analyze_host_loops

        with timings.phase("host_loop_analysis"):
            host_report = analyze_host_loops(metadata)
        payload["auto_parallel"] = host_report
        payload["diagnostics"].extend(host_report["diagnostics"])
        ok = ok and host_report["complete"]
        payload["ok"] = ok
        payload["timings"] = timings.to_dict()
    return CommandResult(0 if ok else 1, payload)
