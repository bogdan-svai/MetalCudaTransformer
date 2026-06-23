import json


SUCCESS = "\033[92m[SUCCESS]\033[0m"
WARNING = "\033[93m[WARNING]\033[0m"
ERROR = "\033[91m[CRITICAL ERROR]\033[0m"


def emit(payload, as_json):
    if as_json:
        print(json.dumps(payload, sort_keys=True))
        return

    metadata = payload.get("metadata", {})
    print(f"MCU {_stage_name(payload)}: {metadata.get('entry', '-')}")
    print(f"Parser: {metadata.get('parser', '-')}")
    print(f"Sources: {len(metadata.get('sources', []))}")
    print(f"Kernels: {len(metadata.get('kernels', []))}")
    print(f"Launches: {len(metadata.get('launches', []))}")

    for item in metadata.get("parser_diagnostics", []):
        label = WARNING if item.get("severity") == "warning" else "[INFO]"
        print(f"{label} {format_location(item)}{item['message']}")

    diagnostics = payload.get("diagnostics", [])
    if not diagnostics:
        print(f"{SUCCESS} Sanity checks passed.")
    for item in diagnostics:
        label = ERROR if item["severity"] == "error" else WARNING
        print(f"{label} {format_location(item)}{item['message']}")

    if payload.get("object"):
        print(f"MCU object: {payload['object']['path']}")
    if payload.get("make"):
        if payload["make"].get("stdout"):
            print(payload["make"]["stdout"], end="")
        if payload["make"].get("stderr"):
            print(payload["make"]["stderr"], end="")
    if payload.get("program"):
        if payload["program"].get("stdout"):
            print(payload["program"]["stdout"], end="")
        if payload["program"].get("stderr"):
            print(payload["program"]["stderr"], end="")
    timings = payload.get("timings", {}).get("phases_seconds", {})
    if timings:
        print(f"Time: {timings.get('total', 0):.3f}s")

    if not payload.get("artifact"):
        return
    artifact_contents = "ast_output.txt"
    if payload.get("build"):
        artifact_contents += ", generated.metal, routing_table.txt"
        if payload["build"].get("symbol_manifest"):
            artifact_contents += ", symbol_manifest.json"
        print(f"Translated kernels: {payload['build']['translated_kernel_count']}")
        for kernel, plan in payload["build"]["coarsening"]["plans"].items():
            if plan["coarsening_factor"] > 1:
                print(
                    f"Coarsening: {kernel} factor={plan['coarsening_factor']} "
                    f"physical_block={plan['physical_block']} mode={plan['mode']}"
                )
        native = payload["build"].get("native")
        if native and native.get("ok"):
            print(f"Native executable: {native['executable']}")
    print(f"Artifact: {payload['artifact']} -> {artifact_contents}")


def format_location(item):
    file = item.get("file")
    line = item.get("line")
    column = item.get("column")
    if not file:
        return ""
    if line and column:
        return f"{file}:{line}:{column}: "
    if line:
        return f"{file}:{line}: "
    return f"{file}: "


def _stage_name(payload):
    if payload.get("command") in {"compile", "run", "make", "nvcc-driver"}:
        return f"Stage 7 {payload['command']}"
    if payload.get("command") != "build":
        return "Stage 1 check"
    native = payload.get("build", {}).get("native")
    if native:
        if payload.get("build", {}).get("device_function_count", 0):
            return "Stage 6 native build"
        return "Stage 5 native build"
    plans = payload.get("build", {}).get("coarsening", {}).get("plans", {})
    optimized = any(plan["coarsening_factor"] > 1 for plan in plans.values())
    return "Stage 3 optimized build" if optimized else "Stage 2 build"
