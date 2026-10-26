"""Integrate proven host loops into generated artifacts and native source copies."""

import json
from pathlib import Path

from core.backends.host_parallel import generate_loop


def integrate_host_loops(report, generated, mode):
    """Return source overrides; never change the user's files or uncertain loops."""
    plans_by_file = {}
    source_data = {}
    for item in report["loops"]:
        if item["status"] not in {"eligible", "conditional"}:
            continue
        path = Path(item["node"]["file"]).resolve()
        # A header may have several preprocessor interpretations across units.
        if path.suffix in {".h", ".hpp"}:
            item["reasons"].append("HostHeaderContextUnsupported")
            item["status"] = "unproven"
            continue
        if str(path) not in source_data:
            source_data[str(path)] = path.read_bytes()
        plan = generate_loop(item, source_data[str(path)])
        plans_by_file.setdefault(str(path), []).append((item, plan))
    overrides = {}
    for name, plans in plans_by_file.items():
        data = source_data[name]
        previous = len(data)
        chunks = []
        for item, plan in sorted(plans, key=lambda pair: pair[1]["start"], reverse=True):
            if plan["end"] > previous:
                raise ValueError("Overlapping host-loop edits")
            chunks.extend([data[plan["end"] : previous], plan["host_replacement"].encode()])
            previous = plan["start"]
        chunks.append(data[:previous])
        overrides[name] = "#include <stdexcept>\n" + b"".join(reversed(chunks)).decode("utf-8")
        for item, plan in plans:
            routes = generated["routing_table"]["translated_kernels"]
            if plan["name"] in routes:
                raise ValueError("Generated host kernel collision")
            routes[plan["name"]] = plan["route"]
            generated["metal_source"] += "\n" + plan["metal_source"]
            item["execution"] = "gpu_with_cpu_guard_fallback"
            item["kernel"] = plan["name"]
            item["captures"] = plan["captures"]
            report["transformed_loop_count"] += 1
    generated["routing_table_text"] = json.dumps(
        generated["routing_table"], indent=2, sort_keys=True
    )
    generated["host_source_overrides"] = overrides
    generated["host_parallel_cache"] = {
        "mode": mode,
        "schema": report["schema_version"],
        "sources": overrides,
    }
