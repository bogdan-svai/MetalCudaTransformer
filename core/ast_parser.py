from pathlib import Path

from core.clang_probe import probe_with_clang
from core.fallback import discover_sources, extract_functions, extract_includes, extract_launches, read_text


def parse_project(entry_path, verbose=False):
    entry = Path(entry_path)
    sources = discover_sources(entry)
    parser_name, parser_diagnostics = probe_with_clang(sources, read_text)
    metadata = {
        "entry": entry.as_posix(),
        "parser": parser_name,
        "sources": [path.as_posix() for path in sources],
        "functions": [],
        "kernels": [],
        "launches": [],
        "includes": [],
        "parser_diagnostics": parser_diagnostics,
    }

    for path in sources:
        text = read_text(path)
        functions = extract_functions(path, text)
        launches = extract_launches(path, text)
        includes = extract_includes(path, text)

        metadata["functions"].extend(function.to_dict() for function in functions)
        metadata["launches"].extend(launch.to_dict() for launch in launches)
        metadata["includes"].extend(includes)

    metadata["kernels"] = [
        function for function in metadata["functions"]
        if function["kind"] == "KERNEL"
    ]

    known_kernels = {kernel["name"] for kernel in metadata["kernels"]}
    for launch in metadata["launches"]:
        if launch["kernel"] not in known_kernels:
            metadata["parser_diagnostics"].append({
                "severity": "warning",
                "message": f"Launch target '{launch['kernel']}' has no detected __global__ definition.",
                "file": launch["file"],
                "line": launch["line"],
            })

    if verbose:
        metadata["parser_diagnostics"].append({
            "severity": "info",
            "message": f"Parsed {len(metadata['sources'])} source files with regex fallback parser.",
            "file": None,
            "line": None,
        })

    return metadata
