from pathlib import Path

from core.frontend.clang_probe import probe_with_clang
from core.frontend.regex_parser import discover_sources, extract_functions, extract_includes, extract_launches, read_text
from core.ir.models import ProjectIR


def parse_project(entry_path, verbose=False, include_paths=None, definitions=None, undefinitions=None):
    entries = list(entry_path) if isinstance(entry_path, (list, tuple)) else [entry_path]
    entry = Path(entries[0])
    if len(entries) == 1:
        sources = discover_sources(entry)
    else:
        sources = _explicit_sources(entries, include_paths or [])
    compile_context = {
        "include_paths": list(include_paths or []),
        "definitions": list(definitions or []),
        "undefinitions": list(undefinitions or []),
    }
    parser_name, parser_diagnostics = probe_with_clang(sources, read_text, compile_context)
    project = ProjectIR(
        entry=entry.as_posix(),
        parser=parser_name,
        sources=[path.as_posix() for path in sources],
        parser_diagnostics=parser_diagnostics,
    )

    for path in sources:
        text = read_text(path)
        functions = extract_functions(path, text)
        launches = extract_launches(path, text)
        includes = extract_includes(path, text)

        project.functions.extend(functions)
        project.launches.extend(launches)
        project.includes.extend(includes)

    known_kernels = {kernel.name for kernel in project.kernels}
    for launch in project.launches:
        if launch.kernel not in known_kernels:
            project.parser_diagnostics.append({
                "severity": "warning",
                "message": f"Launch target '{launch.kernel}' has no detected __global__ definition.",
                "file": launch.file,
                "line": launch.line,
            })

    if verbose:
        project.parser_diagnostics.append({
            "severity": "info",
            "message": f"Parsed {len(project.sources)} source files with regex fallback parser.",
            "file": None,
            "line": None,
        })

    result = project.to_dict()
    result["compile_context"] = compile_context
    return result


def _explicit_sources(entries, include_paths):
    sources = []
    visited = set()

    def visit(path):
        resolved = Path(path).resolve()
        if resolved in visited or not resolved.is_file():
            return
        visited.add(resolved)
        sources.append(resolved)
        for include in extract_includes(resolved, read_text(resolved)):
            name = include["name"]
            candidates = [resolved.parent / name, *[Path(root) / name for root in include_paths]]
            local = next((candidate for candidate in candidates if candidate.is_file()), None)
            if local:
                visit(local)

    for entry in entries:
        visit(entry)
    return sorted(sources)
