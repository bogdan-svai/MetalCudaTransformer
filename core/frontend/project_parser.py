from pathlib import Path

from core.frontend.clang_probe import probe_with_clang
from core.frontend.regex_parser import discover_sources, extract_functions, extract_includes, extract_launches, read_text
from core.ir.models import ProjectIR


def parse_project(entry_path, verbose=False):
    entry = Path(entry_path)
    sources = discover_sources(entry)
    parser_name, parser_diagnostics = probe_with_clang(sources, read_text)
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

    return project.to_dict()
