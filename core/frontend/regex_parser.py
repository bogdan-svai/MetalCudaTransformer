import re
from functools import lru_cache
from pathlib import Path

from core.ir.models import FunctionInfo, LaunchInfo

SOURCE_EXTENSIONS = {".cpp", ".cc", ".cxx", ".cu", ".hpp", ".h"}
IGNORED_SOURCE_DIRECTORIES = {".git", ".mcu", "__pycache__"}

__all__ = [
    "discover_sources",
    "extract_functions",
    "extract_includes",
    "extract_launches",
    "find_matching_brace",
    "normalize_expr",
    "read_text",
    "resolve_expr",
    "split_top_level",
    "strip_comments",
]


def read_text(path):
    target = Path(path)
    stat = target.stat()
    return _read_text_cached(str(target.resolve()), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=256)
def _read_text_cached(path, _mtime_ns, _size):
    try:
        return Path(path).read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return Path(path).read_text(encoding="latin-1")


def line_for_offset(text, offset):
    return text.count("\n", 0, offset) + 1


def strip_comments(text):
    without_blocks = re.sub(
        r"/\*.*?\*/",
        lambda match: "\n" * match.group(0).count("\n"),
        text,
        flags=re.DOTALL,
    )
    return re.sub(r"//[^\n]*", "", without_blocks)


def split_top_level(text, delimiter=","):
    parts = []
    current = []
    depth = 0
    for char in text:
        if char in "([{<":
            depth += 1
        elif char in ")]}>":
            depth = max(0, depth - 1)
        if char == delimiter and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    if current or text.strip():
        parts.append("".join(current).strip())
    return parts


def find_matching_brace(text, open_index):
    depth = 0
    for index in range(open_index, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return index
    return len(text) - 1


def normalize_expr(expr):
    return re.sub(r"\s+", " ", expr).strip()


def discover_sources(entry_path):
    entry = Path(entry_path)
    root = _source_root(entry)
    if entry.is_dir():
        return sorted(
            path
            for path in entry.rglob("*")
            if path.is_file() and path.suffix in SOURCE_EXTENSIONS and not _is_ignored_source(path)
        )
    if entry.is_file():
        return sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix in SOURCE_EXTENSIONS and not _is_ignored_source(path)
        )
    return []


def _is_ignored_source(path):
    return any(part in IGNORED_SOURCE_DIRECTORIES or part.endswith(".mcu") for part in path.parts) or any(
        (parent / ".mcu-native-build").is_file() for parent in path.parents
    )


def _source_root(entry):
    start = entry if entry.is_dir() else entry.parent
    current = start.resolve()
    candidates = [current, *current.parents]
    for candidate in candidates:
        child_names = {child.name for child in candidate.iterdir() if child.is_dir()}
        if {"app", "src"}.issubset(child_names) or {"src", "kernels"}.issubset(child_names):
            return candidate
    return start


def extract_includes(path, text):
    includes = []
    for match in re.finditer(r'^\s*#include\s*[<"]([^>"]+)[>"]', text, re.MULTILINE):
        includes.append(
            {
                "name": match.group(1),
                "file": path.as_posix(),
                "line": line_for_offset(text, match.start()),
            }
        )
    return includes


def extract_variables(text):
    variables = {}
    pattern = re.compile(
        r"(?:^|[;\n])\s*(?:const\s+)?(?:int|unsigned|uint|size_t|long|auto|dim3|float|double)\s+"
        r"([A-Za-z_]\w*)\s*(?:=\s*([^;]+)|\(([^;]+)\))\s*;",
        re.MULTILINE,
    )
    for match in pattern.finditer(strip_comments(text)):
        value = match.group(2)
        ctor = match.group(3)
        if ctor is not None:
            prefix = "dim3" if "dim3" in match.group(0).split(match.group(1), 1)[0] else ""
            variables[match.group(1)] = normalize_expr(f"{prefix}({ctor})" if prefix else ctor)
        elif value is not None:
            variables[match.group(1)] = normalize_expr(value)
    return variables


def resolve_expr(expr, variables):
    expr = normalize_expr(expr)
    return variables.get(expr, expr)


def extract_functions(path, text):
    functions = []
    clean = strip_comments(text)
    clean = re.sub(r"^\s*#.*$", "", clean, flags=re.MULTILINE)
    pattern = re.compile(
        r"^[ \t]*(?P<prefix>(?:__global__[ \t]+|__host__[ \t]+|__device__[ \t]+|static[ \t]+|inline[ \t]+|extern[ \t]+)*)"
        r"(?P<ret>[A-Za-z_][\w:<>,~*& \t]*?)[ \t]+"
        r"(?P<name>[A-Za-z_]\w*)\s*\((?P<params>[^;{}]*)\)\s*\{",
        re.MULTILINE,
    )
    reserved = {"if", "for", "while", "switch", "catch", "sizeof"}
    for match in pattern.finditer(clean):
        name = match.group("name")
        if name in reserved:
            continue
        open_brace = match.end() - 1
        close_brace = find_matching_brace(clean, open_brace)
        body = clean[open_brace + 1 : close_brace]
        prefix = match.group("prefix")
        if "__global__" in prefix:
            kind = "KERNEL"
        elif "__device__" in prefix and "__host__" in prefix:
            kind = "HOST_DEVICE_FUNC"
        elif "__device__" in prefix:
            kind = "DEVICE_FUNC"
        elif name == "main":
            kind = "MAIN_FUNC"
        else:
            kind = "HOST_FUNC"
        if kind == "HOST_FUNC" and re.search(r"\b(threadIdx|blockIdx|blockDim|gridDim)\b", body):
            kind = "KERNEL"
        params = [
            normalize_expr(part) for part in split_top_level(match.group("params")) if part.strip()
        ]
        functions.append(
            FunctionInfo(
                name=name,
                kind=kind,
                file=path.as_posix(),
                line=line_for_offset(clean, match.start()),
                return_type=normalize_expr(match.group("ret")),
                parameters=params,
                body=body,
                body_line=line_for_offset(clean, open_brace),
            )
        )
    return functions


def extract_launches(path, text):
    launches = []
    variables = extract_variables(text)
    clean = strip_comments(text)

    for start, argument_text in iter_function_calls(clean, "cudaLaunch"):
        args = split_top_level(argument_text)
        if len(args) < 3:
            continue
        launches.append(
            LaunchInfo(
                kernel=normalize_expr(args[0]),
                file=path.as_posix(),
                line=line_for_offset(clean, start),
                grid_expr=resolve_expr(args[1], variables),
                block_expr=resolve_expr(args[2], variables),
                args=[normalize_expr(arg) for arg in args[3:]],
                syntax="cudaLaunch",
            )
        )

    for start, argument_text in iter_function_calls(clean, "cudaLaunchShared"):
        args = split_top_level(argument_text)
        if len(args) < 4:
            continue
        launches.append(
            LaunchInfo(
                kernel=normalize_expr(args[0]),
                file=path.as_posix(),
                line=line_for_offset(clean, start),
                grid_expr=resolve_expr(args[1], variables),
                block_expr=resolve_expr(args[2], variables),
                shared_expr=resolve_expr(args[3], variables),
                args=[normalize_expr(arg) for arg in args[4:]],
                syntax="cudaLaunchShared",
            )
        )

    triple = re.compile(
        r"\b([A-Za-z_]\w*)\s*<<<\s*(.*?)\s*>>>\s*\((.*?)\)\s*;",
        re.DOTALL,
    )
    for match in triple.finditer(clean):
        configuration = [item.strip() for item in split_top_level(match.group(2))]
        if len(configuration) < 2:
            continue
        launches.append(
            LaunchInfo(
                kernel=match.group(1),
                file=path.as_posix(),
                line=line_for_offset(clean, match.start()),
                grid_expr=resolve_expr(configuration[0], variables),
                block_expr=resolve_expr(configuration[1], variables),
                shared_expr=(
                    resolve_expr(configuration[2], variables) if len(configuration) >= 3 else "0"
                ),
                stream_expr=(
                    resolve_expr(configuration[3], variables) if len(configuration) >= 4 else "0"
                ),
                args=[
                    normalize_expr(arg) for arg in split_top_level(match.group(3)) if arg.strip()
                ],
                syntax="triple_chevron",
            )
        )
    return launches


def iter_function_calls(text, function_name):
    pattern = re.compile(rf"\b{re.escape(function_name)}\s*\(")
    for match in pattern.finditer(text):
        opening = match.end() - 1
        closing = find_matching_delimiter(text, opening, "(", ")")
        if closing is not None:
            yield match.start(), text[opening + 1 : closing]


def find_matching_delimiter(text, opening, open_char, close_char):
    depth = 0
    for index in range(opening, len(text)):
        if text[index] == open_char:
            depth += 1
        elif text[index] == close_char:
            depth -= 1
            if depth == 0:
                return index
    return None
