import math
import re

from core.frontend.regex_parser import split_top_level, normalize_expr
from core.ir.models import Diagnostic


APPLE_THREADGROUP_LIMIT = 1024
UNSUPPORTED_CUDA_LIBRARIES = {
    "cublas.h",
    "cublas_v2.h",
    "cufft.h",
    "curand.h",
    "cusolverDn.h",
    "cusparse.h",
}


def parse_dimension(expr):
    text = str(expr).strip()
    result = {"expr": text, "dims": [], "total": None, "rank": None}
    if not text:
        return result

    dim3_match = re.match(r"dim3\s*\((.*)\)$", text)
    if dim3_match:
        dims = []
        for part in split_top_level(dim3_match.group(1)):
            part = normalize_expr(part)
            dims.append(int(part) if re.fullmatch(r"\d+", part) else part)
        while len(dims) < 3:
            dims.append(1)
        dims = dims[:3]
        result["dims"] = dims
        result["rank"] = 3 if dims[2] != 1 else 2 if dims[1] != 1 else 1
        if all(isinstance(dim, int) for dim in dims):
            result["total"] = math.prod(dims)
        return result

    if re.fullmatch(r"\d+", text):
        value = int(text)
        result["dims"] = [value, 1, 1]
        result["total"] = value
        result["rank"] = 1
        return result

    result["dims"] = [text]
    result["rank"] = 1
    return result


def find_divergent_barrier(function):
    body = function.body
    if "__syncthreads" not in body:
        return False

    pointer_parameters = {
        match.group(1)
        for parameter in getattr(function, "parameters", [])
        for match in [re.search(r"\*\s*([A-Za-z_]\w*)\s*(?:\[[^]]*\])?$", parameter)]
        if match
    }
    tainted = _thread_dependent_variables(body, pointer_parameters)
    for _kind, condition, start, end in _control_regions(body):
        if not _thread_dependent(condition, tainted, pointer_parameters):
            continue
        region = body[start:end]
        if "__syncthreads" in region:
            return True
        if re.search(r"\b(?:return|break|continue)\b", region) and "__syncthreads" in body[end:]:
            return True
    return False


def _thread_dependent_variables(body, pointer_parameters):
    tainted = set()
    assignments = list(re.finditer(
        r"\b(?:auto|bool|char|short|int|long|size_t|u?int(?:32|64)?_t|float|double|unsigned(?:\s+int)?)\s+"
        r"([A-Za-z_]\w*)\s*=\s*([^;]+)",
        body,
    ))
    changed = True
    while changed:
        changed = False
        for match in assignments:
            name, expression = match.group(1), match.group(2)
            if name in tainted:
                continue
            if _thread_dependent(expression, tainted, pointer_parameters):
                tainted.add(name)
                changed = True
    return tainted


def _thread_dependent(expression, tainted, pointer_parameters):
    if re.search(r"\b(?:threadIdx|blockIdx)\b", expression):
        return True
    if any(re.search(rf"\b{re.escape(name)}\b", expression) for name in tainted):
        return True
    return any(
        re.search(rf"(?:\*\s*{re.escape(name)}\b|\b{re.escape(name)}\s*\[)", expression)
        for name in pointer_parameters
    )


def _control_regions(body):
    pattern = re.compile(r"\b(?:else\s+)?(if|switch|while|for)\s*\(")
    for match in pattern.finditer(body):
        kind = match.group(1)
        opening = body.find("(", match.start())
        closing = _find_matching_delimiter(body, opening, "(", ")")
        if closing is None:
            continue
        condition = body[opening + 1:closing]
        if kind == "for":
            parts = split_top_level(condition, delimiter=";")
            condition = parts[1] if len(parts) == 3 else condition
        statement = closing + 1
        while statement < len(body) and body[statement].isspace():
            statement += 1
        if statement < len(body) and body[statement] == "{":
            end = _find_matching_brace(body, statement)
            yield kind, condition, statement + 1, end
        else:
            end = body.find(";", statement)
            yield kind, condition, statement, len(body) if end < 0 else end + 1


def _find_matching_delimiter(text, opening, open_char, close_char):
    depth = 0
    for index in range(opening, len(text)):
        if text[index] == open_char:
            depth += 1
        elif text[index] == close_char:
            depth -= 1
            if depth == 0:
                return index
    return None


def _find_matching_brace(text, open_index):
    depth = 0
    for index in range(open_index, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return index
    return len(text)


def validate(metadata):
    diagnostics = []
    kernels_by_name = {
        item["name"]: item for item in metadata.get("kernels", [])
    }

    for include in metadata.get("includes", []):
        if include["name"] in UNSUPPORTED_CUDA_LIBRARIES:
            diagnostics.append(Diagnostic(
                code="UnsupportedExternalLibrary",
                severity="error",
                message=f"CUDA library '{include['name']}' has no Stage 4 runtime shim mapping.",
                file=include["file"],
                line=include["line"],
            ))

    for launch in metadata.get("launches", []):
        block = parse_dimension(launch["block_expr"])
        grid = parse_dimension(launch["grid_expr"])
        launch["block"] = block
        launch["grid"] = grid
        total = block.get("total")
        kernel_body = kernels_by_name.get(launch["kernel"], {}).get("body", "")
        has_dynamic_shared = bool(re.search(r"\bextern\s+__shared__\b", kernel_body))
        shared_expr = str(launch.get("shared_expr", "0")).strip()
        has_launch_shared = shared_expr not in {"", "0", "0u", "0U"}
        if has_dynamic_shared and not has_launch_shared:
            diagnostics.append(Diagnostic(
                code="DynamicSharedSizeMissing",
                severity="error",
                message=f"Kernel '{launch['kernel']}' declares dynamic shared memory but launch size is zero.",
                file=launch["file"],
                line=launch["line"],
            ))
        if has_launch_shared and not has_dynamic_shared:
            diagnostics.append(Diagnostic(
                code="InvalidSharedDeclaration",
                severity="error",
                message=f"Kernel '{launch['kernel']}' receives dynamic shared bytes without a declaration.",
                file=launch["file"],
                line=launch["line"],
            ))
        if total is not None and total > APPLE_THREADGROUP_LIMIT:
            diagnostics.append(Diagnostic(
                code="HardwareLimitExceeded",
                severity="error",
                message=(
                    f"Kernel '{launch['kernel']}' launches {total} threads per block; "
                    f"Apple Silicon threadgroup limit is {APPLE_THREADGROUP_LIMIT}."
                ),
                file=launch["file"],
                line=launch["line"],
            ))

    for function in metadata.get("functions", []):
        if function.get("kind") not in {"KERNEL", "DEVICE_FUNC", "HOST_DEVICE_FUNC"}:
            continue
        pseudo = type("Function", (), function)
        if find_divergent_barrier(pseudo):
            diagnostics.append(Diagnostic(
                code="DivergentThreadgroupBarrier",
                severity="error",
                message=f"Kernel '{function['name']}' contains a barrier with divergent thread participation.",
                file=function["file"],
                line=function["line"],
            ))

    return [diagnostic.to_dict() for diagnostic in diagnostics]
