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

    index_vars = set()
    for match in re.finditer(
        r"\b(?:int|unsigned|uint|size_t|long|auto)\s+([A-Za-z_]\w*)\s*=\s*[^;]*(?:threadIdx|blockIdx|blockDim|gridDim)[^;]*;",
        body,
    ):
        index_vars.add(match.group(1))

    condition_pattern = re.compile(r"\b(?:if|else\s+if|switch)\s*\(([^)]*)\)\s*\{", re.MULTILINE)
    for match in condition_pattern.finditer(body):
        condition = match.group(1)
        depends_on_thread = re.search(r"\b(threadIdx|blockIdx|blockDim|gridDim)\b", condition)
        depends_on_index_var = any(re.search(r"\b" + re.escape(name) + r"\b", condition) for name in index_vars)
        if not depends_on_thread and not depends_on_index_var:
            continue
        close = _find_matching_brace(body, match.end() - 1)
        if "__syncthreads" in body[match.end():close]:
            return True
    return False


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
        if function.get("kind") != "KERNEL":
            continue
        pseudo = type("Function", (), function)
        if find_divergent_barrier(pseudo):
            diagnostics.append(Diagnostic(
                code="DivergentBarrierWarning",
                severity="warning",
                message=f"Kernel '{function['name']}' contains __syncthreads() inside a thread-dependent branch.",
                file=function["file"],
                line=function["line"],
            ))

    return [diagnostic.to_dict() for diagnostic in diagnostics]
