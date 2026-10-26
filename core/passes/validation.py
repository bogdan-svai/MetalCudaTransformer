import math
import re

from core.frontend.regex_parser import split_top_level, normalize_expr
from core.ir.models import Diagnostic
from core.registry.library_adapters import default_library_registry


APPLE_THREADGROUP_LIMIT = 1024
LIBRARY_REGISTRY = default_library_registry()
UNSUPPORTED_CUDA_LIBRARIES = {"cublas.h", *LIBRARY_REGISTRY.unsupported_libraries()}
SUPPORTED_WARP_INTRINSICS = {
    "__syncwarp",
    "__activemask",
    "__shfl_sync",
    "__shfl_down_sync",
    "__any_sync",
    "__all_sync",
}
UNSUPPORTED_WARP_INTRINSICS = {
    "__shfl_up_sync",
    "__shfl_xor_sync",
    "__ballot_sync",
}
WARP_INTRINSICS = SUPPORTED_WARP_INTRINSICS | UNSUPPORTED_WARP_INTRINSICS


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
    statements = getattr(function, "statements", None)
    if statements:
        return _find_divergent_barrier_ir(statements)
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


def _find_divergent_barrier_ir(statements):
    """Check typed control-flow facts without rescanning source text."""
    for index, statement in enumerate(statements):
        attributes = statement.attributes or {}
        if not attributes.get("thread_dependent"):
            continue
        if attributes.get("contains_barrier"):
            return True
        if attributes.get("has_early_exit") and _cfg_reaches_barrier(statements, index):
            return True
    return False


def _cfg_reaches_barrier(statements, start):
    pending = list((statements[start].attributes or {}).get("cfg_successors", ()))
    visited = set()
    while pending:
        index = pending.pop()
        if index in visited or index < 0 or index >= len(statements):
            continue
        visited.add(index)
        attributes = statements[index].attributes or {}
        if attributes.get("contains_barrier"):
            return True
        pending.extend(attributes.get("cfg_successors", ()))
    return False


def _thread_dependent_variables(body, pointer_parameters):
    tainted = set()
    assignments = list(
        re.finditer(
            r"\b(?:auto|bool|char|short|int|long|size_t|u?int(?:32|64)?_t|float|double|unsigned(?:\s+int)?)\s+"
            r"([A-Za-z_]\w*)\s*=\s*([^;]+)",
            body,
        )
    )
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
        condition = body[opening + 1 : closing]
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


def validate(metadata, *, validate_warp=True, validate_barriers=True, validate_shared=True):
    diagnostics = []
    kernels_by_name = {item["name"]: item for item in metadata.get("kernels", [])}

    for include in metadata.get("includes", []):
        if include["name"] in UNSUPPORTED_CUDA_LIBRARIES:
            diagnostics.append(
                Diagnostic(
                    code="UnsupportedExternalLibrary",
                    severity="error",
                    message=f"CUDA library '{include['name']}' has no supported adapter.",
                    file=include["file"],
                    line=include["line"],
                )
            )

    for launch in metadata.get("launches", []):
        block = parse_dimension(launch["block_expr"])
        grid = parse_dimension(launch["grid_expr"])
        launch["block"] = block
        launch["grid"] = grid
        total = block.get("total")
        kernel_body = kernels_by_name.get(launch["kernel"], {}).get("body", "")
        has_dynamic_shared = validate_shared and bool(
            re.search(r"\bextern\s+__shared__\b", kernel_body)
        )
        shared_expr = str(launch.get("shared_expr", "0")).strip()
        has_launch_shared = shared_expr not in {"", "0", "0u", "0U"}
        if has_dynamic_shared and not has_launch_shared:
            diagnostics.append(
                Diagnostic(
                    code="DynamicSharedSizeMissing",
                    severity="error",
                    message=f"Kernel '{launch['kernel']}' declares dynamic shared memory but launch size is zero.",
                    file=launch["file"],
                    line=launch["line"],
                )
            )
        if validate_shared and has_launch_shared and not has_dynamic_shared:
            diagnostics.append(
                Diagnostic(
                    code="InvalidSharedDeclaration",
                    severity="error",
                    message=f"Kernel '{launch['kernel']}' receives dynamic shared bytes without a declaration.",
                    file=launch["file"],
                    line=launch["line"],
                )
            )
        if total is not None and total > APPLE_THREADGROUP_LIMIT:
            diagnostics.append(
                Diagnostic(
                    code="HardwareLimitExceeded",
                    severity="error",
                    message=(
                        f"Kernel '{launch['kernel']}' launches {total} threads per block; "
                        f"Apple Silicon threadgroup limit is {APPLE_THREADGROUP_LIMIT}."
                    ),
                    file=launch["file"],
                    line=launch["line"],
                )
            )

    for function in metadata.get("functions", []):
        if function.get("kind") not in {"KERNEL", "DEVICE_FUNC", "HOST_DEVICE_FUNC"}:
            continue
        pseudo = type("Function", (), function)
        if validate_barriers and find_divergent_barrier(pseudo):
            diagnostics.append(
                Diagnostic(
                    code="DivergentThreadgroupBarrier",
                    severity="error",
                    message=f"Kernel '{function['name']}' contains a barrier with divergent thread participation.",
                    file=function["file"],
                    line=function["line"],
                )
            )
        if validate_warp:
            diagnostics.extend(_warp_diagnostics(function))

    return [diagnostic.to_dict() for diagnostic in diagnostics]


def validate_semantic_project(project, metadata=None, *, validate_warp=False):
    """Validate device control-flow after semantic IR has been collected."""
    diagnostics = []
    launches = {}
    for launch in (metadata or {}).get("launches", []):
        launches.setdefault(launch["kernel"], []).append(launch)
    for function in project.functions.values():
        if not function.is_device_callable:
            continue
        if function.kind == "KERNEL" and metadata is not None:
            dynamic_shared = any(
                item.storage == "threadgroup" and (item.incomplete_array or item.variable_length)
                for item in function.local_objects
            )
            for launch in launches.get(function.name, []):
                shared_expr = str(launch.get("shared_expr", "0")).strip()
                has_launch_shared = shared_expr not in {"", "0", "0u", "0U"}
                if dynamic_shared and not has_launch_shared:
                    diagnostics.append(
                        Diagnostic(
                            code="DynamicSharedSizeMissing",
                            severity="error",
                            message=f"Kernel '{function.name}' declares dynamic shared memory but launch size is zero.",
                            file=launch.get("file"),
                            line=launch.get("line"),
                        )
                    )
                if has_launch_shared and not dynamic_shared:
                    diagnostics.append(
                        Diagnostic(
                            code="InvalidSharedDeclaration",
                            severity="error",
                            message=f"Kernel '{function.name}' receives dynamic shared bytes without a declaration.",
                            file=launch.get("file"),
                            line=launch.get("line"),
                        )
                    )
        if find_divergent_barrier(function):
            diagnostics.append(
                Diagnostic(
                    code="DivergentThreadgroupBarrier",
                    severity="error",
                    message=(
                        f"Kernel '{function.name}' contains a barrier with divergent thread participation."
                    ),
                    file=function.source.file,
                    line=function.source.line,
                )
            )
        if validate_warp:
            diagnostics.extend(_warp_diagnostics(function.to_dict()))
    return [diagnostic.to_dict() for diagnostic in diagnostics]


def _warp_diagnostics(function, calls=None):
    diagnostics = []
    body = function.get("body", "")
    for intrinsic in sorted(UNSUPPORTED_WARP_INTRINSICS) if calls is None else ():
        pattern = re.compile(rf"\b{re.escape(intrinsic)}\s*\(")
        for _match in pattern.finditer(body):
            diagnostics.append(
                Diagnostic(
                    code="UnsupportedWarpIntrinsic",
                    severity="error",
                    message=(
                        f"CUDA warp intrinsic '{intrinsic}' is not implemented in this build."
                    ),
                    file=function.get("file"),
                    line=function.get("body_line") or function.get("line"),
                )
            )
    for name, arguments in _warp_calls(body) if calls is None else calls:
        if name not in SUPPORTED_WARP_INTRINSICS:
            continue
        if name == "__activemask":
            if arguments:
                diagnostics.append(_warp_signature_diagnostic(function, name))
            continue
        if name == "__syncwarp":
            if len(arguments) > 1:
                diagnostics.append(_warp_signature_diagnostic(function, name))
            elif arguments and not _is_full_warp_mask(arguments[0]):
                diagnostics.append(_warp_mask_diagnostic(function, name, arguments[0]))
            continue
        if name in {"__shfl_sync", "__shfl_down_sync"}:
            if len(arguments) not in {3, 4}:
                diagnostics.append(_warp_signature_diagnostic(function, name))
                continue
            if not _is_full_warp_mask(arguments[0]):
                diagnostics.append(_warp_mask_diagnostic(function, name, arguments[0]))
            if len(arguments) == 4 and normalize_expr(arguments[3]) != "32":
                diagnostics.append(
                    Diagnostic(
                        code="UnsupportedWarpWidth",
                        severity="error",
                        message=f"CUDA warp intrinsic '{name}' currently supports width 32 only.",
                        file=function.get("file"),
                        line=function.get("body_line") or function.get("line"),
                    )
                )
            continue
        if name in {"__any_sync", "__all_sync"}:
            if len(arguments) != 2:
                diagnostics.append(_warp_signature_diagnostic(function, name))
                continue
            if not _is_full_warp_mask(arguments[0]):
                diagnostics.append(_warp_mask_diagnostic(function, name, arguments[0]))
    return diagnostics


def _warp_calls(body):
    pattern = re.compile(
        r"\b(__syncwarp|__activemask|__shfl_sync|__shfl_down_sync|__shfl_up_sync|__shfl_xor_sync|__ballot_sync|__any_sync|__all_sync)\s*\("
    )
    for match in pattern.finditer(body):
        opening = match.end() - 1
        closing = _find_matching_delimiter(body, opening, "(", ")")
        if closing is None:
            continue
        arguments = [
            item.strip() for item in split_top_level(body[opening + 1 : closing]) if item.strip()
        ]
        yield match.group(1), arguments


def _is_full_warp_mask(expression):
    normalized = re.sub(r"\s+", "", expression).lower().replace("ul", "u")
    return normalized in {"0xffffffff", "0xffffffffu", "__activemask()"}


def _warp_signature_diagnostic(function, name):
    return Diagnostic(
        code="UnsupportedWarpIntrinsic",
        severity="error",
        message=f"CUDA warp intrinsic '{name}' was called with an unsupported signature.",
        file=function.get("file"),
        line=function.get("body_line") or function.get("line"),
    )


def _warp_mask_diagnostic(function, name, mask):
    return Diagnostic(
        code="UnsupportedWarpMask",
        severity="error",
        message=f"CUDA warp intrinsic '{name}' currently supports full mask 0xffffffff only, not '{mask}'.",
        file=function.get("file"),
        line=function.get("body_line") or function.get("line"),
    )
