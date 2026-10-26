"""Typed optimization preconditions derived from the semantic statement IR."""

import re


def analyze_static_loops(project, max_iterations=128):
    candidates = []
    for function in project.functions.values():
        for statement in function.statements:
            if statement.kind != "FOR_STMT":
                continue
            parsed = _parse_for_ir(statement.attributes)
            if parsed is None:
                candidates.append(
                    {
                        "function": function.name,
                        "source": statement.source.to_dict(),
                        "eligible": False,
                        "reason": "non_static_header",
                    }
                )
                continue
            variable, start, operator, stop = parsed
            end = stop + 1 if operator == "<=" else stop
            iterations = end - start
            candidates.append(
                {
                    "function": function.name,
                    "source": statement.source.to_dict(),
                    "variable": variable,
                    "start": start,
                    "stop": stop,
                    "operator": operator,
                    "iterations": iterations,
                    "eligible": 0 < iterations <= max_iterations,
                    "reason": "" if 0 < iterations <= max_iterations else "iteration_limit",
                }
            )
    return {"max_iterations": max_iterations, "candidates": candidates}


def analyze_loop_invariant_inputs(project):
    """Find pointer parameters read inside loops without loop-body writes.

    The result is deliberately analysis-only: later passes can use these
    proven invariant regions for midstate/prefix specialization without
    relying on test-specific constants.
    """
    candidates = []
    for function in project.functions.values():
        parameters = {item.name for item in function.parameters}
        if not parameters:
            continue
        for statement in function.statements:
            if statement.kind != "FOR_STMT":
                continue
            accesses = statement.attributes.get("array_accesses", [])
            for parameter in sorted(parameters):
                relevant = [item for item in accesses if item.get("array") == parameter]
                if not relevant or any(item.get("mode") == "write" for item in relevant):
                    continue
                candidates.append(
                    {
                        "function": function.name,
                        "parameter": parameter,
                        "source": statement.source.to_dict(),
                        "reads": len(relevant),
                        "eligible": True,
                    }
                )
    return {"candidates": candidates}


def analyze_scalarizable_arrays(project, max_elements=16):
    candidates = []
    static_loop_count = sum(
        1 for item in analyze_static_loops(project)["candidates"] if item.get("eligible")
    )
    for function in project.functions.values():
        for local in function.local_objects:
            if not local.array_extent or local.array_extent > max_elements:
                continue
            accesses = [
                access.get("index", "")
                for statement in function.statements
                for access in statement.attributes.get("array_accesses", [])
                if access.get("array") == local.name
            ]
            if not accesses:
                continue
            # The declaration itself is represented as `name[extent]` in the
            # token stream; it is not an indexed runtime access.
            accesses = [index for index in accesses if index.strip() != str(local.array_extent)]
            dynamic = [index.strip() for index in accesses if not index.strip().isdigit()]
            out_of_range = any(
                index.strip().isdigit() and int(index.strip()) >= local.array_extent
                for index in accesses
            )
            if out_of_range or (dynamic and not static_loop_count):
                continue
            candidates.append(
                {
                    "function": function.name,
                    "array": local.name,
                    "elements": local.array_extent,
                    "source": local.source.to_dict(),
                    "eligible": True,
                }
            )
    return {"max_elements": max_elements, "candidates": candidates}


def analyze_recurrent_arrays(project):
    candidates = []
    for function in project.functions.values():
        loop_statements = [item for item in function.statements if item.kind == "FOR_STMT"]
        accesses = [
            access
            for item in loop_statements
            for access in item.attributes.get("array_accesses", [])
        ]
        for local in function.local_objects:
            if not local.array_extent:
                continue
            recurrence = [
                access
                for access in accesses
                if access.get("array") == local.name and "-" in access.get("index", "")
            ]
            if recurrence:
                writes = [
                    access
                    for access in accesses
                    if access.get("array") == local.name and access.get("mode") == "write"
                ]
                if not writes:
                    continue
                producer_index = next(
                    index
                    for index, item in enumerate(loop_statements)
                    if any(
                        access in item.attributes.get("array_accesses", []) for access in recurrence
                    )
                )
                consumer_exists = any(
                    any(
                        access.get("array") == local.name
                        and access.get("mode") == "read"
                        and "-" not in access.get("index", "")
                        for access in item.attributes.get("array_accesses", [])
                    )
                    for item in loop_statements[producer_index + 1 :]
                )
                if not consumer_exists:
                    continue
                distances = sorted(
                    {distance for access in recurrence for distance in _offsets(access["index"])}
                )
                if not distances:
                    continue
                element_bytes = _type_size(local.type.canonical)
                window_elements = max(distances)
                window_bytes = window_elements * element_bytes
                bytes_saved = max(0, local.array_extent - window_elements) * element_bytes
                worth_it = bytes_saved > 0 and window_bytes <= 256
                candidates.append(
                    {
                        "function": function.name,
                        "array": local.name,
                        "reason": "constant_distance_recurrence_candidate",
                        "accesses": recurrence,
                        "distances": distances,
                        "window_elements": window_elements,
                        "element_bytes": element_bytes,
                        "window_bytes": window_bytes,
                        "bytes_saved": bytes_saved,
                        "has_producer": True,
                        "producer_before_consumer": True,
                        "pressure": (
                            "low"
                            if window_bytes <= 128
                            else "medium" if window_bytes <= 256 else "high"
                        ),
                        "worth_it": worth_it,
                        "eligible": worth_it,
                    }
                )
    return {"candidates": candidates}


def analyze_global_memory_accesses(project):
    """Classify kernel pointer accesses for safe backend memory decisions."""
    reports = []
    for function in project.functions.values():
        if function.kind != "KERNEL":
            continue
        pointer_parameters = {
            parameter.name
            for parameter in function.parameters
            if parameter.type.pointer_depth > 0 and parameter.name
        }
        accesses = [
            access
            for statement in function.statements
            for access in statement.attributes.get("array_accesses", [])
            if access.get("array") in pointer_parameters
        ]
        for array in sorted(pointer_parameters):
            observed = [access for access in accesses if access.get("array") == array]
            if not observed:
                continue
            indices = sorted({str(access.get("index", "")).strip() for access in observed})
            classes = {_classify_global_index(index) for index in indices}
            reports.append(
                {
                    "function": function.name,
                    "array": array,
                    "access_count": len(observed),
                    "reads": sum(access.get("mode") == "read" for access in observed),
                    "writes": sum(access.get("mode") == "write" for access in observed),
                    "indices": indices[:8],
                    "patterns": sorted(classes),
                    "coalesced_candidate": classes == {"thread_linear"},
                }
            )
    return {"candidates": reports}


def _classify_global_index(index):
    normalized = "".join(index.split())
    if normalized in {"threadIdx.x", "blockIdx.x*blockDim.x+threadIdx.x"}:
        return "thread_linear"
    if "threadIdx.x" in normalized:
        return "thread_affine"
    if normalized.isdigit():
        return "uniform"
    return "dynamic"


def _type_size(type_name):
    normalized = type_name.replace("const", "").strip()
    if any(token in normalized for token in ("char", "uchar", "bool")):
        return 1
    if any(token in normalized for token in ("short", "half")):
        return 2
    if any(token in normalized for token in ("long", "double", "64")):
        return 8
    return 4


def _offsets(index):
    tokens = index.replace("(", " ").replace(")", " ").split()
    distances = []
    for position, token in enumerate(tokens[:-1]):
        if token == "-" and tokens[position + 1].isdigit():
            distances.append(int(tokens[position + 1]))
    return distances


def _indexed_accesses(source, name):
    result = []
    cursor = 0
    while True:
        position = source.find(name, cursor)
        if position < 0:
            return result
        after = position + len(name)
        before_char = source[position - 1] if position else " "
        after_char = source[after] if after < len(source) else " "
        if (before_char.isalnum() or before_char == "_") or (
            after_char.isalnum() or after_char == "_"
        ):
            cursor = after
            continue
        while after < len(source) and source[after].isspace():
            after += 1
        if after >= len(source) or source[after] != "[":
            cursor = position + len(name)
            continue
        closing = source.find("]", after + 1)
        if closing < 0:
            return result
        result.append(source[after + 1 : closing])
        cursor = closing + 1


def _parse_for_ir(attributes):
    """Parse a static ``for`` header from semantic loop fields.

    The frontend has already split the header at top-level semicolons, so this
    avoids rescanning the complete statement text.  Missing fields are kept
    compatible with cached/legacy IR by returning ``None``.
    """
    init = str(attributes.get("loop_init", "")).strip().rstrip(";").strip()
    condition = str(attributes.get("loop_condition", "")).strip().rstrip(";").strip()
    update = str(attributes.get("loop_update", "")).strip().rstrip(";").strip()
    if not init or not condition or not update:
        return None
    init_match = re.search(r"\b([A-Za-z_]\w*)\s*=\s*(-?\d+)\s*$", init)
    condition_match = re.fullmatch(r"([A-Za-z_]\w*)\s*(<=|<)\s*(-?\d+)", condition)
    update_match = re.fullmatch(r"(?:\+\+\s*([A-Za-z_]\w*)|([A-Za-z_]\w*)\s*\+\+)", update)
    if not init_match or not condition_match or not update_match:
        return None
    variable = init_match.group(1)
    if condition_match.group(1) != variable:
        return None
    increment_variable = update_match.group(1) or update_match.group(2)
    if increment_variable != variable:
        return None
    return (
        variable,
        int(init_match.group(2)),
        condition_match.group(2),
        int(condition_match.group(3)),
    )
