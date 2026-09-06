import hashlib
import os
import re
import time
from dataclasses import dataclass, field

from core.frontend.regex_parser import find_matching_brace, find_matching_delimiter, split_top_level


@dataclass
class PassReport:
    name: str
    changed: bool
    replacements: int = 0
    details: dict = field(default_factory=dict)

    def to_dict(self):
        payload = {
            "name": self.name,
            "changed": self.changed,
            "replacements": self.replacements,
        }
        if self.details:
            payload["details"] = self.details
        return payload


@dataclass
class MetalOptimizationResult:
    source: str
    reports: list[PassReport]
    analysis: dict

    def to_dict(self):
        return {
            "passes": [report.to_dict() for report in self.reports],
            "analysis": self.analysis,
        }


def optimize_metal_source(source, ir_analysis=None):
    reports = []
    started = time.perf_counter()
    pass_started = started
    source, report = rewrite_rotate_macros(source)
    _record_pass_timing(report, pass_started)
    reports.append(report)
    pass_started = time.perf_counter()
    if os.environ.get("MCU_METAL_CONSTANT_PROMOTION", "1") != "0":
        source, report = promote_literal_constant_arrays(source)
    else:
        report = PassReport(
            name="promote_literal_constant_arrays",
            changed=False,
            details={"enabled": False},
        )
    _record_pass_timing(report, pass_started)
    reports.append(report)
    pass_started = time.perf_counter()
    unroll_limit = _bounded_environment_integer(
        "MCU_METAL_UNROLL_MAX", default=8, minimum=0, maximum=128
    )
    approved_loops = None
    if ir_analysis is not None:
        approved_loops = sum(
            1
            for item in ir_analysis.get("static_loops", {}).get("candidates", [])
            if item.get("eligible")
        )
    source, report = unroll_small_static_loops(
        source,
        max_iterations=unroll_limit,
        approved_loops=approved_loops,
    )
    _record_pass_timing(report, pass_started)
    reports.append(report)
    pass_started = time.perf_counter()
    if os.environ.get("MCU_METAL_SCALARIZE", "1") != "0":
        approved_arrays = None
        if ir_analysis is not None:
            approved_arrays = sum(
                1
                for item in ir_analysis.get("scalarizable_arrays", {}).get("candidates", [])
                if item.get("eligible")
            )
        source, report = scalarize_small_constant_index_arrays(
            source, approved_arrays=approved_arrays
        )
    else:
        report = PassReport(
            name="scalarize_small_constant_index_arrays", changed=False, details={"enabled": False}
        )
    _record_pass_timing(report, pass_started)
    reports.append(report)
    pass_started = time.perf_counter()
    if os.environ.get("MCU_METAL_RECURRENCE_FUSION", "0") != "0":
        approved_fusions = None
        if ir_analysis is not None:
            approved_fusions = sum(
                1
                for item in ir_analysis.get("recurrent_arrays", {}).get("candidates", [])
                if item.get("eligible")
            )
        source, report = fuse_recurrent_array_consumers(source, approved_fusions=approved_fusions)
    else:
        report = PassReport(
            name="fuse_recurrent_array_consumers",
            changed=False,
            details={"enabled": False},
        )
    _record_pass_timing(report, pass_started)
    reports.append(report)
    analysis_started = time.perf_counter()
    analysis = {
        "local_memory": estimate_local_memory_pressure(source),
        "dynamic_array_candidates": analyze_dynamic_array_candidates(source),
        "register_pressure": analyze_register_pressure(source),
        "predicate_materialization": analyze_predicate_materialization(source),
    }
    analysis["timing_ms"] = round((time.perf_counter() - analysis_started) * 1000, 3)
    analysis["optimization_total_ms"] = round((time.perf_counter() - started) * 1000, 3)
    return MetalOptimizationResult(source=source, reports=reports, analysis=analysis)


def _record_pass_timing(report, started):
    report.details.setdefault("timing_ms", round((time.perf_counter() - started) * 1000, 3))


def _bounded_environment_integer(name, default, minimum, maximum):
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return min(maximum, max(minimum, value))


def _scan_metal_functions(source):
    """Find function bodies by scanning identifiers and balanced braces."""
    functions = []
    cursor = 0
    while cursor < len(source):
        position = _find_identifier(source, "inline", cursor)
        kernel_position = _find_identifier(source, "kernel", cursor)
        positions = [item for item in (position, kernel_position) if item is not None]
        if not positions:
            break
        start = min(positions)
        opening = source.find("{", start)
        if opening < 0:
            break
        closing = find_matching_brace(source, opening)
        if closing is None or closing <= opening:
            break
        functions.append({"start": start, "opening": opening, "closing": closing})
        cursor = closing + 1
    return functions


def _function_name(source, start, opening):
    paren = source.find("(", start, opening)
    while paren >= 0:
        close = find_matching_delimiter(source, paren, "(", ")")
        if close is not None and close < opening:
            break
        paren = source.find("(", paren + 1, opening)
    if paren < 0:
        return "<anonymous>"
    end = paren
    while end and source[end - 1].isspace():
        end -= 1
    begin = end
    while begin and (source[begin - 1].isalnum() or source[begin - 1] == "_"):
        begin -= 1
    return source[begin:end] or "<anonymous>"


def promote_literal_constant_arrays(source, minimum_bytes=64):
    replacements = []
    global_tables = {}
    promoted_arrays = 0
    bytes_saved = 0
    for function in _scan_metal_functions(source):
        opening = function["opening"]
        closing = function["closing"]
        if closing is None or closing <= opening:
            continue
        body = source[opening + 1 : closing]
        rewritten = body
        changed = False
        declarations = [
            item for item in _scan_array_declarations(body) if item["values"] is not None
        ]
        for declaration in reversed(declarations):
            type_name = declaration["type"]
            size = declaration["size"]
            array_bytes = size * _type_size(type_name)
            values = _split_initializer_values(declaration["values"])
            if not _is_known_local_type(type_name) or array_bytes < minimum_bytes:
                continue
            if len(values) != size or not all(_is_numeric_literal(value) for value in values):
                continue
            name = declaration["name"]
            remainder = body[: declaration["start"]] + body[declaration["end"] :]
            if not _array_uses_are_readonly_indexing(remainder, name):
                continue
            normalized_values = ",".join(values)
            identity = f"{type_name}|{size}|{normalized_values}"
            digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
            global_name = f"mcu_const_{digest}"
            global_tables.setdefault(global_name, (type_name, size, values))
            segment = rewritten[declaration["end"] :]
            segment = re.sub(rf"\b{re.escape(name)}\b", global_name, segment)
            rewritten = rewritten[: declaration["start"]] + segment
            promoted_arrays += 1
            bytes_saved += array_bytes
            changed = True
        if changed:
            replacements.append((opening + 1, closing, rewritten))
    for start, end, replacement in reversed(replacements):
        source = source[:start] + replacement + source[end:]
    if global_tables:
        declarations = []
        for name, (type_name, size, values) in sorted(global_tables.items()):
            rendered_values = ", ".join(values)
            declarations.append(f"constant {type_name} {name}[{size}] = {{{rendered_values}}};")
        first_function = re.search(r"(?m)^(?:template\s*<|inline\s+|kernel\s+)", source)
        insertion = first_function.start() if first_function else len(source)
        global_source = "\n".join(declarations) + "\n\n"
        source = source[:insertion] + global_source + source[insertion:]
    return source, PassReport(
        name="promote_literal_constant_arrays",
        changed=promoted_arrays > 0,
        replacements=promoted_arrays,
        details={
            "enabled": True,
            "minimum_bytes": minimum_bytes,
            "unique_tables": len(global_tables),
            "local_bytes_removed": bytes_saved,
        },
    )


def _is_numeric_literal(value):
    text = value.strip()
    if not text:
        return False
    if text[0] in "+-":
        text = text[1:]
    suffix = ""
    while text and text[-1] in "uUlLfF":
        suffix = text[-1] + suffix
        text = text[:-1]
    if not text or len(suffix) > 3:
        return False
    if text.startswith(("0x", "0X")):
        digits = text[2:]
        return bool(digits) and all(char in "0123456789abcdefABCDEF" for char in digits)
    exponent = text.find("e")
    if exponent < 0:
        exponent = text.find("E")
    if exponent >= 0:
        mantissa, exponent_text = text[:exponent], text[exponent + 1 :]
        if exponent_text.startswith(("+", "-")):
            exponent_text = exponent_text[1:]
        if not exponent_text.isdigit():
            return False
        text = mantissa
    if text.count(".") > 1 or not text:
        return False
    digits = text.replace(".", "")
    return bool(digits) and digits.isdigit()


def _array_uses_are_readonly_indexing(body, name):
    indexed = _find_indexed_accesses(body, name)
    if not indexed:
        return False
    without_indexed = body
    for start, end, _index in reversed(indexed):
        without_indexed = without_indexed[:start] + without_indexed[end:]
    return _find_identifier(without_indexed, name, 0) is None


def rewrite_rotate_macros(source):
    replacements = 0
    source, count = re.subn(
        r"#define ROTLEFT\(a,b\) \(\(\(a\) << \(b\)\) \| \(\(a\) >> \(32-\(b\)\)\)\)",
        "#define ROTLEFT(a,b) rotate((a), uint(b))",
        source,
    )
    replacements += count
    source, count = re.subn(
        r"#define ROTRIGHT\(a,b\) \(\(\(a\) >> \(b\)\) \| \(\(a\) << \(32-\(b\)\)\)\)",
        "#define ROTRIGHT(a,b) rotate((a), uint(32 - (b)))",
        source,
    )
    replacements += count
    return source, PassReport(
        name="rewrite_rotate_macros",
        changed=replacements > 0,
        replacements=replacements,
    )


def unroll_small_static_loops(source, max_iterations=8, approved_loops=None):
    cursor = 0
    pieces = []
    replacements = 0
    unrolled_iterations = 0
    while True:
        position = _find_identifier(source, "for", cursor)
        if position is None:
            pieces.append(source[cursor:])
            break
        opening = position + 3
        while opening < len(source) and source[opening].isspace():
            opening += 1
        if opening >= len(source) or source[opening] != "(":
            pieces.append(source[cursor : position + 3])
            cursor = position + 3
            continue
        header_end = find_matching_delimiter(source, opening, "(", ")")
        if header_end is None:
            pieces.append(source[cursor:])
            break
        specification = _parse_static_for_header(source[opening + 1 : header_end])
        if specification is None:
            pieces.append(source[cursor : header_end + 1])
            cursor = header_end + 1
            continue
        body_opening = header_end + 1
        while body_opening < len(source) and source[body_opening].isspace():
            body_opening += 1
        if body_opening >= len(source) or source[body_opening] != "{":
            body_end = source.find(";", body_opening)
            if body_end < 0:
                pieces.append(source[cursor : header_end + 1])
                cursor = header_end + 1
                continue
            body = source[body_opening : body_end + 1]
            replacement, iterations = _render_unrolled_loop(specification, body, max_iterations)
            if replacement is not None and (approved_loops is None or approved_loops > 0):
                pieces.append(source[cursor:position])
                pieces.append(replacement)
                replacements += 1
                unrolled_iterations += iterations
                if approved_loops is not None:
                    approved_loops -= 1
            else:
                pieces.append(source[cursor : body_end + 1])
            cursor = body_end + 1
            continue
        opening = body_opening
        closing = find_matching_brace(source, opening)
        if closing is None or closing <= opening:
            pieces.append(source[cursor : match.end()])
            cursor = match.end()
            continue
        body = source[opening + 1 : closing]
        replacement, iterations = _render_unrolled_loop(specification, body, max_iterations)
        if replacement is not None and approved_loops is not None and approved_loops <= 0:
            replacement, iterations = None, 0
        if replacement is None:
            pieces.append(source[cursor : closing + 1])
        else:
            pieces.append(source[cursor:position])
            pieces.append(replacement)
            replacements += 1
            unrolled_iterations += iterations
            if approved_loops is not None:
                approved_loops -= 1
        cursor = closing + 1
    return "".join(pieces), PassReport(
        name="unroll_small_static_loops",
        changed=replacements > 0,
        replacements=replacements,
        details={
            "max_iterations": max_iterations,
            "unrolled_iterations": unrolled_iterations,
        },
    )


def _render_unrolled_loop(specification, body, max_iterations):
    if re.search(r"\b(break|continue|return)\b", body):
        return None, 0
    variable, start, op, stop = specification
    end = stop + 1 if op == "<=" else stop
    iterations = end - start
    if iterations <= 0 or iterations > max_iterations:
        return None, 0
    if variable.startswith("mcu_"):
        return None, 0
    unrolled = []
    for value in range(start, end):
        rewritten = re.sub(rf"\b{re.escape(variable)}\b", str(value), body)
        unrolled.append("{\n" + rewritten.strip("\n") + "\n}")
    return "\n".join(unrolled), iterations


def _find_identifier(source, identifier, start):
    cursor = start
    while True:
        position = source.find(identifier, cursor)
        if position < 0:
            return None
        before = source[position - 1] if position else " "
        after = (
            source[position + len(identifier)] if position + len(identifier) < len(source) else " "
        )
        if not (before.isalnum() or before == "_") and not (after.isalnum() or after == "_"):
            return position
        cursor = position + len(identifier)


def _parse_static_for_header(header):
    clauses = []
    current = []
    depth = 0
    for char in header:
        if char in "([{":
            depth += 1
        elif char in ")]}" and depth:
            depth -= 1
        if char == ";" and depth == 0:
            clauses.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    clauses.append("".join(current).strip())
    if len(clauses) != 3:
        return None
    init, condition, increment = (item.strip() for item in clauses)
    if "=" not in init:
        return None
    declaration, start_text = (item.strip() for item in init.split("=", 1))
    variable = declaration.split()[-1] if declaration.split() else ""
    if not variable.isidentifier() or not start_text.isdigit():
        return None
    op = "<" if "<" in condition and "<=" not in condition else "<=" if "<=" in condition else None
    if op is None:
        return None
    left, stop_text = (item.strip() for item in condition.split(op, 1))
    if left != variable or not stop_text.isdigit():
        return None
    if increment not in {variable + "++", "++" + variable}:
        return None
    return variable, int(start_text), op, int(stop_text)


def scalarize_small_constant_index_arrays(source, max_elements=16, approved_arrays=None):
    replacements = []
    arrays_replaced = 0
    scalar_variables = 0
    for function in _scan_metal_functions(source):
        opening = function["opening"]
        closing = function["closing"]
        if closing is None or closing <= opening:
            continue
        body = source[opening + 1 : closing]
        rewritten, count, variables = _scalarize_arrays_in_body(body, max_elements)
        if approved_arrays is not None and count > approved_arrays:
            rewritten, count, variables = body, 0, 0
        if count:
            replacements.append((opening + 1, closing, rewritten))
            arrays_replaced += count
            scalar_variables += variables
            if approved_arrays is not None:
                approved_arrays -= count
    for start, end, replacement in reversed(replacements):
        source = source[:start] + replacement + source[end:]
    return source, PassReport(
        name="scalarize_small_constant_index_arrays",
        changed=arrays_replaced > 0,
        replacements=arrays_replaced,
        details={
            "max_elements": max_elements,
            "scalar_variables": scalar_variables,
        },
    )


def fuse_recurrent_array_consumers(source, approved_fusions=None):
    replacements = []
    fused = 0
    bytes_saved = 0
    skipped_high_pressure = 0
    for function in _scan_metal_functions(source):
        opening = function["opening"]
        closing = function["closing"]
        if closing is None or closing <= opening:
            continue
        body = source[opening + 1 : closing]
        rewritten, details, skipped = _fuse_recurrences_in_body(body)
        if approved_fusions is not None and len(details) > approved_fusions:
            details, skipped, rewritten = [], 0, body
        if details:
            replacements.append((opening + 1, closing, rewritten))
            fused += len(details)
            if approved_fusions is not None:
                approved_fusions -= len(details)
            bytes_saved += sum(item["bytes_saved"] for item in details)
            skipped_high_pressure += skipped
    for start, end, replacement in reversed(replacements):
        source = source[:start] + replacement + source[end:]
    return source, PassReport(
        name="fuse_recurrent_array_consumers",
        changed=fused > 0,
        replacements=fused,
        details=(
            {"bytes_saved": bytes_saved, "skipped_high_pressure": skipped_high_pressure}
            if fused or skipped_high_pressure
            else {}
        ),
    )


def _fuse_recurrences_in_body(body):
    details = []
    skipped_high_pressure = 0
    while True:
        fused = _find_recurrent_consumer_fusion(body)
        if fused is None:
            break
        if fused == "high_pressure":
            skipped_high_pressure += 1
            break
        start, end, replacement, array_name, original_elements, window, type_name = fused
        body = body[:start] + replacement + body[end:]
        body = _shrink_first_array_declaration(body, array_name, original_elements, window)
        details.append(
            {
                "array": array_name,
                "original_elements": original_elements,
                "sliding_window_elements": window,
                "bytes_saved": max(0, original_elements - window) * _type_size(type_name),
            }
        )
    return body, details, skipped_high_pressure


def _find_recurrent_consumer_fusion(body):
    arrays = {array["name"]: array for array in _local_arrays(body)}
    for candidate in _braced_recurrence_candidates(body):
        fused = _try_recurrent_consumer_fusion(body, arrays, candidate)
        if fused is not None:
            return fused
    for recurrence in _unbraced_recurrence_candidates(body):
        candidate = {
            "start": recurrence["start"],
            "end": recurrence["end"],
            "index_var": recurrence["index_var"],
            "stop": recurrence["stop"],
            "array_name": recurrence["array_name"],
            "recurrence_body": recurrence["statement"],
            "rhs": recurrence["rhs"],
            "recurrence_start": None,
        }
        fused = _try_recurrent_consumer_fusion(body, arrays, candidate)
        if fused is not None:
            return fused
    return None


def _unbraced_recurrence_candidates(body):
    """Yield single-statement recurrence loops using structural scanning."""
    cursor = 0
    while True:
        keyword = _find_identifier(body, "for", cursor)
        if keyword is None:
            return
        opening = keyword + len("for")
        while opening < len(body) and body[opening].isspace():
            opening += 1
        if opening >= len(body) or body[opening] != "(":
            cursor = keyword + len("for")
            continue
        closing = find_matching_delimiter(body, opening, "(", ")")
        if closing is None:
            return
        clauses = _parse_unbraced_for_header(body[opening + 1 : closing])
        if clauses is None:
            cursor = closing + 1
            continue
        index_var, _start, op, stop = clauses
        if op != "<":
            cursor = closing + 1
            continue
        statement_start = closing + 1
        while statement_start < len(body) and body[statement_start].isspace():
            statement_start += 1
        if statement_start >= len(body) or body[statement_start] == "{":
            cursor = closing + 1
            continue
        semicolon = body.find(";", statement_start)
        if semicolon < 0:
            return
        statement = body[statement_start : semicolon + 1]
        assignments = _find_recurrence_assignments(statement, index_var)
        if len(assignments) == 1:
            array_name, rhs = assignments[0]
            yield {
                "start": keyword,
                "end": semicolon + 1,
                "index_var": index_var,
                "stop": stop,
                "array_name": array_name,
                "statement": statement,
                "rhs": rhs,
            }
        cursor = semicolon + 1


def _parse_unbraced_for_header(header):
    clauses = _parse_static_for_header(header)
    if clauses is not None:
        return clauses
    parts = [part.strip() for part in header.split(";")]
    if len(parts) != 3 or parts[0]:
        return None
    condition, increment = parts[1], parts[2]
    if "<=" in condition:
        return None
    marker = condition.find("<")
    if marker < 0:
        return None
    variable, stop_text = condition[:marker].strip(), condition[marker + 1 :].strip()
    if not variable.isidentifier() or not stop_text.isdigit():
        return None
    if increment not in {variable + "++", "++" + variable}:
        return None
    return variable, 0, "<", int(stop_text)


def _braced_recurrence_candidates(body):
    cursor = 0
    while True:
        keyword = _find_identifier(body, "for", cursor)
        if keyword is None:
            return
        opening_paren = keyword + len("for")
        while opening_paren < len(body) and body[opening_paren].isspace():
            opening_paren += 1
        if opening_paren >= len(body) or body[opening_paren] != "(":
            cursor = keyword + len("for")
            continue
        closing_paren = find_matching_delimiter(body, opening_paren, "(", ")")
        if closing_paren is None:
            return
        parsed = _parse_static_for_header(body[opening_paren + 1 : closing_paren])
        if parsed is None:
            cursor = closing_paren + 1
            continue
        index_var, start, op, stop = parsed
        if op != "<":
            cursor = closing_paren + 1
            continue
        loop_opening = closing_paren + 1
        while loop_opening < len(body) and body[loop_opening].isspace():
            loop_opening += 1
        if loop_opening >= len(body) or body[loop_opening] != "{":
            cursor = closing_paren + 1
            continue
        closing = find_matching_brace(body, loop_opening)
        if closing is None:
            return
        loop_body = body[loop_opening + 1 : closing]
        if any(
            _find_identifier(loop_body, token, 0) is not None
            for token in ("break", "continue", "return")
        ):
            cursor = closing + 1
            continue
        assignments = _find_recurrence_assignments(loop_body, index_var)
        if len(assignments) != 1:
            cursor = closing + 1
            continue
        array_name, rhs = assignments[0]
        yield {
            "start": keyword,
            "end": closing + 1,
            "index_var": index_var,
            "stop": stop,
            "array_name": array_name,
            "recurrence_body": loop_body,
            "rhs": rhs,
            "recurrence_start": start,
        }
        cursor = closing + 1


def _find_recurrence_assignments(loop_body, index_var):
    assignments = []
    cursor = 0
    processed_names = set()
    while True:
        name_start = _next_identifier_start(loop_body, cursor)
        if name_start is None:
            break
        name_end = name_start
        while name_end < len(loop_body) and (
            loop_body[name_end].isalnum() or loop_body[name_end] == "_"
        ):
            name_end += 1
        name = loop_body[name_start:name_end]
        if name in processed_names:
            cursor = name_end
            continue
        processed_names.add(name)
        accesses = _find_indexed_accesses(loop_body[name_start:], name)
        for access_start, access_end, raw_index in accesses:
            if "".join(raw_index.split()) != index_var:
                continue
            absolute_end = name_start + access_end
            equals = absolute_end
            while equals < len(loop_body) and loop_body[equals].isspace():
                equals += 1
            if (
                equals >= len(loop_body)
                or loop_body[equals] != "="
                or (equals + 1 < len(loop_body) and loop_body[equals + 1] == "=")
            ):
                continue
            semicolon = loop_body.find(";", equals + 1)
            if semicolon < 0:
                continue
            assignments.append((name, loop_body[equals + 1 : semicolon].strip()))
        cursor = name_end
    return assignments


def _next_identifier_start(source, start):
    cursor = start
    while cursor < len(source):
        if (source[cursor].isalpha() or source[cursor] == "_") and (
            cursor == 0 or not (source[cursor - 1].isalnum() or source[cursor - 1] == "_")
        ):
            return cursor
        cursor += 1
    return None


def _try_recurrent_consumer_fusion(body, arrays, candidate):
    array_name = candidate["array_name"]
    array = arrays.get(array_name)
    if not array:
        return None
    index_var = candidate["index_var"]
    stop = candidate["stop"]
    distances = _recurrence_distances(candidate["rhs"], array_name, index_var)
    if not distances:
        return None
    window = max(distances)
    recurrence_start = candidate["recurrence_start"]
    if recurrence_start is None:
        recurrence_start = window
    if window >= array["elements"] or recurrence_start < window or stop > array["elements"]:
        return None
    # Circular indexing is costly when the source array is already large:
    # keep the safe transform available, but require an explicit override
    # for high-pressure cases where spilling dominates the memory saving.
    if array["bytes"] >= 256 and os.environ.get("MCU_METAL_ALLOW_HIGH_PRESSURE_FUSION", "0") != "1":
        return "high_pressure"
    if not _recurrence_array_accesses_are_safe(candidate["recurrence_body"], array_name, index_var):
        return None
    consumer = _next_consumer_loop(body, candidate["end"], index_var, stop, array_name)
    if consumer is None:
        return None
    consumer_start, consumer_prefix, consumer_opening, consumer_closing = consumer
    between = body[candidate["end"] : consumer_start]
    fused_loop = _render_fused_consumer_loop(
        consumer_prefix,
        body[consumer_opening + 1 : consumer_closing],
        candidate["recurrence_body"],
        array_name,
        index_var,
        window,
        recurrence_start,
    )
    return (
        candidate["start"],
        consumer_closing + 1,
        between + fused_loop,
        array_name,
        array["elements"],
        window,
        array["type"],
    )


def _recurrence_array_accesses_are_safe(recurrence_body, array_name, index_var):
    accesses = _find_indexed_accesses(recurrence_body, array_name)
    if not accesses:
        return False
    for _start, _end, raw_index in accesses:
        index = "".join(raw_index.split())
        if index == index_var:
            continue
        if index.startswith(index_var + "-") and index[len(index_var) + 1 :].isdigit():
            continue
        return False
    return True


def _recurrence_distances(rhs, array_name, index_var):
    distances = []
    for _start, _end, raw_index in _find_indexed_accesses(rhs, array_name):
        index = "".join(raw_index.split())
        prefix = index_var + "-"
        if index.startswith(prefix) and index[len(prefix) :].isdigit():
            distances.append(int(index[len(prefix) :]))
    return distances


def _next_consumer_loop(body, offset, index_var, stop, array_name):
    cursor = offset
    while True:
        keyword = _find_identifier(body, "for", cursor)
        if keyword is None:
            return None
        opening_paren = keyword + len("for")
        while opening_paren < len(body) and body[opening_paren].isspace():
            opening_paren += 1
        if opening_paren >= len(body) or body[opening_paren] != "(":
            cursor = keyword + len("for")
            continue
        closing_paren = find_matching_delimiter(body, opening_paren, "(", ")")
        if closing_paren is None:
            return None
        clauses = _parse_static_for_header(body[opening_paren + 1 : closing_paren])
        if (
            clauses is None
            or clauses[:2] != (index_var, 0)
            or clauses[2] != "<"
            or clauses[3] != stop
        ):
            cursor = closing_paren + 1
            continue
        opening = closing_paren + 1
        while opening < len(body) and body[opening].isspace():
            opening += 1
        if opening >= len(body) or body[opening] != "{":
            cursor = closing_paren + 1
            continue
        closing = find_matching_brace(body, opening)
        if closing is None:
            return None
        loop_body = body[opening + 1 : closing]
        accesses = _find_indexed_accesses(loop_body, array_name)
        if not any("".join(index.split()) == index_var for _s, _e, index in accesses):
            cursor = closing + 1
            continue
        between = body[offset:keyword]
        if _find_indexed_accesses(between, array_name):
            return None
        return keyword, body[keyword : opening + 1], opening, closing


def _render_fused_consumer_loop(
    prefix, consumer_body, recurrence_statement, array_name, index_var, window, recurrence_start
):
    recurrence_assignment = _rewrite_ring_indices(
        recurrence_statement, array_name, index_var, window
    )
    consumer_body = _rewrite_ring_indices(consumer_body, array_name, index_var, window)
    indent = prefix[: len(prefix) - len(prefix.lstrip(" \t"))]
    inner_indent = indent + "    "
    recurrence_assignment = _indent_block(recurrence_assignment.strip(), inner_indent + "    ")
    consumer_body = consumer_body.strip("\n")
    return (
        f"{prefix}\n"
        f"{inner_indent}if ({index_var} >= {recurrence_start}) {{\n"
        f"{recurrence_assignment}\n"
        f"{inner_indent}}}\n"
        f"{consumer_body}\n"
        f"{indent}}}"
    )


def _rewrite_ring_indices(source, array_name, index_var, window):
    """Rewrite safe recurrence accesses using the structural access scanner."""
    rewritten = source
    for start, end, raw_index in reversed(_find_indexed_accesses(source, array_name)):
        index = "".join(raw_index.split())
        if index == index_var:
            expression = index_var
        else:
            prefix = index_var + "-"
            if not index.startswith(prefix) or not index[len(prefix) :].isdigit():
                continue
            expression = f"({index_var} - {index[len(prefix):]})"
        replacement = f"{array_name}[{_ring_index(expression, window)}]"
        rewritten = rewritten[:start] + replacement + rewritten[end:]
    return rewritten


def _shrink_first_array_declaration(body, array_name, original_elements, window):
    for declaration in _scan_array_declarations(body):
        if declaration["name"] != array_name or declaration["size"] != original_elements:
            continue
        opening = body.find("[", declaration["start"], declaration["end"])
        closing = find_matching_delimiter(body, opening, "[", "]")
        if opening < 0 or closing is None:
            return body
        return body[: opening + 1] + str(window) + body[closing:]
    # Declarations can be grouped (``value, i, buffer[32]``), which the
    # single-declarator scanner intentionally leaves untouched. Locate that
    # declarator structurally without treating source text as a regex.
    cursor = 0
    while True:
        position = _find_identifier(body, array_name, cursor)
        if position is None:
            break
        opening = position + len(array_name)
        while opening < len(body) and body[opening].isspace():
            opening += 1
        if opening < len(body) and body[opening] == "[":
            closing = find_matching_delimiter(body, opening, "[", "]")
            if closing is not None and body[opening + 1 : closing].strip() == str(
                original_elements
            ):
                statement_start = max(body.rfind(";", 0, position), body.rfind("\n", 0, position))
                prefix = body[statement_start + 1 : position]
                if "=" not in prefix:
                    return body[: opening + 1] + str(window) + body[closing:]
        cursor = position + len(array_name)
    return body


def _ring_index(expression, window):
    if window > 0 and (window & (window - 1)) == 0:
        return f"{expression} & {window - 1}"
    return f"{expression} % {window}"


def _indent_block(text, indent):
    return "\n".join(f"{indent}{line}" if line.strip() else line for line in text.splitlines())


def _scalarize_arrays_in_body(body, max_elements):
    arrays_replaced = 0
    scalar_variables = 0
    while True:
        changed = False
        for declaration in _scan_array_declarations(body):
            type_name = declaration["type"]
            if not _is_known_local_type(type_name):
                continue
            name = declaration["name"]
            size = declaration["size"]
            if size <= 0 or size > max_elements:
                continue
            remainder = body[: declaration["start"]] + body[declaration["end"] :]
            if not _array_uses_are_constant_indexed(remainder, name, size):
                continue
            rendered = _render_scalarized_declaration(declaration, size)
            rewritten_remainder = _replace_constant_array_uses(remainder, name)
            body = (
                rewritten_remainder[: declaration["start"]]
                + rendered
                + rewritten_remainder[declaration["start"] :]
            )
            arrays_replaced += 1
            scalar_variables += size
            changed = True
            break
        if not changed:
            break
    return body, arrays_replaced, scalar_variables


def _scan_array_declarations(body):
    declarations = []
    cursor = 0
    while cursor < len(body):
        end = body.find(";", cursor)
        if end < 0:
            break
        array_marker = body.find("[", cursor, end)
        if array_marker < 0:
            cursor = end + 1
            continue
        statement_start = body.rfind("\n", 0, array_marker) + 1
        statement = body[statement_start : end + 1]
        opening = statement.find("[")
        closing = statement.find("]", opening + 1)
        if opening < 0 or closing < 0:
            cursor = end + 1
            continue
        size_text = statement[opening + 1 : closing].strip()
        if not size_text.isdigit():
            cursor = end + 1
            continue
        name_end = opening
        while name_end and statement[name_end - 1].isspace():
            name_end -= 1
        name_start = name_end
        while name_start and (
            statement[name_start - 1].isalnum() or statement[name_start - 1] == "_"
        ):
            name_start -= 1
        name = statement[name_start:name_end]
        prefix = statement[:name_start].strip()
        const = prefix.startswith("const ")
        if const:
            prefix = prefix[6:].strip()
        if not name or not prefix or not all(part.isidentifier() for part in prefix.split()):
            cursor = end + 1
            continue
        values = None
        equals = statement.find("=", closing + 1)
        if equals >= 0:
            brace_open = statement.find("{", equals + 1)
            brace_close = statement.rfind("}")
            if brace_open >= 0 and brace_close > brace_open:
                values = statement[brace_open + 1 : brace_close]
        declarations.append(
            {
                "start": statement_start,
                "end": end + 1,
                "indent": statement[: len(statement) - len(statement.lstrip())],
                "const": "const " if const else "",
                "type": prefix,
                "name": name,
                "size": int(size_text),
                "values": values,
            }
        )
        cursor = end + 1
    return declarations


def _array_uses_are_constant_indexed(body, name, size):
    indexed = _find_indexed_accesses(body, name)
    if not indexed:
        return False
    for _start, _end, index in indexed:
        index = index.strip()
        if not index.isdigit():
            return False
        if int(index) >= size:
            return False
    without_indexed = body
    for start, end, _index in reversed(indexed):
        without_indexed = without_indexed[:start] + without_indexed[end:]
    return _find_identifier(without_indexed, name, 0) is None


def _render_scalarized_declaration(match, size):
    indent = match["indent"]
    const = match["const"]
    type_name = match["type"]
    name = match["name"]
    values = _split_initializer_values(match["values"])
    lines = []
    for index in range(size):
        initializer = ""
        if index < len(values):
            initializer = f" = {values[index]}"
        lines.append(f"{indent}{const}{type_name} {name}_{index}{initializer};")
    return "\n".join(lines)


def _split_initializer_values(values):
    if values is None:
        return []
    return [item.strip() for item in values.split(",") if item.strip()]


def _replace_constant_array_uses(body, name):
    rewritten = body
    accesses = _find_indexed_accesses(body, name)
    for start, end, index in reversed(accesses):
        if index.strip().isdigit():
            rewritten = rewritten[:start] + f"{name}_{index.strip()}" + rewritten[end:]
    return rewritten


def _find_indexed_accesses(source, name):
    accesses = []
    cursor = 0
    while True:
        position = source.find(name, cursor)
        if position < 0:
            break
        before = source[position - 1] if position else " "
        after_name = position + len(name)
        after = source[after_name] if after_name < len(source) else " "
        if (before.isalnum() or before == "_") or (after.isalnum() or after == "_"):
            cursor = after_name
            continue
        opening = after_name
        while opening < len(source) and source[opening].isspace():
            opening += 1
        if opening >= len(source) or source[opening] != "[":
            cursor = after_name
            continue
        closing = find_matching_delimiter(source, opening, "[", "]")
        if closing is None:
            break
        accesses.append((position, closing + 1, source[opening + 1 : closing]))
        cursor = closing + 1
    return accesses


def estimate_local_memory_pressure(source):
    summaries = []
    for function in _scan_metal_functions(source):
        opening, closing = function["opening"], function["closing"]
        if closing is None or closing <= opening:
            continue
        name = _function_name(source, function["start"], opening)
        body = source[opening + 1 : closing]
        arrays = []
        total_bytes = 0
        for array in _local_arrays(body):
            total_bytes += array["bytes"]
            arrays.append(array)
        if arrays:
            summaries.append(
                {
                    "function": name,
                    "array_count": len(arrays),
                    "estimated_bytes_per_thread": total_bytes,
                    "arrays": arrays[:8],
                }
            )
    summaries.sort(key=lambda item: item["estimated_bytes_per_thread"], reverse=True)
    return {
        "functions": summaries[:16],
        "max_estimated_bytes_per_thread": (
            summaries[0]["estimated_bytes_per_thread"] if summaries else 0
        ),
    }


def analyze_dynamic_array_candidates(source):
    candidates = []
    for function in _scan_metal_functions(source):
        opening, closing = function["opening"], function["closing"]
        if closing is None or closing <= opening:
            continue
        function_name = _function_name(source, function["start"], opening)
        body = source[opening + 1 : closing]
        for array in _local_arrays(body):
            details = _dynamic_array_details(body, array["name"], array["elements"])
            if details:
                candidates.append(
                    {
                        "function": function_name,
                        "array": array["name"],
                        "elements": array["elements"],
                        "estimated_bytes": array["bytes"],
                        **details,
                    }
                )
    candidates.sort(
        key=lambda item: (item.get("sliding_window_bytes_saved", 0), item["estimated_bytes"]),
        reverse=True,
    )
    return {
        "candidates": candidates[:16],
    }


def analyze_register_pressure(source):
    """Conservative source-level pressure signal used by optimization cost models.

    This is deliberately an estimate, not a hardware register count. It combines
    local array storage with dynamic-index traffic, loop nesting, and helper-call
    count so risky transforms can be gated consistently across workloads.
    """
    summaries = []
    for function in _scan_metal_functions(source):
        opening, closing = function["opening"], function["closing"]
        if closing is None or closing <= opening:
            continue
        name = _function_name(source, function["start"], opening)
        body = source[opening + 1 : closing]
        arrays = list(_local_arrays(body))
        local_bytes = sum(item["bytes"] for item in arrays)
        dynamic_accesses = 0
        for array in arrays:
            uses = _find_indexed_accesses(body, array["name"])
            dynamic_accesses += sum(1 for _s, _e, index in uses if not index.strip().isdigit())
        loops = len(re.findall(r"\b(?:for|while)\s*\(", body))
        calls = len(re.findall(r"\b[A-Za-z_]\w*\s*\([^;{}]*\)", body))
        score = local_bytes + dynamic_accesses * 8 + loops * 16 + calls * 4
        summaries.append(
            {
                "function": name,
                "estimated_local_bytes": local_bytes,
                "dynamic_array_accesses": dynamic_accesses,
                "loop_count": loops,
                "call_count": calls,
                "pressure_score": score,
                "risk": "high" if score >= 512 else "medium" if score >= 256 else "low",
            }
        )
    summaries.sort(key=lambda item: item["pressure_score"], reverse=True)
    return {"functions": summaries[:32]}


def analyze_predicate_materialization(source):
    """Find local buffers materialized by a call and consumed only by a predicate.

    The report is intentionally analysis-only for now. A later IR pass can prove
    purity/alias safety and replace these buffers with a predicate-returning form.
    """
    candidates = []
    for function in _scan_metal_functions(source):
        opening, closing = function["opening"], function["closing"]
        if closing is None or closing <= opening:
            continue
        body = source[opening + 1 : closing]
        function_name = _function_name(source, function["start"], opening)
        arrays = {item["name"] for item in _local_arrays(body)}
        for array_name in arrays:
            producer = _find_array_producer_call(body, array_name)
            if producer is None:
                continue
            tail = body[producer["end"] :]
            predicate = _find_array_predicate(tail, array_name)
            if predicate is None:
                continue
            intervening = tail[: predicate["start"]]
            writes = [item for item in _find_indexed_accesses(intervening, array_name)]
            if writes:
                continue
            candidates.append(
                {
                    "function": function_name,
                    "buffer": array_name,
                    "producer": producer["name"],
                    "predicate_offset": predicate["start"],
                    "suggested_action": "prove_purity_then_fuse",
                }
            )
    return {"candidates": candidates[:32]}


def _find_array_producer_call(body, array_name):
    cursor = 0
    while True:
        name_start = _next_identifier_start(body, cursor)
        if name_start is None:
            return None
        name_end = name_start
        while name_end < len(body) and (body[name_end].isalnum() or body[name_end] == "_"):
            name_end += 1
        name = body[name_start:name_end]
        if name in {"if", "for", "while", "switch", "return"}:
            cursor = name_end
            continue
        opening = name_end
        while opening < len(body) and body[opening].isspace():
            opening += 1
        if opening >= len(body) or body[opening] != "(":
            cursor = name_end
            continue
        closing = find_matching_delimiter(body, opening, "(", ")")
        if closing is None:
            return None
        semicolon = closing + 1
        while semicolon < len(body) and body[semicolon].isspace():
            semicolon += 1
        if (
            semicolon < len(body)
            and body[semicolon] == ";"
            and _find_identifier(body[opening + 1 : closing], array_name, 0) is not None
        ):
            return {"name": name, "end": semicolon + 1}
        cursor = closing + 1


def _find_array_predicate(body, array_name):
    cursor = 0
    while True:
        keyword = _find_identifier(body, "if", cursor)
        if keyword is None:
            return None
        opening = keyword + 2
        while opening < len(body) and body[opening].isspace():
            opening += 1
        if opening < len(body) and body[opening] == "(":
            closing = find_matching_delimiter(body, opening, "(", ")")
            if (
                closing is not None
                and _find_identifier(body[opening + 1 : closing], array_name, 0) is not None
            ):
                return {"start": keyword}
        cursor = keyword + 2


def _dynamic_array_details(body, name, elements):
    uses = _find_indexed_accesses(body, name)
    if not uses:
        return None
    indices = []
    dynamic_indices = []
    for _start, _end, raw_index in uses:
        index = " ".join(raw_index.strip().split())
        indices.append(index)
        if not index.isdigit():
            dynamic_indices.append(index)
    if not dynamic_indices:
        return None
    recurrence = _recurrence_details(body, name)
    unique_indices = sorted(set(dynamic_indices))
    result = {
        "dynamic_indices": unique_indices[:12],
        "dynamic_use_count": len(dynamic_indices),
    }
    if recurrence:
        assignment_end = recurrence.pop("assignment_end")
        type_name = recurrence.pop("type")
        window = recurrence["max_distance"]
        result.update(
            {
                "recurrence": recurrence,
                "sliding_window_elements": window,
                "sliding_window_bytes_saved": max(0, elements - window) * _type_size(type_name),
                "requires_loop_fusion": _has_post_recurrence_dynamic_read(
                    body, name, assignment_end
                ),
            }
        )
    return result


def _recurrence_details(body, name):
    best = None
    for start, end, raw_index in _find_indexed_accesses(body, name):
        var = raw_index.strip()
        if not var.isidentifier():
            continue
        equals = end
        while equals < len(body) and body[equals].isspace():
            equals += 1
        if (
            equals >= len(body)
            or body[equals] != "="
            or (equals + 1 < len(body) and body[equals + 1] == "=")
        ):
            continue
        semicolon = body.find(";", equals + 1)
        if semicolon < 0:
            continue
        rhs = body[equals + 1 : semicolon].strip()
        distances = []
        for _rhs_start, _rhs_end, rhs_index in _find_indexed_accesses(rhs, name):
            normalized = "".join(rhs_index.split())
            prefix = var + "-"
            if normalized.startswith(prefix) and normalized[len(prefix) :].isdigit():
                distances.append(int(normalized[len(prefix) :]))
        if not distances:
            continue
        max_distance = max(distances)
        candidate = {
            "index_variable": var,
            "distances": sorted(set(distances)),
            "max_distance": max_distance,
            "assignment_end": semicolon + 1,
            "type": _array_type_for_name(body, name) or "uint",
        }
        if best is None or candidate["max_distance"] > best["max_distance"]:
            best = candidate
    if not best:
        return None
    return best


def _has_post_recurrence_dynamic_read(body, name, offset):
    tail = body[offset:]
    return any(not index.strip().isdigit() for _s, _e, index in _find_indexed_accesses(tail, name))


def _array_type_for_name(body, name):
    for array in _local_arrays(body):
        if array["name"] == name:
            return array["type"]
    return None


def _local_arrays(body):
    pattern = re.compile(
        r"(?m)^[ \t]*(?:const\s+|thread\s+|threadgroup\s+)*"
        r"(?P<type>(?:(?:unsigned|signed)\s+)?[A-Za-z_]\w*)\s+"
        r"(?P<declarators>[^;{}()]*\[[^;{}()]*\][^;{}()]*)[;=]"
    )
    for match in pattern.finditer(body):
        type_name = match.group("type")
        if not _is_known_local_type(type_name):
            continue
        for array in re.finditer(
            r"\b(?P<name>[A-Za-z_]\w*)\s*\[\s*(?P<size>\d+)\s*\]",
            match.group("declarators"),
        ):
            size = int(array.group("size"))
            yield {
                "name": array.group("name"),
                "type": type_name,
                "elements": size,
                "bytes": size * _type_size(type_name),
            }


def _type_size(type_name):
    normalized = re.sub(r"\s+", " ", type_name).strip()
    if normalized in {"char", "uchar", "BYTE", "unsigned char", "signed char"}:
        return 1
    if normalized in {"short", "ushort", "unsigned short", "signed short"}:
        return 2
    if normalized in {"long", "ulong", "size_t", "double", "uint64_t", "int64_t"}:
        return 8
    return 4


def _is_known_local_type(type_name):
    normalized = re.sub(r"\s+", " ", type_name).strip()
    return normalized in {
        "char",
        "uchar",
        "BYTE",
        "unsigned char",
        "signed char",
        "short",
        "ushort",
        "unsigned short",
        "signed short",
        "int",
        "uint",
        "WORD",
        "unsigned int",
        "signed int",
        "uint32_t",
        "int32_t",
        "long",
        "ulong",
        "size_t",
        "unsigned long",
        "signed long",
        "float",
        "double",
        "uint64_t",
        "int64_t",
    }
