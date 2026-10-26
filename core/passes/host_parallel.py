"""Conservative independence proof for fixed-size local unsigned array maps."""


def unwrap(node):
    while node and node.kind in {"UNEXPOSED_EXPR", "PAREN_EXPR"} and len(node.children) == 1:
        node = node.children[0]
    return node


def prove_loop(loop, declarations):
    reasons = [
        r
        for r in loop.reasons
        if r
        not in {"DependenceProofMissing", "BufferExtentProofMissing", "ProfitabilityNotMeasured"}
    ]
    accesses = []
    result = {
        "eligible": False,
        "conditional": False,
        "runtime_guards": [],
        "reasons": reasons,
        "iterations": None,
        "accesses": accesses,
        "alias_policy": "known automatic origins; same mapping or disjoint regions for writes",
    }

    def reject(reason):
        if reason not in reasons:
            reasons.append(reason)

    if reasons or not all((loop.init, loop.condition, loop.update, loop.body)):
        reject("LoopStructureUnsupported")
        return result
    init = loop.init
    if init.kind != "DECL_STMT" or len(init.children) != 1:
        reject("IteratorInitializationUnsupported")
        return result
    iterator = init.children[0]
    if (
        iterator.kind != "VAR_DECL"
        or iterator.type != "unsigned int"
        or len(iterator.children) != 1
    ):
        reject("IteratorTypeUnsupported")
        return result
    zero = unwrap(iterator.children[0])
    condition, update = unwrap(loop.condition), unwrap(loop.update)
    if zero.integer_value != 0 or condition.operator != "<" or len(condition.children) != 2:
        reject("LoopBoundsUnsupported")
        return result
    left, bound = map(unwrap, condition.children)
    count = bound.integer_value
    dynamic = None
    if left.kind != "DECL_REF_EXPR" or left.symbol_usr != iterator.symbol_usr:
        reject("LoopBoundsUnsupported")
        return result
    if count is None:
        declaration = declarations.get(bound.symbol_usr)
        if (
            bound.kind != "DECL_REF_EXPR"
            or not declaration
            or declaration.kind != "PARM_DECL"
            or declaration.type != "unsigned int"
        ):
            reject("LoopBoundsUnsupported")
            return result
        dynamic = bound.symbol_usr
    elif not 0 <= count <= 4294967295:
        reject("LoopBoundsUnsupported")
        return result
    if (
        update.operator != "++"
        or len(update.children) != 1
        or unwrap(update.children[0]).symbol_usr != iterator.symbol_usr
    ):
        reject("IteratorUpdateUnsupported")
        return result
    result["iterations"] = count

    origins = {}
    resolving = set()

    def origin(symbol):
        if symbol in origins:
            return origins[symbol]
        if symbol in resolving:
            return None
        declaration = declarations.get(symbol)
        if not declaration or declaration.storage not in {"NONE", "AUTO"}:
            return None
        if (
            declaration.array_extent is not None
            and declaration.type == f"unsigned int[{declaration.array_extent}]"
        ):
            value = (symbol, 0, declaration.array_extent, declaration.initialized)
            origins[symbol] = value
            return value
        # An immutable pointer cannot be reassigned after initialization.
        if (
            declaration.type not in {"unsigned int *const", "const unsigned int *const"}
            or len(declaration.children) != 1
        ):
            return None
        resolving.add(symbol)
        initializer = unwrap(declaration.children[0])
        offset = 0
        if (
            initializer.kind == "BINARY_OPERATOR"
            and initializer.operator == "+"
            and len(initializer.children) == 2
        ):
            initializer, shift = map(unwrap, initializer.children)
            offset = shift.integer_value
        value = None
        if initializer.kind == "DECL_REF_EXPR" and offset is not None and offset >= 0:
            parent = origin(initializer.symbol_usr)
            if parent and parent[1] + offset <= parent[2]:
                value = (parent[0], parent[1] + offset, parent[2], parent[3])
        resolving.remove(symbol)
        origins[symbol] = value
        return value

    def guard_maximum(maximum):
        guard = {
            "kind": "unsigned_le_constant",
            "symbol_usr": dynamic,
            "maximum": min(maximum, 4294967295),
        }
        if not result["runtime_guards"]:
            result["runtime_guards"].append(guard)
        else:
            result["runtime_guards"][0]["maximum"] = min(
                result["runtime_guards"][0]["maximum"], guard["maximum"]
            )

    def access(node, mode):
        if len(node.children) != 2:
            reject("ArrayAccessUnsupported")
            return
        base, index = map(unwrap, node.children)
        if (
            base.kind != "DECL_REF_EXPR"
            or index.kind != "DECL_REF_EXPR"
            or index.symbol_usr != iterator.symbol_usr
        ):
            reject("CrossIterationAccessUnsupported")
            return
        provenance = origin(base.symbol_usr)
        if not provenance:
            reject("BufferExtentOrAliasProofMissing")
            return
        object_usr, offset, extent, initialized = provenance
        available = extent - offset
        if dynamic:
            guard_maximum(available)
        elif available < count:
            reject("BufferBoundsExceeded")
        if mode == "read" and not initialized:
            reject("BufferInitializationProofMissing")
        accesses.append(
            {
                "symbol_usr": base.symbol_usr,
                "name": base.spelling,
                "mode": mode,
                "extent": available,
                "origin_usr": object_usr,
                "origin_name": declarations[object_usr].spelling,
                "origin_extent": extent,
                "offset_elements": offset,
                "index": "iterator",
            }
        )

    def expression(node):
        node = unwrap(node)
        if node.kind == "ARRAY_SUBSCRIPT_EXPR":
            access(node, "read")
        elif (
            node.kind == "INTEGER_LITERAL"
            and node.integer_value is not None
            and 0 <= node.integer_value <= 4294967295
        ):
            pass
        elif node.kind == "DECL_REF_EXPR" and node.symbol_usr == iterator.symbol_usr:
            pass
        elif (
            node.kind == "BINARY_OPERATOR"
            and node.operator in {"+", "-", "*", "&", "|", "^"}
            and node.type == "unsigned int"
            and len(node.children) == 2
        ):
            for child in node.children:
                expression(child)
        else:
            reject("ExpressionOrEffectUnsupported")

    def statement(node):
        node = unwrap(node)
        if node.kind == "COMPOUND_STMT":
            for child in node.children:
                statement(child)
        elif node.kind == "BINARY_OPERATOR" and node.operator == "=" and len(node.children) == 2:
            target = unwrap(node.children[0])
            if target.kind != "ARRAY_SUBSCRIPT_EXPR":
                reject("ScalarWriteOrEffectUnsupported")
            else:
                access(target, "write")
            expression(node.children[1])
        else:
            reject("StatementOrEffectUnsupported")

    statement(loop.body)
    # Compare regions on the same known object. Same mapping is per-iteration;
    # shifted overlapping regions can carry values between iterations.
    for index, first in enumerate(accesses):
        for second in accesses[index + 1 :]:
            if (
                first["origin_usr"] != second["origin_usr"]
                or first["offset_elements"] == second["offset_elements"]
                or first["mode"] == second["mode"] == "read"
            ):
                continue
            distance = abs(first["offset_elements"] - second["offset_elements"])
            if dynamic:
                guard_maximum(distance)
            elif count > distance:
                reject("CrossIterationAliasOverlap")
    if not accesses:
        reject("NoSupportedArrayAccess")
    result["eligible"] = not reasons and not dynamic
    result["conditional"] = not reasons and bool(dynamic)
    if result["conditional"]:
        reject("RuntimeGuardRequired")
    return result
