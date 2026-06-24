from core.ir.models import Diagnostic


SUPPORTED_FACTORS = (2, 4, 8)
THREADGROUP_LIMIT = 1024


def plan_coarsening(metadata, requested_factor=None):
    launches_by_kernel = {}
    for launch in metadata.get("launches", []):
        launches_by_kernel.setdefault(launch["kernel"], []).append(launch)

    plans = {}
    diagnostics = []
    recovered_hardware_limit = False

    for kernel in metadata.get("kernels", []):
        name = kernel["name"]
        launches = launches_by_kernel.get(name, [])
        static_blocks = [_static_block(launch.get("block", {})) for launch in launches]
        known_blocks = [block for block in static_blocks if block is not None]
        unique_blocks = set(known_blocks)
        needs_optimization = bool(requested_factor and requested_factor > 1) or any(
            _volume(block) > THREADGROUP_LIMIT for block in known_blocks
        )
        synchronized = "__shared__" in kernel.get("body", "") or "__syncthreads" in kernel.get("body", "")
        if synchronized and needs_optimization:
            diagnostics.append(_diagnostic(
                "CoarseningSharedKernelUnsupported",
                f"Kernel '{name}' uses shared memory or barriers and cannot be coarsened safely.",
                launches[0] if launches else kernel,
            ))
            continue

        if len(unique_blocks) > 1 and needs_optimization:
            diagnostics.append(_diagnostic(
                "IncompatibleCoarseningPlan",
                f"Kernel '{name}' is launched with incompatible static block dimensions.",
                launches[0] if launches else kernel,
            ))
            continue

        original_block = known_blocks[0] if len(unique_blocks) == 1 else None
        factor = requested_factor or 1
        mode = "manual" if requested_factor and requested_factor > 1 else "none"

        if requested_factor is None and original_block and _volume(original_block) > THREADGROUP_LIMIT:
            factor = _automatic_factor(original_block)
            mode = "auto"
            if factor is None:
                diagnostics.append(_diagnostic(
                    "UnsupportedCoarseningGeometry",
                    (
                        f"Kernel '{name}' block {list(original_block)} cannot be reduced below "
                        f"{THREADGROUP_LIMIT} threads with supported X-axis factors."
                    ),
                    launches[0],
                ))
                continue

        if requested_factor and requested_factor > 1:
            if original_block is None:
                diagnostics.append(_diagnostic(
                    "UnresolvedCoarseningGeometry",
                    f"Kernel '{name}' requires a static launch block for manual coarsening.",
                    kernel,
                ))
                continue
            if original_block[0] % requested_factor != 0:
                diagnostics.append(_diagnostic(
                    "NonDivisibleCoarseningFactor",
                    (
                        f"Kernel '{name}' block.x={original_block[0]} is not divisible by "
                        f"coarsening factor {requested_factor}."
                    ),
                    launches[0],
                ))
                continue

        physical_block = _physical_block(original_block, factor)
        if physical_block and _volume(physical_block) > THREADGROUP_LIMIT:
            diagnostics.append(_diagnostic(
                "PhysicalBlockLimitExceeded",
                (
                    f"Kernel '{name}' physical block {list(physical_block)} still exceeds "
                    f"{THREADGROUP_LIMIT} threads."
                ),
                launches[0],
            ))
            continue

        if original_block and _volume(original_block) > THREADGROUP_LIMIT and factor > 1:
            recovered_hardware_limit = True

        plans[name] = {
            "mode": mode,
            "axis": "x",
            "coarsening_factor": factor,
            "original_block": list(original_block) if original_block else None,
            "physical_block": list(physical_block) if physical_block else None,
        }

    return {
        "plans": plans,
        "diagnostics": [item.to_dict() for item in diagnostics],
        "recovered_hardware_limit": recovered_hardware_limit and not diagnostics,
    }


def _static_block(block):
    dims = block.get("dims", [])
    if len(dims) != 3 or not all(isinstance(value, int) for value in dims):
        return None
    return tuple(dims)


def _automatic_factor(block):
    for factor in SUPPORTED_FACTORS:
        if block[0] % factor == 0 and _volume(block) // factor <= THREADGROUP_LIMIT:
            return factor
    return None


def _physical_block(block, factor):
    if block is None:
        return None
    if block[0] % factor != 0:
        return None
    return (block[0] // factor, block[1], block[2])


def _volume(block):
    return block[0] * block[1] * block[2]


def _diagnostic(code, message, source):
    return Diagnostic(
        code=code,
        severity="error",
        message=message,
        file=source.get("file"),
        line=source.get("line"),
    )
