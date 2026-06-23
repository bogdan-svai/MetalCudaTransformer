from core.registry.model import TranslationRule


class TranslationRegistry:
    def __init__(self, rules):
        self._rules = {rule.source_symbol: rule for rule in rules}

    def resolve(self, symbol, argument_count):
        rule = self._rules.get(symbol)
        if not rule:
            return None
        if rule.supported_arities and argument_count not in rule.supported_arities:
            return False
        return rule

    def to_dict(self):
        return {
            name: rule.to_dict()
            for name, rule in sorted(self._rules.items())
        }

    def token_replacements(self):
        replacements = {}
        for rule in self._rules.values():
            if rule.lowering_handler == "replace_token" and rule.target_symbol:
                replacements[rule.source_symbol] = rule.target_symbol
            elif rule.lowering_handler == "lower_threadgroup_barrier" and rule.target_symbol:
                replacements[rule.source_symbol + "()"] = rule.target_symbol
        return replacements


def default_registry():
    rules = [
        TranslationRule(
            source_symbol="memset",
            category="device_intrinsic",
            supported_arities=(3,),
            capability="device_byte_fill",
            lowering_handler="lower_device_memset",
            target_symbol="mcu_device_memset",
        ),
        TranslationRule(
            source_symbol="__syncthreads",
            category="device_builtin",
            supported_arities=(0,),
            capability="threadgroup_barrier",
            lowering_handler="lower_threadgroup_barrier",
            target_symbol="threadgroup_barrier(mem_flags::mem_threadgroup)",
        ),
    ]
    geometry = {
        "threadIdx.x": "thread_position_in_threadgroup.x",
        "threadIdx.y": "thread_position_in_threadgroup.y",
        "threadIdx.z": "thread_position_in_threadgroup.z",
        "blockIdx.x": "threadgroup_position_in_grid.x",
        "blockIdx.y": "threadgroup_position_in_grid.y",
        "blockIdx.z": "threadgroup_position_in_grid.z",
        "blockDim.x": "threads_per_threadgroup.x",
        "blockDim.y": "threads_per_threadgroup.y",
        "blockDim.z": "threads_per_threadgroup.z",
        "gridDim.x": "threadgroups_per_grid.x",
        "gridDim.y": "threadgroups_per_grid.y",
        "gridDim.z": "threadgroups_per_grid.z",
    }
    rules.extend(
        TranslationRule(
            source_symbol=source,
            category="geometry_builtin",
            supported_arities=(),
            capability="cuda_geometry",
            lowering_handler="replace_token",
            target_symbol=target,
        )
        for source, target in geometry.items()
    )
    return TranslationRegistry(rules)
