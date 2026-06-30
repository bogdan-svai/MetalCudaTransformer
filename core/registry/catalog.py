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
        TranslationRule(
            source_symbol="__syncwarp",
            category="warp_intrinsic",
            supported_arities=(0, 1),
            capability="warp_sync",
            lowering_handler="lower_warp_intrinsic",
            target_symbol="simdgroup_barrier(mem_flags::mem_none)",
        ),
        TranslationRule(
            source_symbol="__activemask",
            category="warp_intrinsic",
            supported_arities=(0,),
            capability="warp_active_mask",
            lowering_handler="lower_warp_intrinsic",
            target_symbol="0xffffffffu",
        ),
        TranslationRule(
            source_symbol="__shfl_sync",
            category="warp_intrinsic",
            supported_arities=(3, 4),
            capability="warp_shuffle",
            lowering_handler="lower_warp_intrinsic",
            target_symbol="mcu_warp_shfl_sync",
        ),
        TranslationRule(
            source_symbol="__shfl_down_sync",
            category="warp_intrinsic",
            supported_arities=(3, 4),
            capability="warp_shuffle_down",
            lowering_handler="lower_warp_intrinsic",
            target_symbol="mcu_warp_shfl_down_sync",
        ),
        TranslationRule(
            source_symbol="__any_sync",
            category="warp_intrinsic",
            supported_arities=(2,),
            capability="warp_vote_any",
            lowering_handler="lower_warp_intrinsic",
            target_symbol="mcu_warp_any_sync",
        ),
        TranslationRule(
            source_symbol="__all_sync",
            category="warp_intrinsic",
            supported_arities=(2,),
            capability="warp_vote_all",
            lowering_handler="lower_warp_intrinsic",
            target_symbol="mcu_warp_all_sync",
        ),
    ]
    atomic_targets = {
        "atomicAdd": "atomic_fetch_add_explicit",
        "atomicCAS": "atomic_compare_exchange_weak_explicit",
        "atomicMin": "atomic_fetch_min_explicit",
        "atomicMax": "atomic_fetch_max_explicit",
        "atomicExch": "atomic_exchange_explicit",
    }
    rules.extend(
        TranslationRule(
            source_symbol=source,
            category="device_atomic",
            supported_arities=((3,) if source == "atomicCAS" else (2,)),
            capability="typed_integer_atomics",
            lowering_handler="lower_typed_atomic",
            target_symbol=target,
        )
        for source, target in atomic_targets.items()
    )
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
