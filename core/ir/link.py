from dataclasses import dataclass, field

from core.ir.symbols import FunctionVariantIR


@dataclass
class DeviceLinkPlan:
    kernel_ids: list[str]
    ordered_variants: list[FunctionVariantIR]
    call_targets: dict[tuple, str]
    intrinsics_used: dict[str, dict]
    diagnostics: list[dict] = field(default_factory=list)
    atomic_parameters: dict[str, dict[str, dict]] = field(default_factory=dict)
    atomic_locals: dict[str, dict[str, dict]] = field(default_factory=dict)
    intrinsic_calls: dict[tuple, dict] = field(default_factory=dict)

    @property
    def ok(self):
        return not any(item["severity"] == "error" for item in self.diagnostics)

    def to_manifest(self, project):
        reachable_ids = set(self.kernel_ids) | {
            variant.symbol_id for variant in self.ordered_variants
        }
        variants_by_symbol = {}
        for variant in self.ordered_variants:
            variants_by_symbol.setdefault(variant.symbol_id, []).append(variant.to_dict())
        capabilities = {item["capability"] for item in self.intrinsics_used.values()}
        if self.ordered_variants:
            capabilities.add("device_functions")
        if self.atomic_parameters or self.atomic_locals:
            capabilities.add("typed_atomics")
        if project.types:
            capabilities.add("project_types")
        if project.macros:
            capabilities.add("device_macros")
        if any(
            any(local.array_extent is not None for local in project.functions[item].local_objects)
            for item in reachable_ids
        ):
            capabilities.add("fixed_local_arrays")
        if any(
            any(space in {"device", "thread"} for space in variant.parameter_address_spaces)
            for variant in self.ordered_variants
        ):
            capabilities.add("address_space_specialization")
        return {
            "schema_version": 1,
            "frontend": project.frontend,
            "translation_units": sorted(project.translation_units),
            "symbols": {
                symbol_id: {
                    **project.functions[symbol_id].to_dict(),
                    "generated_variants": variants_by_symbol.get(symbol_id, []),
                }
                for symbol_id in sorted(reachable_ids)
            },
            "intrinsics_used": self.intrinsics_used,
            "atomic_parameters": self.atomic_parameters,
            "atomic_locals": self.atomic_locals,
            "capabilities": sorted(capabilities),
            "diagnostics": self.diagnostics,
        }
