from dataclasses import dataclass, field
import hashlib
import json


DEVICE_KINDS = {"KERNEL", "DEVICE_FUNC", "HOST_DEVICE_FUNC"}


@dataclass(frozen=True)
class SourceLocationIR:
    file: str
    line: int | None = None
    column: int | None = None

    def to_dict(self):
        return {"file": self.file, "line": self.line, "column": self.column}


@dataclass(frozen=True)
class TypeIR:
    spelling: str
    canonical: str
    kind: str = "value"
    is_const: bool = False
    pointer_depth: int = 0

    def to_dict(self):
        return {
            "spelling": self.spelling,
            "canonical": self.canonical,
            "kind": self.kind,
            "is_const": self.is_const,
            "pointer_depth": self.pointer_depth,
        }


@dataclass(frozen=True)
class ParameterIR:
    name: str
    type: TypeIR

    def to_dict(self):
        return {"name": self.name, "type": self.type.to_dict()}


@dataclass(frozen=True)
class CallEdgeIR:
    caller_id: str
    callee_name: str
    callee_id: str | None
    arguments: tuple[str, ...]
    source: SourceLocationIR
    argument_types: tuple[TypeIR, ...] = tuple()
    argument_symbols: tuple[str | None, ...] = tuple()

    def to_dict(self):
        return {
            "caller_id": self.caller_id,
            "callee_name": self.callee_name,
            "callee_id": self.callee_id,
            "arguments": list(self.arguments),
            "argument_types": [item.to_dict() for item in self.argument_types],
            "argument_symbols": list(self.argument_symbols),
            "source": self.source.to_dict(),
        }


@dataclass(frozen=True)
class LocalObjectIR:
    name: str
    type: TypeIR
    storage: str
    array_extent: int | None
    variable_length: bool
    incomplete_array: bool
    source: SourceLocationIR

    def to_dict(self):
        return {
            "name": self.name,
            "type": self.type.to_dict(),
            "storage": self.storage,
            "array_extent": self.array_extent,
            "variable_length": self.variable_length,
            "incomplete_array": self.incomplete_array,
            "source": self.source.to_dict(),
        }


@dataclass
class FunctionSymbolIR:
    symbol_id: str
    name: str
    kind: str
    qualifiers: tuple[str, ...]
    return_type: TypeIR
    parameters: list[ParameterIR]
    source: SourceLocationIR
    body_line: int | None
    body: str
    translation_unit: str
    calls: list[CallEdgeIR] = field(default_factory=list)
    local_objects: list[LocalObjectIR] = field(default_factory=list)

    @property
    def is_device_callable(self):
        return self.kind in DEVICE_KINDS

    def to_dict(self):
        return {
            "symbol_id": self.symbol_id,
            "name": self.name,
            "kind": self.kind,
            "qualifiers": list(self.qualifiers),
            "return_type": self.return_type.to_dict(),
            "parameters": [parameter.to_dict() for parameter in self.parameters],
            "source": self.source.to_dict(),
            "body_line": self.body_line,
            "translation_unit": self.translation_unit,
            "calls": [call.to_dict() for call in self.calls],
            "local_objects": [item.to_dict() for item in self.local_objects],
        }


@dataclass(frozen=True)
class TypeDeclarationIR:
    name: str
    kind: str
    fields: tuple[ParameterIR, ...]
    underlying_type: str | None
    source: SourceLocationIR

    def to_dict(self):
        return {
            "name": self.name,
            "kind": self.kind,
            "fields": [field.to_dict() for field in self.fields],
            "underlying_type": self.underlying_type,
            "source": self.source.to_dict(),
        }


@dataclass(frozen=True)
class MacroIR:
    name: str
    parameters: tuple[str, ...]
    expression: str
    source: SourceLocationIR

    def to_dict(self):
        return {
            "name": self.name,
            "parameters": list(self.parameters),
            "expression": self.expression,
            "source": self.source.to_dict(),
        }


@dataclass(frozen=True)
class FunctionVariantIR:
    symbol_id: str
    generated_name: str
    parameter_address_spaces: tuple[str | None, ...]

    def key(self):
        return self.symbol_id, self.parameter_address_spaces

    def to_dict(self):
        return {
            "symbol_id": self.symbol_id,
            "generated_name": self.generated_name,
            "parameter_address_spaces": list(self.parameter_address_spaces),
        }


@dataclass
class ProjectSymbolIR:
    frontend: str
    translation_units: list[str]
    functions: dict[str, FunctionSymbolIR]
    types: dict[str, TypeDeclarationIR] = field(default_factory=dict)
    macros: dict[str, MacroIR] = field(default_factory=dict)
    diagnostics: list[dict] = field(default_factory=list)

    def functions_named(self, name):
        return [function for function in self.functions.values() if function.name == name]

    def to_dict(self):
        return {
            "frontend": self.frontend,
            "translation_units": self.translation_units,
            "functions": {
                symbol_id: function.to_dict()
                for symbol_id, function in sorted(self.functions.items())
            },
            "types": {
                name: declaration.to_dict()
                for name, declaration in sorted(self.types.items())
            },
            "macros": {
                name: macro.to_dict()
                for name, macro in sorted(self.macros.items())
            },
            "diagnostics": self.diagnostics,
        }


def canonical_symbol_id(name, parameter_types):
    signature = ",".join(normalize_type_name(item) for item in parameter_types)
    return f"{name}({signature})"


def generated_symbol_name(symbol_id, address_spaces=()):
    readable = symbol_id.split("(", 1)[0]
    context = symbol_id + "|" + ",".join(space or "value" for space in address_spaces)
    digest = hashlib.sha1(context.encode("utf-8")).hexdigest()[:10]
    return f"{readable}__mcu_{digest}"


def normalize_type_name(type_name):
    return " ".join(str(type_name).replace(" *", "*").split())


def manifest_text(manifest):
    return json.dumps(manifest, indent=2, sort_keys=True)
