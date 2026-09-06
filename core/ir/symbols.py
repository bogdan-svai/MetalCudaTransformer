import hashlib
import json
from dataclasses import dataclass, field

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


@dataclass(frozen=True)
class StatementIR:
    """Typed AST statement retained for later lowering and optimization passes."""

    kind: str
    source: SourceLocationIR
    text: str
    children: tuple[str, ...] = tuple()
    attributes: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "kind": self.kind,
            "source": self.source.to_dict(),
            "text": self.text,
            "children": list(self.children),
            "attributes": self.attributes,
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
    statements: list[StatementIR] = field(default_factory=list)
    effects: dict = field(default_factory=dict)

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
            "body": self.body,
            "translation_unit": self.translation_unit,
            "calls": [call.to_dict() for call in self.calls],
            "local_objects": [item.to_dict() for item in self.local_objects],
            "statements": [item.to_dict() for item in self.statements],
            "effects": self.effects,
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
    _name_index: dict[str, list[FunctionSymbolIR]] | None = field(
        default=None, init=False, repr=False
    )

    def functions_named(self, name):
        if self._name_index is None:
            index = {}
            for function in self.functions.values():
                index.setdefault(function.name, []).append(function)
            self._name_index = index
        return self._name_index.get(name, [])

    def to_dict(self):
        return {
            "frontend": self.frontend,
            "translation_units": self.translation_units,
            "functions": {
                symbol_id: function.to_dict()
                for symbol_id, function in sorted(self.functions.items())
            },
            "types": {
                name: declaration.to_dict() for name, declaration in sorted(self.types.items())
            },
            "macros": {name: macro.to_dict() for name, macro in sorted(self.macros.items())},
            "diagnostics": self.diagnostics,
        }

    @classmethod
    def from_dict(cls, payload):
        functions = {}
        for symbol_id, item in payload.get("functions", {}).items():
            functions[symbol_id] = FunctionSymbolIR(
                symbol_id=item["symbol_id"],
                name=item["name"],
                kind=item["kind"],
                qualifiers=tuple(item.get("qualifiers", [])),
                return_type=_type_from_dict(item["return_type"]),
                parameters=[_parameter_from_dict(value) for value in item.get("parameters", [])],
                source=_source_from_dict(item["source"]),
                body_line=item.get("body_line"),
                body=item.get("body", ""),
                translation_unit=item.get("translation_unit", ""),
                calls=[_call_from_dict(value) for value in item.get("calls", [])],
                local_objects=[_local_from_dict(value) for value in item.get("local_objects", [])],
                statements=[_statement_from_dict(value) for value in item.get("statements", [])],
                effects=item.get("effects", {}),
            )
        types = {
            name: TypeDeclarationIR(
                name=item["name"],
                kind=item["kind"],
                fields=tuple(_parameter_from_dict(value) for value in item.get("fields", [])),
                underlying_type=item.get("underlying_type"),
                source=_source_from_dict(item["source"]),
            )
            for name, item in payload.get("types", {}).items()
        }
        macros = {
            name: MacroIR(
                name=item["name"],
                parameters=tuple(item.get("parameters", [])),
                expression=item["expression"],
                source=_source_from_dict(item["source"]),
            )
            for name, item in payload.get("macros", {}).items()
        }
        return cls(
            frontend=payload["frontend"],
            translation_units=list(payload.get("translation_units", [])),
            functions=functions,
            types=types,
            macros=macros,
            diagnostics=list(payload.get("diagnostics", [])),
        )


def _source_from_dict(payload):
    return SourceLocationIR(payload["file"], payload.get("line"), payload.get("column"))


def _type_from_dict(payload):
    return TypeIR(
        payload["spelling"],
        payload["canonical"],
        payload.get("kind", "value"),
        payload.get("is_const", False),
        payload.get("pointer_depth", 0),
    )


def _parameter_from_dict(payload):
    return ParameterIR(payload["name"], _type_from_dict(payload["type"]))


def _call_from_dict(payload):
    return CallEdgeIR(
        payload["caller_id"],
        payload["callee_name"],
        payload.get("callee_id"),
        tuple(payload.get("arguments", [])),
        _source_from_dict(payload["source"]),
        tuple(_type_from_dict(item) for item in payload.get("argument_types", [])),
        tuple(payload.get("argument_symbols", [])),
    )


def _local_from_dict(payload):
    return LocalObjectIR(
        payload["name"],
        _type_from_dict(payload["type"]),
        payload["storage"],
        payload.get("array_extent"),
        payload.get("variable_length", False),
        payload.get("incomplete_array", False),
        _source_from_dict(payload["source"]),
    )


def _statement_from_dict(payload):
    return StatementIR(
        payload["kind"],
        _source_from_dict(payload["source"]),
        payload.get("text", ""),
        tuple(payload.get("children", [])),
        payload.get("attributes", {}),
    )


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
