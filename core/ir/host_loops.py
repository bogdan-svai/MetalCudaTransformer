"""Original-source host AST records. These records do not prove loop legality."""

from dataclasses import asdict, dataclass, field

HOST_LOOP_SCHEMA = 5


@dataclass
class HostNodeIR:
    kind: str
    type: str
    file: str
    start: int
    end: int
    spelling: str = ""
    symbol_usr: str | None = None
    operator: str | None = None
    integer_value: int | None = None
    array_extent: int | None = None
    storage: str = ""
    initialized: bool = False
    tokens: list[str] = field(default_factory=list)
    children: list["HostNodeIR"] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        values = dict(data)
        values["children"] = [cls.from_dict(item) for item in values.get("children", [])]
        return cls(**values)


@dataclass
class HostLoopIR:
    function_usr: str
    function: str
    translation_unit: str
    line: int
    column: int
    node: HostNodeIR
    init: HostNodeIR | None
    condition: HostNodeIR | None
    update: HostNodeIR | None
    body: HostNodeIR | None
    reasons: list[str]
    source_sha256: str = ""

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        values = dict(data)
        for key in ("node", "init", "condition", "update", "body"):
            values[key] = HostNodeIR.from_dict(values[key]) if values[key] else None
        return cls(**values)
