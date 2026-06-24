from dataclasses import dataclass, field


@dataclass
class Diagnostic:
    code: str
    severity: str
    message: str
    file: str | None = None
    line: int | None = None
    column: int | None = None

    def to_dict(self):
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "file": self.file,
            "line": self.line,
            "column": self.column,
        }


@dataclass
class FunctionInfo:
    name: str
    kind: str
    file: str
    line: int
    return_type: str
    parameters: list[str] = field(default_factory=list)
    body: str = ""
    body_line: int | None = None

    def to_dict(self):
        return {
            "name": self.name,
            "kind": self.kind,
            "file": self.file,
            "line": self.line,
            "return_type": self.return_type,
            "parameters": self.parameters,
            "body": self.body,
            "body_line": self.body_line,
        }


@dataclass
class LaunchInfo:
    kernel: str
    file: str
    line: int
    grid_expr: str
    block_expr: str
    shared_expr: str = "0"
    stream_expr: str = "0"
    args: list[str] = field(default_factory=list)
    syntax: str = "cudaLaunch"
    grid: dict = field(default_factory=dict)
    block: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "kernel": self.kernel,
            "file": self.file,
            "line": self.line,
            "grid_expr": self.grid_expr,
            "block_expr": self.block_expr,
            "shared_expr": self.shared_expr,
            "stream_expr": self.stream_expr,
            "args": self.args,
            "syntax": self.syntax,
            "grid": self.grid,
            "block": self.block,
        }


@dataclass
class ProjectIR:
    entry: str
    parser: str
    sources: list[str] = field(default_factory=list)
    functions: list[FunctionInfo] = field(default_factory=list)
    launches: list[LaunchInfo] = field(default_factory=list)
    includes: list[dict] = field(default_factory=list)
    parser_diagnostics: list[dict] = field(default_factory=list)

    @property
    def kernels(self):
        return [function for function in self.functions if function.kind == "KERNEL"]

    def to_dict(self):
        return {
            "entry": self.entry,
            "parser": self.parser,
            "sources": self.sources,
            "functions": [function.to_dict() for function in self.functions],
            "kernels": [kernel.to_dict() for kernel in self.kernels],
            "launches": [launch.to_dict() for launch in self.launches],
            "includes": self.includes,
            "parser_diagnostics": self.parser_diagnostics,
        }


@dataclass
class BindingIR:
    argument_index: int
    identifier: str
    type: str
    buffer_index: int
    kind: str
    space: str

    def to_dict(self):
        return {
            "argument_index": self.argument_index,
            "identifier": self.identifier,
            "type": self.type,
            "binding_target": f"buffer({self.buffer_index})",
            "buffer_index": self.buffer_index,
            "kind": self.kind,
            "space": self.space,
        }


@dataclass
class KernelRouteIR:
    msl_entry_point: str
    bindings: list[BindingIR]
    optimization_metrics: dict
    source: dict

    def to_dict(self):
        return {
            "msl_entry_point": self.msl_entry_point,
            "argument_count": len(self.bindings),
            "bindings": [binding.to_dict() for binding in self.bindings],
            "optimization_metrics": self.optimization_metrics,
            "source": self.source,
        }
