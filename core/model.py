from dataclasses import dataclass, field


@dataclass
class Diagnostic:
    code: str
    severity: str
    message: str
    file: str | None = None
    line: int | None = None

    def to_dict(self):
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "file": self.file,
            "line": self.line,
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

    def to_dict(self):
        return {
            "name": self.name,
            "kind": self.kind,
            "file": self.file,
            "line": self.line,
            "return_type": self.return_type,
            "parameters": self.parameters,
            "body": self.body,
        }


@dataclass
class LaunchInfo:
    kernel: str
    file: str
    line: int
    grid_expr: str
    block_expr: str
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
            "args": self.args,
            "syntax": self.syntax,
            "grid": self.grid,
            "block": self.block,
        }
