import re

from core.ir.models import Diagnostic


DIAGNOSTIC_PATTERN = re.compile(
    r"^(?P<file>.+?):(?P<line>\d+):(?:(?P<column>\d+):)?\s*"
    r"(?P<severity>error|warning|note):\s*(?P<message>.*)$"
)


def parse_compiler_diagnostics(stderr, code="NativeCompileDiagnostic"):
    diagnostics = []
    for line in stderr.splitlines():
        match = DIAGNOSTIC_PATTERN.match(line.strip())
        if not match:
            continue
        severity = match.group("severity")
        diagnostics.append(Diagnostic(
            code=code,
            severity="error" if severity == "error" else "warning",
            message=match.group("message"),
            file=match.group("file"),
            line=int(match.group("line")),
            column=int(match.group("column")) if match.group("column") else None,
        ).to_dict())
    return diagnostics
