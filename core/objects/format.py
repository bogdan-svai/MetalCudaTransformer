import hashlib
import json
import os
from pathlib import Path
import platform
import re
import tempfile
import subprocess
from functools import lru_cache

from core.frontend.regex_parser import read_text
from core.version import MCU_OBJECT_SCHEMA, MCU_VERSION, RUNTIME_ABI


MAGIC = "MCUOBJ\n"
LOCAL_INCLUDE = re.compile(r'^\s*#\s*include\s*"([^"]+)"', re.MULTILINE)


class McuObjectError(ValueError):
    def __init__(self, code, message, path=None):
        super().__init__(message)
        self.code = code
        self.path = str(path) if path else None

    def diagnostic(self):
        return {
            "code": self.code,
            "severity": "error",
            "message": str(self),
            "file": self.path,
            "line": None,
            "column": None,
            "phase": "object",
        }


def create_mcu_object(source, invocation):
    source_path = Path(source).resolve()
    dependencies = _collect_dependencies(source_path, invocation.include_paths)
    records = []
    for path in dependencies:
        content = path.read_bytes()
        records.append({
            "path": str(path),
            "sha256": hashlib.sha256(content).hexdigest(),
        })
    return {
        "schema_version": MCU_OBJECT_SCHEMA,
        "mcu_version": MCU_VERSION,
        "runtime_abi": RUNTIME_ABI,
        "target": {
            "system": platform.system(),
            "architecture": platform.machine(),
            "os_version": platform.mac_ver()[0],
            "sdk": _sdk_identity(),
        },
        "source": str(source_path),
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "dependencies": records,
        "invocation": invocation.to_dict(),
        "capabilities": ["typed_device_symbols", "device_link"],
    }


def write_mcu_object(path, payload):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    serialized = MAGIC + json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return target


def load_mcu_object(path, validate_inputs=True):
    target = Path(path)
    try:
        content = target.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise McuObjectError("CorruptMcuObject", f"Cannot read MCU object: {exc}", target) from exc
    if not content.startswith(MAGIC):
        raise McuObjectError("IncompatibleMcuObject", "Input is not an MCU object.", target)
    try:
        payload = json.loads(content[len(MAGIC):])
    except json.JSONDecodeError as exc:
        raise McuObjectError("CorruptMcuObject", f"Invalid MCU object JSON: {exc}", target) from exc
    if payload.get("schema_version") != MCU_OBJECT_SCHEMA:
        raise McuObjectError("IncompatibleMcuObject", "Unsupported MCU object schema.", target)
    if payload.get("runtime_abi") != RUNTIME_ABI:
        raise McuObjectError("IncompatibleMcuObject", "MCU runtime ABI mismatch.", target)
    if payload.get("mcu_version") != MCU_VERSION:
        raise McuObjectError("IncompatibleMcuObject", "MCU compiler version mismatch.", target)
    object_target = payload.get("target", {})
    if object_target.get("architecture") != platform.machine():
        raise McuObjectError("IncompatibleMcuObject", "MCU object architecture mismatch.", target)
    if object_target.get("sdk") != _sdk_identity():
        raise McuObjectError("IncompatibleMcuObject", "MCU object SDK mismatch.", target)
    if validate_inputs:
        _validate_inputs(payload, target)
    return payload


def _collect_dependencies(source, include_paths):
    discovered = []
    visited = set()

    def visit(path):
        path = path.resolve()
        if path in visited or not path.is_file():
            return
        visited.add(path)
        discovered.append(path)
        text = read_text(path)
        for include in LOCAL_INCLUDE.findall(text):
            candidates = [path.parent / include, *[Path(root) / include for root in include_paths]]
            resolved = next((candidate for candidate in candidates if candidate.is_file()), None)
            if resolved:
                visit(resolved)

    visit(source)
    return discovered


def _validate_inputs(payload, object_path):
    for record in payload.get("dependencies", []):
        path = Path(record["path"])
        if not path.is_file():
            raise McuObjectError("IncompatibleMcuObject", f"Object dependency no longer exists: {path}", object_path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != record.get("sha256"):
            raise McuObjectError("IncompatibleMcuObject", f"Object dependency changed after compilation: {path}", object_path)


@lru_cache(maxsize=1)
def _sdk_identity():
    try:
        result = subprocess.run(
            ["xcrun", "--sdk", "macosx", "--show-sdk-path"],
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError:
        return "unavailable"
    return result.stdout.strip() or "unavailable"
