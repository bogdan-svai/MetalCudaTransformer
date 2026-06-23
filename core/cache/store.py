import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


def fingerprint(payload):
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class CacheStore:
    def __init__(self, root=None):
        configured = root or os.environ.get("MCU_CACHE_DIR")
        self.root = Path(configured or Path.home() / ".cache" / "mcu").expanduser()

    def path(self, layer, key, name):
        return self.root / layer / key / name

    def restore(self, layer, key, name, destination):
        source = self.path(layer, key, name)
        if not _valid_entry(source):
            return False
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.mcu-cache")
        shutil.copy2(source, temporary)
        os.replace(temporary, target)
        return True

    def publish(self, layer, key, name, source):
        target = self.path(layer, key, name)
        if _valid_entry(target):
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{name}.", dir=target.parent)
        os.close(descriptor)
        temporary = Path(temporary_name)
        checksum_temporary = temporary.with_suffix(temporary.suffix + ".sha256")
        try:
            shutil.copy2(source, temporary)
            checksum_temporary.write_text(_digest(temporary) + "\n", encoding="ascii")
            os.replace(temporary, target)
            os.replace(checksum_temporary, _checksum_path(target))
        finally:
            if temporary.exists():
                temporary.unlink()
            if checksum_temporary.exists():
                checksum_temporary.unlink()
        return target


def _checksum_path(path):
    return Path(f"{path}.sha256")


def _digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_entry(path):
    target = Path(path)
    checksum = _checksum_path(target)
    if not target.is_file() or not checksum.is_file():
        return False
    try:
        return checksum.read_text(encoding="ascii").strip() == _digest(target)
    except OSError:
        return False
