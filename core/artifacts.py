import json
import zipfile
from pathlib import Path


def write_artifact(payload, artifact_path="mcu_artifacts.zip"):
    target = Path(artifact_path)
    serialized = json.dumps(payload, indent=2, sort_keys=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("ast_output.txt", serialized)
    return target
