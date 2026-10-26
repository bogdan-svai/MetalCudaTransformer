import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
MCU = ROOT / "mcu"
FIXTURE = ROOT / "examples" / "sha256_miner"


def _fields(output):
    return dict(line.split("=", 1) for line in output.splitlines() if "=" in line)


def _oracle(project, input_name):
    result = subprocess.run(
        [sys.executable, "oracle.py", input_name],
        cwd=project,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return _fields(result.stdout)


def test_independent_oracle_has_stable_sha_merkle_nonce_and_block_hash():
    assert _oracle(FIXTURE, "inputs/test1.in") == {
        "sha256_first": "8ed3f6ad685b959ead7022518e1af76cd816f8e8ec7ccdda1ed4018e8f2223f8",
        "merkle_root": "56159c1e12bd685143a1b7b1741502124dc07ff62f57cedb3f2bb71187e9b244",
        "nonce": "20632",
        "block_hash": "0000ce278e4b6ad08f9c3a3694af81f16e53ae6d1b1593eb18aaf5ea048ad174",
    }


@pytest.mark.skipif(platform.system() != "Darwin", reason="native project requires Metal")
def test_make_project_matches_oracle_is_deterministic_and_emits_metrics(tmp_path):
    project = tmp_path / "project"
    shutil.copytree(FIXTURE, project)
    environment = os.environ.copy()
    environment["MCU_CACHE_DIR"] = str(tmp_path / "cache")
    build = subprocess.run(
        [sys.executable, str(MCU), "make", "-C", str(project), "clean", "compile", "--json"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert build.returncode == 0, build.stderr or build.stdout
    payload = json.loads(build.stdout)
    assert len(payload["make"]["driver_invocations"]) == 4
    assert all(item["ok"] for item in payload["make"]["driver_invocations"])

    expected = _oracle(project, "inputs/test1.in")
    outputs = []
    for index in range(5):
        metrics_path = tmp_path / f"runtime-{index}.json"
        run_environment = environment.copy()
        run_environment["MCU_METRICS_FILE"] = str(metrics_path)
        result = subprocess.run(
            [str(project / "miner"), "inputs/test1.in"],
            cwd=project,
            env=run_environment,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr or result.stdout
        outputs.append(_fields(result.stdout))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        assert metrics["schema_version"] == 1
        assert metrics["counters"]["kernel_dispatches"] >= 4
        assert metrics["counters"]["memory_copy_bytes"] > 0
        assert metrics["phases"]["artifact_load"]["count"] == 1
        assert metrics["phases"]["command_submission"]["count"] >= 4
        assert metrics["phases"]["command_submission.kernel.find_nonce"]["count"] == 1
        assert metrics["phases"]["gpu_execution.kernel.find_nonce"]["count"] == 1
        assert metrics["counters"]["kernel_dispatches.find_nonce"] == 1
        assert metrics["phases"]["synchronization_wait"]["count"] >= 4

    assert all(output == expected for output in outputs)

    second = subprocess.run(
        [str(project / "miner"), "inputs/test2.in"],
        cwd=project,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert second.returncode == 0, second.stderr or second.stdout
    assert _fields(second.stdout) == _oracle(project, "inputs/test2.in")
