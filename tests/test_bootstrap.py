import os
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def prepare(tmp_path):
    commands = tmp_path / "commands"
    commands.mkdir()
    uname = commands / "uname"
    uname.write_text('#!/bin/bash\ncase "$1" in -s) echo Darwin;; -m) echo arm64;; esac\n')
    curl = commands / "curl"
    curl.write_text(
        "#!/bin/bash\nset -eu\n"
        'touch "$MCU_TEST_DOWNLOAD_MARKER"\n'
        'if [[ "${MCU_TEST_DOWNLOAD_FAIL:-0}" == 1 ]]; then exit 22; fi\n'
        'while [[ "$1" != --output ]]; do shift; done\n'
        'cp "$MCU_TEST_ARCHIVE" "$2"\n'
    )
    uname.chmod(0o755)
    curl.chmod(0o755)
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    (fixture / "mcu").write_text("fixture command\n")
    (fixture / "install.sh").write_text(
        "#!/bin/bash\nset -eu\n"
        'printf "%s\\n" "$*" >> "$MCU_TEST_INSTALL_LOG"\n'
        'if [[ "${1:-}" == --check && "${MCU_TEST_CHECK_FAIL:-0}" == 1 ]]; then exit 23; fi\n'
    )
    archive = tmp_path / "fixture.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(fixture, arcname="repository-ref")
    return dict(
        os.environ,
        PATH=str(commands) + os.pathsep + os.environ["PATH"],
        MCU_SOURCE_DIR=str(tmp_path / "source with $ and ` characters"),
        MCU_TEST_ARCHIVE=str(archive),
        MCU_TEST_INSTALL_LOG=str(tmp_path / "installation.log"),
        MCU_TEST_DOWNLOAD_MARKER=str(tmp_path / "downloaded"),
    )


def run(environment):
    return subprocess.run(
        ["bash", str(ROOT / "scripts/bootstrap.sh")],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def test_bootstrap_checks_before_installing_and_preserves_source(tmp_path):
    environment = prepare(tmp_path)
    result = run(environment)
    assert result.returncode == 0, result.stderr
    assert (Path(environment["MCU_SOURCE_DIR"]) / "mcu").is_file()
    assert Path(environment["MCU_TEST_INSTALL_LOG"]).read_text().splitlines() == ["--check", ""]


def test_bootstrap_refuses_existing_source_without_downloading(tmp_path):
    environment = prepare(tmp_path)
    source = Path(environment["MCU_SOURCE_DIR"])
    source.mkdir()
    (source / "keep").write_text("unchanged")
    result = run(environment)
    assert result.returncode != 0
    assert "already exists" in result.stderr
    assert (source / "keep").read_text() == "unchanged"
    assert not Path(environment["MCU_TEST_DOWNLOAD_MARKER"]).exists()


def test_bootstrap_prerequisite_failure_does_not_create_source_directory(tmp_path):
    environment = prepare(tmp_path)
    environment["MCU_TEST_CHECK_FAIL"] = "1"
    assert run(environment).returncode == 23
    assert not Path(environment["MCU_SOURCE_DIR"]).exists()


def test_bootstrap_download_failure_does_not_install(tmp_path):
    environment = prepare(tmp_path)
    environment["MCU_TEST_DOWNLOAD_FAIL"] = "1"
    assert run(environment).returncode == 22
    assert not Path(environment["MCU_SOURCE_DIR"]).exists()
    assert not Path(environment["MCU_TEST_INSTALL_LOG"]).exists()
