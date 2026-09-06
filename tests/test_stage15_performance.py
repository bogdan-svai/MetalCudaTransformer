import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _stage15():
    spec = importlib.util.spec_from_file_location(
        "verify_stage15",
        ROOT / "scripts" / "verify_stage15.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stage15_summary_reports_percentiles_and_slowest_cases():
    stage15 = _stage15()
    report = {
        "ok": True,
        "passed": 3,
        "total": 3,
        "cases": [
            {
                "id": "01",
                "name": "easy",
                "tier": "easy",
                "passed": True,
                "run_seconds": 0.1,
                "build_seconds": 0.5,
            },
            {
                "id": "02",
                "name": "medium",
                "tier": "medium",
                "passed": True,
                "run_seconds": 0.3,
                "build_seconds": 0.7,
            },
            {
                "id": "03",
                "name": "hard",
                "tier": "hard",
                "passed": True,
                "run_seconds": 0.2,
                "build_seconds": 0.6,
            },
        ],
    }

    summary = stage15.summarize_checker(report)

    assert summary["ok"] is True
    assert summary["passed"] == 3
    assert summary["tiers"]["easy"]["run_seconds"]["median"] == 0.1
    assert summary["tiers"]["medium"]["build_seconds"]["p95"] == 0.7
    assert summary["slowest_cases"][0]["id"] == "02"


def test_stage15_stage9_summary_requires_budget_passes():
    stage15 = _stage15()
    report = {
        "ok": True,
        "machine": {"device": "Apple Test"},
        "correctness": {
            "inputs": {
                "inputs/test1.in": {
                    "nonce": "20632",
                    "block_hash": "0000ce278e4b6ad08f9c3a3694af81f16e53ae6d1b1593eb18aaf5ea048ad174",
                }
            }
        },
        "build": {
            "clean_seconds": 1.0,
            "incremental_seconds": 0.2,
            "no_op": {"p95": 0.05},
        },
        "runtime": {
            "inputs": {
                "inputs/test1.in": {
                    "metrics": [
                        {
                            "phases": {
                                "gpu_execution.kernel.find_nonce": {
                                    "seconds": 0.003,
                                    "count": 1,
                                }
                            }
                        }
                    ]
                }
            }
        },
        "budgets": {
            "observed": {
                "clean_pass": True,
                "incremental_pass": True,
                "make_no_op_p95_pass": False,
            }
        },
    }

    summary = stage15.summarize_stage9(report)

    assert summary["ok"] is False
    assert summary["device"] == "Apple Test"
    assert summary["no_op"]["p95"] == 0.05
    assert summary["nonce_results"]["inputs/test1.in"]["nonce"] == "20632"
    assert summary["kernel_gpu_seconds"]["inputs/test1.in"]["find_nonce"]["median"] == 0.003


def test_stage9_geometry_configuration_changes_only_find_nonce_launch(tmp_path):
    stage9 = _stage15().__dict__.get("verify_stage9")
    if stage9 is None:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "verify_stage9", ROOT / "scripts" / "verify_stage9.py"
        )
        stage9 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(stage9)
    project = tmp_path / "project"
    (project / "app").mkdir(parents=True)
    main_source = project / "app" / "main.cpp"
    main_source.write_text(
        "hash_leaves<<<1, 32>>>(a);\nfind_nonce<<<32, 128>>>(b);\n",
        encoding="utf-8",
    )

    stage9.configure_find_nonce_geometry(project, 64, 256)

    rewritten = main_source.read_text(encoding="utf-8")
    assert "hash_leaves<<<1, 32>>>" in rewritten
    assert "find_nonce<<<64, 256>>>" in rewritten
