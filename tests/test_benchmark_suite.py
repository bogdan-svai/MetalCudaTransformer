import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _module():
    spec = importlib.util.spec_from_file_location(
        "benchmark_suite", ROOT / "scripts" / "benchmark_suite.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_benchmark_summary_reports_median_and_p95():
    result = _module().summary([4.0, 1.0, 3.0, 2.0, 5.0])
    assert result["minimum"] == 1.0
    assert result["median"] == 3.0
    assert result["p95"] == 5.0


def test_benchmark_functions_use_three_warmups_by_default():
    module = _module()
    assert module.benchmark_stream.__defaults__ == (3,)
    assert module.benchmark_hotspot.__defaults__ == (3,)
    assert module.benchmark_reduction.__defaults__ == (3,)
    assert module.benchmark_histogram.__defaults__ == (3,)


def test_benchmark_tiers_scale_input_sizes_monotonically():
    module = _module()
    small = module._tier_parameters("small")
    medium = module._tier_parameters("medium")
    large = module._tier_parameters("large")
    assert small["elements"] < medium["elements"] < large["elements"]
    assert small["hotspot_size"] < medium["hotspot_size"] < large["hotspot_size"]
    assert small["reduction_elements"] < medium["reduction_elements"] < large["reduction_elements"]


def test_babelstream_variant_is_never_labelled_official():
    manifest = __import__("json").loads(
        (ROOT / "benchmarks" / "babelstream_core" / "benchmark.json").read_text(encoding="utf-8")
    )
    assert manifest["provenance"]["classification"] == "derived_compatibility_variant"
    assert manifest["provenance"]["official_babelstream_result"] is False


def test_hotspot_variant_is_never_labelled_official():
    manifest = __import__("json").loads(
        (ROOT / "benchmarks" / "hotspot_core" / "benchmark.json").read_text(encoding="utf-8")
    )
    assert manifest["category"] == "stencil_2d_shared_memory"
    assert manifest["provenance"]["classification"] == "derived_compatibility_variant"
    assert manifest["provenance"]["official_rodinia_result"] is False


def test_reduction_manifest_defines_transfer_and_operation_counts():
    manifest = __import__("json").loads(
        (ROOT / "benchmarks" / "reduction_core" / "benchmark.json").read_text(encoding="utf-8")
    )
    assert manifest["category"] == "shared_memory_reduction"
    assert manifest["bytes_per_element"] == 8
    assert manifest["operations_per_element"] == 2
    assert manifest["provenance"]["official_babelstream_result"] is False


def test_histogram_profiles_order_contention_by_bin_count():
    manifest = __import__("json").loads(
        (ROOT / "benchmarks" / "histogram_core" / "benchmark.json").read_text(encoding="utf-8")
    )
    profiles = manifest["contention_profiles"]
    assert profiles["high"] < profiles["medium"] < profiles["low"]
    assert manifest["provenance"]["external_dependencies"] == []
