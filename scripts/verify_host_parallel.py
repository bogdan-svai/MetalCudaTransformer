"""Differential native GPU verification for generated host-loop dispatch."""

import argparse
import json
import os
import platform
import time
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.application.pipeline import CompilerPipeline

SOURCE = """
unsigned run(unsigned n) {
    unsigned a[65544]={}; unsigned b[65544]={};
    unsigned* const input=a+4; unsigned* const output=b+4;
    for(unsigned i=0;i<n;++i) output[i]=input[i]*3u+i+7u;
    // Oracle includes canaries on both sides of written regions.
    if(b[0] || b[1] || b[2] || b[3]) return 1;
    for(unsigned j=0;j<65544;++j) {
        unsigned expected=(j>=4 && j<4+n) ? (j-4)+7u : 0u;
        if(b[j]!=expected) return 2;
        if(a[j]!=0) return 3;
    }
    return 0;
}
unsigned inplace(unsigned n) {
    unsigned a[65544]={11u};
    for(unsigned i=0;i<n;++i) a[i]=(a[i]+i)*4294967295u;
    for(unsigned j=0;j<65544;++j) {
        unsigned old=j==0 ? 11u : 0u;
        unsigned expected=j<n ? (old+j)*4294967295u : old;
        if(a[j]!=expected) return 1;
    }
    return 0;
}
unsigned alias(unsigned n) {
    unsigned a[131096]={11u,17u,31u,41u,53u,67u,79u,97u};
    unsigned seed[8]={11u,17u,31u,41u,53u,67u,79u,97u};
    unsigned* const p=a; unsigned* const q=a+65548;
    for(unsigned i=0;i<n;++i) q[i]=p[i]+7u;
    for(unsigned j=0;j<131096;++j) {
        unsigned expected=j<8 ? seed[j] : 0u;
        if(j>=65548 && j<65548+n) {
            unsigned k=j-65548;
            expected=(k<8 ? seed[k] : 0u)+7u;
        }
        if(a[j]!=expected) return 1;
    }
    return 0;
}
unsigned fallback(unsigned n) {
    unsigned a[65544]={11u};
    unsigned* const p=a; unsigned* const q=a+1;
    for(unsigned i=0;i<n;++i) q[i]=p[i]+1u;
    for(unsigned j=0;j<65544;++j) {
        unsigned expected=j<=n ? 11u+j : 0u;
        if(a[j]!=expected) return 1;
    }
    return 0;
}
int main() {
    unsigned sizes[]={0,1,63,64,65,1023,65537};
    for(unsigned k=0;k<7;++k) {
        if(run(sizes[k])) return 10+k;
        if(inplace(sizes[k])) return 20+k;
        if(alias(sizes[k])) return 30+k;
        if(fallback(sizes[k])) return 40+k;
    }
    return 0;
}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    original = directory / "original.cpp"
    original.write_text(SOURCE)
    cpu = directory / "cpu"
    subprocess.run(
        ["xcrun", "clang++", "-std=c++17", "-O2", str(original), "-o", str(cpu)], check=True
    )
    subprocess.run([str(cpu)], check=True)
    pipeline = CompilerPipeline(root, artifact_path=str(directory / "artifacts.zip"))
    build_started = time.perf_counter()
    built = pipeline.build(original, native=True, output=directory / "gpu", auto_parallel="safe")
    native = built.payload.get("build", {}).get(
        "native", {"ok": False, "diagnostics": built.payload["diagnostics"]}
    )
    host_report = built.payload.get("auto_parallel", {})
    assert host_report.get("transformed_loop_count") == 4, built.payload
    build_seconds = time.perf_counter() - build_started
    result = {
        "native_build_seconds": build_seconds,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "hardware": subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True, capture_output=True
        ).stdout.strip(),
        "build": native,
        "cpu_oracle": True,
        "sizes": [0, 1, 63, 64, 65, 1023, 65537],
        "capture_plans": [
            item["captures"] for item in host_report["loops"] if item.get("captures")
        ],
        "workloads": ["map", "inplace_unsigned", "disjoint_alias", "conditional_cpu_fallback"],
        "gpu_oracle": False,
    }
    if native["ok"]:
        gpu_started = time.perf_counter()
        metrics_path = directory / "runtime_metrics.json"
        metrics_path.write_text("{}")
        environment = os.environ.copy()
        environment["MCU_METRICS_FILE"] = str(metrics_path)
        process = subprocess.run(
            [native["executable"]], text=True, capture_output=True, env=environment
        )
        metrics = json.loads(metrics_path.read_text())
        dispatches = metrics.get("counters", {}).get("kernel_dispatches", 0)
        result["kernel_dispatches"] = dispatches
        result.update(
            {
                "gpu_process_seconds": time.perf_counter() - gpu_started,
                "gpu_exit_code": process.returncode,
                "gpu_stdout": process.stdout,
                "gpu_stderr": process.stderr,
                "gpu_oracle": process.returncode == 0 and dispatches == 19,
            }
        )
    if native["ok"] and result["gpu_oracle"]:
        repeated = pipeline.build(
            original, native=True, output=directory / "gpu", auto_parallel="safe"
        )
        result["native_noop_cache_hit"] = repeated.payload["build"]["native"].get(
            "cache_hit", False
        )
        assert repeated.exit_code == 0 and result["native_noop_cache_hit"]
        cpu_build = pipeline.build(
            original, native=True, output=directory / "gpu", auto_parallel="report"
        )
        assert cpu_build.exit_code == 0, cpu_build.payload["diagnostics"]
        assert cpu_build.payload["auto_parallel"]["transformed_loop_count"] == 0
        assert not cpu_build.payload["build"]["native"].get("cache_hit")
        subprocess.run([cpu_build.payload["build"]["native"]["executable"]], check=True)
        safe_again = pipeline.build(
            original, native=True, output=directory / "gpu", auto_parallel="safe"
        )
        assert safe_again.exit_code == 0, safe_again.payload["diagnostics"]
        assert not safe_again.payload["build"]["native"].get("cache_hit")
        result["mode_change_invalidated_cache"] = True
    result["ok"] = native["ok"] and result["gpu_oracle"]
    (directory / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
