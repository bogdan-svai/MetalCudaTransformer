"""Measure optimized CPU map and original Clang parsing before GPU lowering."""

import argparse
import json
import platform
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

from clang.cindex import Config, Index

# Allow direct execution from the repository scripts directory.
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.frontend.clang_probe import _configure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    corpus = root / "tests/fixtures/host_loops/loops.cpp"
    assert _configure(Config), "libclang unavailable"
    parse_ms = []
    for _ in range(8):
        start = time.perf_counter()
        unit = Index.create().parse(str(corpus), args=["-std=c++17"])
        assert not [str(d) for d in unit.diagnostics if d.severity >= d.Error]
        parse_ms.append((time.perf_counter() - start) * 1000)
    with tempfile.TemporaryDirectory(prefix="mcu-host-baseline-") as temporary:
        directory = Path(temporary)
        source = directory / "baseline.cpp"
        source.write_text(
            """#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
extern void map(unsigned*, const unsigned*, unsigned);
void observe(unsigned) {}
int main(int argc, char** argv) {
    if(argc != 2) return 2;
    unsigned n = std::strtoul(argv[1], nullptr, 10);
    std::vector<unsigned> in(n), out(n);
    for(unsigned i=0;i<n;++i) in[i]=i;
    for(int sample=0;sample<8;++sample) {
        auto start=std::chrono::steady_clock::now();
        map(out.data(), in.data(), n);
        auto end=std::chrono::steady_clock::now();
        for(unsigned i=0;i<n;++i) if(out[i] != i*3u+1u) return 3;
        if(sample>=3) std::printf("%.9f\\n", std::chrono::duration<double>(end-start).count());
    }
}
"""
        )
        executable = directory / "baseline"
        command = ["clang++", "-std=c++17", "-O3", str(corpus), str(source), "-o", str(executable)]
        subprocess.run(command, check=True, capture_output=True, text=True)
        cases = []
        for size in [0, 1, 1023, 65536, 1048576, 16777216]:
            run = subprocess.run(
                [str(executable), str(size)], check=True, capture_output=True, text=True
            )
            values = [float(value) for value in run.stdout.splitlines()]
            assert len(values) == 5
            cases.append(
                {
                    "iterations": size,
                    "buffer_bytes": size * 8,
                    "samples_seconds": values,
                    "median_seconds": statistics.median(values),
                    "p95_seconds": sorted(values)[-1],
                    "oracle": "all elements exact",
                    "ok": True,
                }
            )
    report = {
        "ok": True,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "compiler": subprocess.run(
            ["clang++", "--version"], capture_output=True, text=True, check=True
        ).stdout,
        "compile_command": command,
        "samples": 5,
        "warmups": 3,
        "frontend": {
            "first_parse_ms": parse_ms[0],
            "subsequent_parse_ms": parse_ms[3:],
            "median_ms": statistics.median(parse_ms[3:]),
            "scope": "new Index and original corpus parse; no MCU cache; includes Clang diagnostics",
        },
        "cases": cases,
        "limitations": "CPU map only; p95 nearest-rank from five samples; buffer bytes are allocated payload, not observed RSS. No GPU speedup claim.",
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
