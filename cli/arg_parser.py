import argparse


def build_parser():
    parser = argparse.ArgumentParser(
        prog="mcu",
        description="MCU CUDA-to-Metal pipeline utility",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    check = subcommands.add_parser(
        "check",
        help="Analyze CUDA/C++ source structure and validate kernel launches",
    )
    check.add_argument("path", help="Source file or directory to analyze")
    check.add_argument(
        "--json",
        action="store_true",
        help="Emit a single JSON document to stdout",
    )
    check.add_argument(
        "--verbose",
        action="store_true",
        help="Emit parser diagnostics in human-readable mode",
    )

    build = subcommands.add_parser(
        "build",
        help="Validate CUDA/C++ sources and generate Metal source plus routing metadata",
    )
    build.add_argument("path", help="Source file or directory to build")
    build.add_argument(
        "--json",
        action="store_true",
        help="Emit a single JSON document to stdout",
    )
    build.add_argument(
        "--verbose",
        action="store_true",
        help="Emit parser diagnostics in human-readable mode",
    )
    build.add_argument(
        "--coarsening",
        type=int,
        choices=[1, 2, 4, 8],
        default=None,
        help="Logical CUDA X-axis threads processed by each physical Metal thread",
    )
    build.add_argument(
        "--native",
        action="store_true",
        help="Compile transformed host sources and the Metal runtime into a macOS executable",
    )
    build.add_argument(
        "--output",
        default=".mcu/build",
        help="Native build directory (used with --native)",
    )

    build.add_argument(
        "--auto-parallel", choices=["off", "report", "safe"], default="off",
        help="Experimental proven host-loop offload; safe requires --native",
    )

    analyze = subcommands.add_parser(
        "analyze",
        help="Analyze a source tree and report compatibility diagnostics",
    )
    analyze.add_argument("path", help="Source file or directory to analyze")
    analyze.add_argument(
        "--json",
        action="store_true",
        help="Emit a single JSON document to stdout",
    )
    analyze.add_argument(
        "--verbose",
        action="store_true",
        help="Include parser diagnostics in metadata",
    )

    analyze.add_argument(
        "--auto-parallel", choices=["off", "report"], default="off",
        help="Report original host loops without changing execution",
    )

    capabilities = subcommands.add_parser(
        "capabilities",
        help="Report MCU's supported CUDA, runtime, NVCC, and library surface",
    )
    capabilities.add_argument(
        "--json",
        action="store_true",
        help="Emit a single JSON document to stdout",
    )

    for name, help_text in [
        ("compile", "Compile CUDA/C++ inputs into an MCU object or native executable"),
        ("run", "Compile and run a native CUDA-to-Metal executable"),
        ("make", "Run an existing Makefile through the temporary MCU NVCC shim"),
        ("nvcc-driver", "Internal NVCC-compatible driver entry point"),
    ]:
        command = subcommands.add_parser(name, help=help_text)
        command.add_argument("arguments", nargs=argparse.REMAINDER)
    return parser
