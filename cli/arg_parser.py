import argparse


def build_parser():
    parser = argparse.ArgumentParser(
        prog="mcu",
        description="MCU CUDA-to-Metal pipeline utility",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    check = subcommands.add_parser(
        "check",
        help="Analyze CUDA/C++ source structure and run Stage 1 sanity checks",
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
        help="Run Stage 1 validation and generate Metal source plus routing metadata",
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

    for name, help_text in [
        ("compile", "Compile CUDA/C++ inputs into an MCU object or native executable"),
        ("run", "Compile and run a native CUDA-to-Metal executable"),
        ("make", "Run an existing Makefile through the temporary MCU NVCC shim"),
        ("nvcc-driver", "Internal NVCC-compatible driver entry point"),
    ]:
        command = subcommands.add_parser(name, help=help_text)
        command.add_argument("arguments", nargs=argparse.REMAINDER)
    return parser
