# Use MCU with your own CUDA project

MCU is installed once, then used like any other command. Your CUDA project can
live anywhere on disk; it does not need to be copied into the MCU repository.

## 1. Install MCU once

Open a terminal in the MCU checkout and install its Python dependencies:

```bash
cd ~/Projects/CudaMetalTransformer
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Install the `mcu` command in your user bin directory:

```bash
MCU_INSTALL_DIR="$HOME/.local/bin" ./scripts/install_mcu.sh
export PATH="$HOME/.local/bin:$PATH"
```

The installer creates a small wrapper that points back to this checkout. You do
not need to copy files into every CUDA project. Add the `export PATH=...` line to
your shell profile if you want the command to be available in new terminals.

## 2. Try it on a project somewhere else

For example, suppose your project is in `~/Projects/vector-demo`:

```text
vector-demo/
  main.cpp
  kernels.cu
  include/
    types.h
```

Change into that project directory and ask MCU to inspect it:

```bash
cd ~/Projects/vector-demo
mcu check .
mcu analyze . --json
```

`check` gives human-readable diagnostics. `analyze` returns the compatibility
information as JSON, which is useful for scripts or editor integrations.

## 3. Generate Metal output

To translate the project and write generated artifacts in the current directory:

```bash
mcu build .
```

The generated archive is written as `mcu_artifacts.zip`. You can inspect the
Metal source and metadata inside it without compiling a native executable.

## 4. Build and run on Apple Silicon

If the project has a supported host entry point and CUDA kernel launches, build
the native executable into a project-local `.mcu` directory:

```bash
mcu build . --native --output .mcu/vector-demo
```

The command transforms the CUDA device code, compiles the Metal code with
Apple's toolchain, builds the native runtime and places the executable in the
requested output directory. Run it from there according to the project's
normal command-line interface.

Native builds require macOS on Apple Silicon, Xcode and the Metal Toolchain.

## 5. Use a single source file

For a small kernel or a compatibility check, pass the source file directly:

```bash
mcu check kernels.cu
mcu analyze kernels.cu --json
mcu build kernels.cu
```

A source file containing only a kernel can be translated, but native execution
also needs host code that allocates data and launches that kernel.

## 6. Run MCU from another directory

You can stay outside the project and pass an absolute path:

```bash
cd ~/Projects
mcu check "$HOME/Projects/vector-demo"
mcu build "$HOME/Projects/vector-demo" --native \
  --output "$HOME/Projects/vector-demo/.mcu/vector-demo"
```

Relative paths are resolved from the directory where you run the command.

## 7. Remove the command

The wrapper can be removed without touching the compiler checkout or any CUDA
project:

```bash
cd ~/Projects/CudaMetalTransformer
MCU_INSTALL_DIR="$HOME/.local/bin" ./scripts/uninstall_mcu.sh
```

MCU supports a tested CUDA subset. `check` and `analyze` report known limitations;
passing them does not prove that every unsupported construct has been detected
or that native compilation and execution will succeed. See [Architecture](ARCHITECTURE.md) and
[Status](STATUS.md) for the supported scope.

## Experimental automatic host-loop offload

Inspect host loops before opting into offload:

```bash
mcu analyze . --auto-parallel=report
```

For a native build, enable the proven subset explicitly:

```bash
mcu build . --native --auto-parallel=safe --output .mcu/parallel
```

The default is `--auto-parallel=off`. `--auto-parallel=report` on `build`
reports candidates while keeping the original CPU loops. `safe` requires
`--native` and enables only loops whose semantic conditions are proven; it
currently forces GPU execution for qualifying nonempty loops without choosing
based on measured profitability. Small loops can be slower than CPU execution.

The initial subset supports zero-based unsigned-int loops with unit increments,
local unsigned arrays with known sizes, and immutable pointers derived from
those arrays with constant offsets. Dynamic unsigned bounds require runtime
checks against available sizes and alias restrictions. Unknown external or
mutable pointers, general calls, reductions, searches and loops in headers are
not offloaded. A loop marked `conditional` needs its guards to pass.

The compiler preserves source files and rewrites copies in the native output.
A failed guard keeps the original CPU loop. Allocation or upload failures also
keep CPU execution before a kernel is submitted; failures after submission are
reported as exceptions and do not repeat the computation on CPU.

This option is currently supported by `mcu build`, not `mcu make` or the
NVCC-compatible compilation driver. It does not automatically parallelize general search loops.
