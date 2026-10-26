# CudaMetalTransformer

Compile a supported subset of CUDA C++ to Metal and run it on Apple Silicon.
The compiler uses Python and libclang. The runtime uses Objective-C++ and Metal.

## Requirements

macOS on Apple Silicon, Python 3.11 or later, and Xcode with the Metal Toolchain.
Use a native arm64 terminal. NVIDIA hardware and the CUDA toolkit are not required.

## Install

See [HOW_TO_INSTALL.md](HOW_TO_INSTALL.md) for setup and troubleshooting.

Open a terminal in this directory:

```bash
./install.sh --check
./install.sh
export PATH="$HOME/.local/bin:$PATH"
mcu --help
```

The installer checks the platform and compilers, creates `.venv`, installs the
Python dependencies, checks libclang, and adds `mcu` to `~/.local/bin`.
Keep this directory in place: the installed command points to it. Add the PATH
line to `~/.zshrc` to use the command in new terminals.

To select a Python interpreter or command directory:

```bash
MCU_PYTHON=python3.12 MCU_INSTALL_DIR="$HOME/.local/bin" ./install.sh
```

If the Metal compiler is unavailable, finish Xcode setup and install its toolchain:

```bash
sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer
xcodebuild -runFirstLaunch
xcodebuild -downloadComponent MetalToolchain
```

## Use

Run MCU from your CUDA project directory:

```bash
mcu check .
mcu analyze . --json
mcu build . --native --output .mcu/app
```

`check` reports source and launch constraints. It does not prove that the program
compiles or produces correct results. `analyze` returns a compatibility report.
`build --native` generates Metal and host code and builds an executable in the
output directory. Native execution requires host code that allocates buffers and
launches the kernels. Run the generated executable with your program's arguments.

You can also pass a single source file:

```bash
mcu check kernels.cu
mcu build kernels.cu
```

Without `--native`, `build` writes generated sources and metadata to
`mcu_artifacts.zip`. The source files are preserved.

For compilation through the NVCC compatibility driver:

```bash
mcu compile main.cu -o .mcu/app
./.mcu/app
```

Run `mcu capabilities --json` to inspect the supported features.
Use `MCU_CACHE_DIR` to change the compiler cache directory.

## Scope

Supported subsets include device functions, shared memory, atomics, streams,
events, warp intrinsics and CUDA graphs. A limited cuBLAS SGEMM adapter uses
Metal Performance Shaders. Arbitrary CUDA C++, PTX and general CUDA libraries
are not supported. Some lowering paths still inspect source text.

Experimental host loop offload is disabled by default. Inspect it with
`mcu analyze . --auto-parallel=report`. The `build --native --auto-parallel=safe`
option supports a narrow set of independent loops over local unsigned arrays.
It does not select CPU or GPU execution based on measured performance and may
slow down small loops. External pointers, reductions and searches are not covered.

## Source layout

`cli/` contains the command interface. `core/` contains parsing, IR, validation,
code generation and build coordination. `runtime/` contains the native runtime.
`scripts/` contains command installation and removal.

## Uninstall

```bash
./scripts/uninstall_mcu.sh
```

This removes the installed command. The source directory and its `.venv` remain.
