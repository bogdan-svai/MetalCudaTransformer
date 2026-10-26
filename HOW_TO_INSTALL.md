# How to install

MCU requires a Mac with Apple Silicon, Python 3.11 or later, and Xcode.
The CUDA toolkit is not required.

## Prepare Xcode

Install Xcode from the Mac App Store. Open it once and complete the setup.
Then select its developer tools and install the Metal compiler:

```bash
sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer
xcodebuild -runFirstLaunch
xcodebuild -downloadComponent MetalToolchain
xcrun -sdk macosx metal --version
```

If Xcode is installed elsewhere, use its actual path in the first command.

## Install MCU

Download or clone the source and open a terminal in its directory.
Check your Python version, then run the installer:

```bash
python3 --version
./install.sh --check
./install.sh
export PATH="$HOME/.local/bin:$PATH"
mcu --help
```

`--check` checks the platform, Python version and compilers without installing
anything. The installer creates `.venv`, installs the dependencies listed in
`requirements.txt`, verifies libclang, and creates `~/.local/bin/mcu`.
Installing dependencies requires internet access.

Keep the source directory in place. The installed command points to it.
If you move the directory, run the installer again from the new location.
Add this line to `~/.zshrc` to make the command available in new terminals:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

To choose a different Python interpreter or command directory:

```bash
MCU_PYTHON=python3.12 MCU_INSTALL_DIR="$HOME/.local/bin" ./install.sh
```

## Check the installation

```bash
mcu --help
mcu capabilities --json
```

These commands check that MCU starts. To check your CUDA project and build it:

```bash
cd /path/to/your/cuda-project
mcu check .
mcu build . --native --output .mcu/app
```

Run the executable created in `.mcu/app` with your program's arguments.
See [README.md](README.md) for source generation and compilation commands.

## Common problems

If `mcu` is not found, check that `~/.local/bin` is on PATH. You can also run
`~/.local/bin/mcu --help` directly.

If the Metal compiler is missing, complete Xcode setup and repeat
`xcodebuild -downloadComponent MetalToolchain`.

If Python is too old, install Python 3.11 or later and select it with
`MCU_PYTHON`. An existing `.venv` must also use a supported Python version.

If libclang cannot load, check the selected Xcode installation. MCU searches
the standard Xcode and Command Line Tools locations. Python bindings are
installed automatically, but the native library must come from the developer tools.

## Remove MCU

From the source directory:

```bash
./scripts/uninstall_mcu.sh
```

For a custom command directory, pass the same `MCU_INSTALL_DIR` used to install.
The source files, Python environment and compiler cache remain on disk.
