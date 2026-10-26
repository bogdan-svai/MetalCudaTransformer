# Testing

Install the development dependencies in the project environment:

```bash
./.venv/bin/python -m pip install -r requirements-dev.txt
source .venv/bin/activate
make lint
python -m pytest -q
make verify
```

`make lint` checks formatting and import order in the configured compiler modules.
`make lint-full` checks formatting and import order across all Python files. `make verify` checks the Metal toolchain before running the
suite, generated-code compilation, native examples and CUDA compatibility cases.
It requires an Apple Silicon Mac with Xcode and usable GPU access.

## Checks without a native GPU build

With Python bindings and native libclang available:

```bash
python -m pytest -q tests/test_cli.py tests/test_build.py \
  tests/test_compiler_correctness.py tests/test_device_rewriting.py \
  tests/test_clang_source_ranges.py tests/test_semantic_call_edits.py
```

These cover source analysis, source generation, semantic caching and device-call
rewrites. They do not substitute for native execution. Keep the complete suite
unchanged when the local environment cannot run GPU tests; report the failures.

## Native prerequisites

`make verify` checks the Metal compiler before running native builds and GPU
result checks. If `metal` is unavailable, finish Xcode setup and install its Metal
Toolchain as described in [installation](../HOW_TO_INSTALL.md). Source-only tests
do not replace checks of the generated executable or its GPU results.

## CI

The workflow uses the `macos-15` label listed in [GitHub's runner documentation](https://docs.github.com/en/actions/how-tos/write-workflows/choose-where-workflows-run/choose-the-runner-for-a-job).
It runs lint and focused compiler regressions on macOS. A manually
requested native job attempts the full gate after checking the Metal compiler.
Hosted macOS runners may lack a usable Metal device even when toolchain checks
pass. The native job fails when its prerequisites or tests fail.
