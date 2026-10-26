# Contributing

Use Python 3.11 or later. Install `requirements-dev.txt` in a virtual environment
and read [architecture](docs/ARCHITECTURE.md) and [testing](docs/TESTING.md).
Native changes need an Apple Silicon Mac with Xcode and the Metal Toolchain.

A compiler change should include the smallest source that demonstrates the
behavior, the expected diagnostic or generated code, and a regression test.
Preserve source locations, supported APIs and cache invalidation. Diagnose cases
that cannot be translated safely. Keep unrelated refactors separate.

Run `make lint` and the relevant tests. Run `make verify` for changes to lowering,
linking, native builds or runtime behavior. Record missing prerequisites and
failed checks rather than calling them passing or removing their assertions.

For performance work, use comparable inputs and environments. Record warmups,
samples, cache state, hardware, toolchain and correctness checks. Separate
frontend, native compilation, GPU time and process startup measurements.

Do not submit credentials, build output, internal notes or local environment files.
The project has no selected license or contributor agreement. Changes to licensing
or the inclusion of third-party code require the owner's review of provenance
and redistribution rights.
