---
name: build-release
description: Bump the version and build the standalone OpticalFlake app with PyInstaller. Use when asked to build, package, release, or produce a new .app/.exe, or to cut a new version number.
---

# Build a release

## Before building

Run `.venv/bin/python tools/smoke_test.py`. A broken build is slow to discover; a broken
smoke test is not. Do not build on a red smoke test unless the user says to.

## Bumping the version

The version lives in one place: `APP_VERSION` near the top of `opticalFlake.py`. Editing
that line is the bump. The window title interpolates it, and `build_app.py` parses the
literal out of the source (`read_app_version`) to name the bundle, so the two cannot drift.

Two things do not follow automatically:

1. `README.md` → the `OpticalFlake_V<version>` download names under Quick Start and the
   build output paths. `tools/smoke_test.py::test_version` fails when the README no longer
   mentions the current version, so a forgotten update shows up as a red check.
2. The source filename is deliberately version-free (`opticalFlake.py`). Do not put the
   version back into it — that is what used to drift.

`build_app.py` builds exactly `opticalFlake.py` and raises `FileNotFoundError` if it is
missing; it never substitutes another script. Read the `Source:` line it prints to confirm
the version it picked up.

## Building

```bash
.venv/bin/python build_app.py
```

PyInstaller does not cross-compile: build on the OS you are targeting. macOS produces
`dist/OpticalFlake_<version>.app` plus a plain executable, Windows a `.exe`, both `--onedir`
and `--windowed` (no console, so runtime errors are invisible — check the smoke test first).

Each run drops a fresh `.spec` in the repo root and a tree under `build/` and `dist/`. All
three are gitignored and each build is a few hundred MB; leave the old ones alone unless the
user asks you to clean up.

## After building

Report the output path and remind the user of the macOS permission step: the bundle needs
**System Settings → Privacy & Security → Screen Recording** before capture works, and the
permission is tied to the specific bundle, so a rebuilt or renamed app must be re-added.

Do not claim the app runs correctly unless you launched it. The build succeeding only means
PyInstaller resolved the imports.
