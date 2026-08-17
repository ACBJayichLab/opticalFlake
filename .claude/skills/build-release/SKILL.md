---
name: build-release
description: Bump the version and build the standalone OpticalFlake app with PyInstaller. Use when asked to build, package, release, or produce a new .app/.exe, or to cut a new version number.
---

# Build a release

## Before building

Run `.venv/bin/python tools/smoke_test.py`. A broken build is slow to discover; a broken
smoke test is not. Do not build on a red smoke test unless the user says to.

## Bumping the version

The version string is duplicated in four places and they drift. When bumping, change all of
them in the same edit:

1. `build_app.py` → `APP_VERSION = "V0.4.2"` (drives the bundle name and `MAIN_SCRIPT`)
2. `opticalFlake_V0.4.py` → `MainWindow.__init__` window title
3. `README.md` → the download and run instructions
4. The source filename itself, if you are renaming it (`git mv`, don't copy — the repo
   already carries one abandoned rename)

`build_app.py` looks for `opticalFlake_{APP_VERSION}.py`, and when that file is missing it
falls back to the highest-numbered `opticalFlake_V*.py` and prints a warning. That warning
is the signal that items 1 and 4 disagree. Read the build output rather than assuming it
picked the file you meant.

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
