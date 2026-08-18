# opticalFlake

PySide6 desktop tool for optical flake thickness characterization in materials science.
The user captures a screen region of a 2D-material (e.g. graphene) micrograph, marks a
substrate background region, and draws linecuts; the app plots per-channel RGB contrast.

## Commands

```bash
.venv/bin/python opticalFlake.py      # run the app (GUI, needs a real display)
.venv/bin/python tools/smoke_test.py       # headless checks — run this after every edit
.venv/bin/python build_app.py              # PyInstaller bundle into dist/ — see /build-release
```

Use `.venv/bin/python` directly rather than activating; the venv is Python 3.14 while the
system default is not. There is no pytest suite, linter, or formatter configured.

## Verifying changes

`tools/smoke_test.py` runs the contrast math against synthetic images with known answers,
then drives the widgets offscreen (`QT_QPA_PLATFORM=offscreen`): capture → background →
linecut → plot → width edit → removal. It exits non-zero on failure. **Run it before
saying an edit works**, and extend it when you add behavior it doesn't cover.

It cannot cover screen capture (needs a display plus macOS Screen Recording permission) or
anything about how the drawing previews look. For those, launch the app and say plainly
that the check was visual — or ask the user to confirm.

## Architecture

Single file, `opticalFlake.py` (~2400 lines), divided by `# ===` banner comments:
Data Models → Calculation Functions → Screen Capture Overlay → Image Canvas → Data Display
Panel → Image Tab → Main Window → Entry Point.

`MainWindow` owns a `QTabWidget` of `ImageTab`s. Each `ImageTab` holds an `ImageCanvas`
(left, a `QGraphicsView` for drawing) and a `DataDisplayPanel` (right, matplotlib +
measurement list), and is the only place the two talk to each other.

Components communicate by Qt signal, never by reaching across into another widget's
methods: `ImageCanvas.polygon_complete` / `linecut_complete` / `invalid_action` /
`drawing_mode_changed`, `DataDisplayPanel.measurement_removed` / `width_change_requested` /
`baseline_points_changed`. Keep new cross-component wiring on signals.

State lives in the `Measurement` and `ImageData` dataclasses. No module-level mutable state.

## Gotchas

**`matplotlib.use("QtAgg")` must stay above the PySide6 imports.** Moving it below them
segfaults on macOS. The import block order at the top of the file is load-bearing.

**QImage does not own its pixel buffer.** Every conversion from PIL must keep a Python
reference to the raw bytes alive *and* `.copy()` the QImage before wrapping it in a
QPixmap, or the pixmap points at freed memory and crashes later:

```python
self._img_data = pil_image.tobytes("raw", "RGB")     # reference must outlive the QImage
qimage = QImage(self._img_data, w, h, w * 3, QImage.Format.Format_RGB888)
pixmap = QPixmap.fromImage(qimage.copy())            # copy before the bytes can go away
```

**macOS capture takes a completely different path.** `MainWindow._start_capture` branches
on `sys.platform == "darwin"` and shells out to the native `screencapture -i -x` picker;
`ScreenCaptureOverlay` (the mss-based snip overlay) only runs on Windows and Linux. A
change to the overlay is untested on macOS and vice versa — say which platform you tested.
The overlay additionally falls back to `screencapture -x` when mss returns an all-black
frame, which macOS does in some permission states.

**Contrast is stored as a fraction and displayed as a percent, and the two sets of
spinboxes disagree.** `Measurement.*_contrast` and the toolbar Y-axis Min/Max inputs are
fractions that get multiplied by 100 at plot time; the reference-line spinboxes in
`DataDisplayPanel` hold percent already. Real contrast is typically under 10%, so a unit
mistake still produces a plausible-looking plot — check which side of the ×100 you're on.

**`baseline_points` does not affect `calculate_contrast`.** It is accepted as a parameter,
but the subtraction is commented out at the end of that function, so raw contrast is
returned. Baseline only matters for reference lines, via
`DataDisplayPanel._get_reference_baseline`, and only when "Calc Baseline" is on. The
README's "automatic baseline correction" is stale. Don't re-enable the commented block
without asking — it changes every previously recorded number.

**A background region is required before a linecut.** `ImageCanvas.start_linecut_mode`
returns `False` and emits `invalid_action` when `has_background` is unset. That flag is set
in `_finalize_polygon`, not in the `ImageTab` handler, so setting the background by calling
the tab handler directly leaves the canvas out of sync.

**Measurements are addressed by list index in two places at once.** `DataDisplayPanel.
measurements` and `ImageCanvas.persistent_linecut_items` must stay parallel — removing a
measurement renames and reindexes the rest. Any new operation on a measurement has to
update both sides.

**Modal dialogs hang headless runs.** `QMessageBox.warning` in `ImageTab._on_invalid_action`
blocks forever without a user, which is why the smoke test stubs it. Prefer the existing
`invalid_action` signal over adding new modals.

**The version is written in exactly one place.** `APP_VERSION` near the top of
`opticalFlake.py` drives the window title; `build_app.py` parses that literal out of the
source (`read_app_version`) to name the bundle, and has no copy of its own. The source
filename carries no version, and the build hard-fails if `opticalFlake.py` is missing
rather than substituting another script. `tools/smoke_test.py::test_version` fails if the
app, the build script and the README disagree, so a bump is one line plus the README
download names — see the `build-release` skill.

## Conventions

- Keep the single-file structure and the `# ===` section banners; put new code in the
  section it belongs to rather than at the end of the file.
- Calculation functions stay pure (PIL/numpy in, numpy out) and free of Qt imports, so the
  smoke test can exercise them without a display.
- Draw with `QGraphicsScene` items, not `paintEvent` overrides — except in
  `ScreenCaptureOverlay`, which is a plain `QWidget` on purpose.
- Docstrings with `Args:` / `Returns:` on functions; type hints on signatures.
- `Old_Versions/` is a frozen archive. Never edit it.
- Dependencies track current major versions (Python 3.14, PySide6 6.10, numpy 2.x,
  pillow 12). Check current APIs rather than assuming older ones when adding library code.

## Repository etiquette

- `build/`, `dist/`, `.venv/`, and `*.spec` are gitignored build output. They exist in the
  working tree but are untracked — leave them alone unless asked.
- Work on `main`. Only commit when asked.

<!-- Maintainer note: gotchas above are the reason this file exists. Prune anything that
     becomes derivable from the code; keep this file under ~200 lines. -->
