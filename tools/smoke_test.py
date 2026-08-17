#!/usr/bin/env python3
"""
Headless smoke test for opticalFlake.

Runs the pure contrast math against synthetic images with known answers, then
drives the Qt widgets offscreen (capture -> background -> linecut -> plot) so a
GUI regression fails here instead of only being visible on screen.

Usage:
    .venv/bin/python tools/smoke_test.py

Exits 0 when every check passes, 1 otherwise. Screen capture is NOT covered:
it needs a real display and macOS Screen Recording permission, so test it by
running the app.
"""

import importlib.util
import os
import sys
from pathlib import Path

# Qt must run offscreen before the app module creates any widgets.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO_ROOT = Path(__file__).resolve().parent.parent

failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    """Record a pass/fail line."""
    if condition:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}{f' — {detail}' if detail else ''}")
        failures.append(label)


def load_app_module():
    """Import the newest opticalFlake_V*.py by path (the '.' in the name blocks a plain import)."""
    candidates = sorted(REPO_ROOT.glob("opticalFlake_V*.py"))
    if not candidates:
        raise FileNotFoundError("No opticalFlake_V*.py found in the repository root")

    def version_key(path: Path) -> tuple:
        parts = []
        for piece in path.stem.removeprefix("opticalFlake_V").split("."):
            parts.append(int(piece) if piece.isdigit() else 0)
        return tuple(parts)

    source = max(candidates, key=version_key)
    spec = importlib.util.spec_from_file_location("optical_flake_app", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    print(f"Loaded {source.name}\n")
    return module


def make_test_image(app):
    """200x200 substrate of RGB 200 with a darker RGB 100 band across rows 80-120."""
    image = app.Image.new("RGB", (200, 200), (200, 200, 200))
    for y in range(80, 120):
        for x in range(200):
            image.putpixel((x, y), (100, 100, 100))
    return image


def test_pure_functions(app) -> None:
    print("Contrast math")

    coords = app.get_line_coordinates(0, 0, 4, 0)
    check("get_line_coordinates spans both endpoints", coords[0] == (0, 0) and coords[-1] == (4, 0))
    check("get_line_coordinates is gap-free", len(coords) == 5, f"got {len(coords)}")

    offset = app.offset_parallel_line(0, 0, 10, 0, 3)
    check("offset_parallel_line shifts perpendicular", offset == (0, 3, 10, 3), f"got {offset}")

    image = make_test_image(app)
    mask = app.create_polygon_mask(image.size, [(0, 0), (20, 0), (20, 20), (0, 20)])
    avg = app.calculate_average_color(image, mask)
    check("calculate_average_color reads the substrate", avg == (200, 200, 200), f"got {avg}")

    # Vertical cut through the band: contrast is (200-100)/(200+100) = 1/3 inside, 0 outside.
    red, green, blue = app.calculate_contrast(
        image, [((100, 10), (100, 190))], (200, 200, 200), width=5
    )
    check("calculate_contrast returns per-pixel samples", len(red) > 100, f"got {len(red)}")
    check("contrast peaks at the expected 1/3", abs(float(red.max()) - 1 / 3) < 0.01, f"got {float(red.max()):.4f}")
    check("contrast is ~0 off the flake", abs(float(red.min())) < 0.01, f"got {float(red.min()):.4f}")
    check("all three channels agree on a gray target", len(red) == len(green) == len(blue))


def test_widgets(app) -> None:
    print("\nQt widgets (offscreen)")

    from PySide6.QtGui import QImage, QPixmap
    from PySide6.QtWidgets import QApplication

    qapp = QApplication.instance() or QApplication(sys.argv)

    # ImageTab._on_invalid_action opens a modal QMessageBox, which never returns
    # without a user. Record the calls instead of showing them.
    warnings: list[str] = []
    app.QMessageBox.warning = lambda *args, **kwargs: warnings.append(str(args[2]) if len(args) > 2 else "")

    window = app.MainWindow()
    check("MainWindow constructs", window is not None)

    image = make_test_image(app)
    raw = image.tobytes("raw", "RGB")  # Reference must outlive the QImage.
    qimage = QImage(raw, image.width, image.height, image.width * 3, QImage.Format.Format_RGB888)
    pixmap = QPixmap.fromImage(qimage.copy())

    window._create_tab_from_capture(pixmap, image)
    check("capture opens a tab", window.tabs.count() == 1, f"got {window.tabs.count()}")

    tab = window._current_tab()
    check("linecut is blocked before a background exists", tab.canvas.start_linecut_mode() is False)
    check("blocked linecut warns the user", len(warnings) == 1, f"got {len(warnings)} warnings")

    # Finalize through the canvas so the real polygon_complete -> ImageTab wiring runs.
    from PySide6.QtCore import QPointF

    tab.canvas.start_background_mode()
    tab.canvas.polygon_points = [QPointF(10, 10), QPointF(190, 10), QPointF(190, 60), QPointF(10, 60)]
    tab.canvas._finalize_polygon()
    check("background RGB is measured", tab.data.background_rgb == (200, 200, 200), f"got {tab.data.background_rgb}")
    check("linecut is allowed once a background exists", tab.canvas.start_linecut_mode() is True)

    tab._on_linecut_complete([((100, 10), (100, 190))])
    check("linecut adds a measurement", len(tab.data_panel.measurements) == 1)

    measurement = tab.data_panel.measurements[0]
    check("measurement stores contrast as a fraction", abs(float(measurement.red_contrast.max()) - 1 / 3) < 0.01)
    check("measurement records the width used", measurement.width == tab.canvas.averaging_width)

    original_width = measurement.width
    tab._on_width_change_requested(0, original_width + 6)
    check("width edit recalculates in place", tab.data_panel.measurements[0].width == original_width + 6)

    tab.data_panel.remove_measurement(0)
    check("removal clears the measurement", len(tab.data_panel.measurements) == 0)

    tab.data_panel.set_yaxis_limits(True, -0.18, 0.05)
    check("fixed y-axis redraws without error", True)

    qapp.processEvents()
    window.close()


def main() -> int:
    app = load_app_module()
    test_pure_functions(app)
    test_widgets(app)

    print()
    if failures:
        print(f"FAILED ({len(failures)}): " + ", ".join(failures))
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
