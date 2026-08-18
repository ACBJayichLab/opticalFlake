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
import json
import os
import sys
import tempfile
from pathlib import Path

# Qt must run offscreen before the app module creates any widgets.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO_ROOT = Path(__file__).resolve().parent.parent

failures: list[str] = []

# Modal dialogs never return without a user, so they are stubbed for the whole
# run. `warnings` records QMessageBox.warning calls; `dialog_text` is the canned
# QInputDialog.getText reply that tests set before triggering a prompt.
warnings: list[str] = []
dialog_text = {"value": ("", False)}


def stub_dialogs(app) -> None:
    """Replace every modal the app can raise with a non-blocking stand-in."""
    app.QMessageBox.warning = lambda *args, **kwargs: warnings.append(
        str(args[2]) if len(args) > 2 else ""
    )
    app.QMessageBox.information = lambda *args, **kwargs: None
    app.QMessageBox.question = lambda *a, **k: app.QMessageBox.StandardButton.Yes
    app.QInputDialog.getText = lambda *a, **k: dialog_text["value"]


def check(label: str, condition: bool, detail: str = "") -> None:
    """Record a pass/fail line."""
    if condition:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}{f' — {detail}' if detail else ''}")
        failures.append(label)


APP_SOURCE = REPO_ROOT / "opticalFlake.py"


def load_app_module():
    """Import the application source by path, so the test cannot load a stale copy."""
    if not APP_SOURCE.exists():
        raise FileNotFoundError(f"Application source not found: {APP_SOURCE}")

    source = APP_SOURCE
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


def make_pixmap(image):
    """Wrap a PIL image in a QPixmap, keeping the raw bytes alive long enough."""
    from PySide6.QtGui import QImage, QPixmap

    raw = image.tobytes("raw", "RGB")  # Reference must outlive the QImage.
    qimage = QImage(
        raw, image.width, image.height, image.width * 3, QImage.Format.Format_RGB888
    )
    return QPixmap.fromImage(qimage.copy())


def combo_materials(panel) -> list[str]:
    """Material names currently listed in a panel's dropdown (excludes actions)."""
    combo = panel.material_combo
    names = []
    for i in range(combo.count()):
        data = combo.itemData(i)
        if data and data[0] == "material":
            names.append(data[1])
    return names


def baseline_line_count(app, panel) -> int:
    """Count baseline-colored lines across every axis of the panel's figure."""
    total = 0
    for ax in panel.figure.axes:
        for line in ax.lines:
            if line.get_color() == app.BASELINE_COLOR:
                total += 1
    return total


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


def test_version(app) -> None:
    """The version is written once and everything else derives from it."""
    print("\nVersion consistency")

    version = app.APP_VERSION
    check("APP_VERSION looks like a version string", isinstance(version, str) and version.startswith("V") and version[1:].replace(".", "").isdigit(), f"got {version!r}")

    build_app = importlib.util.spec_from_file_location("build_app", REPO_ROOT / "build_app.py")
    build_module = importlib.util.module_from_spec(build_app)
    build_app.loader.exec_module(build_module)

    check("build_app targets the application source", build_module.MAIN_SCRIPT == APP_SOURCE, f"got {build_module.MAIN_SCRIPT}")
    check("build_app reads the version from the source", build_module.read_app_version() == version, f"got {build_module.read_app_version()!r}")
    check("build_app keeps no version of its own", not hasattr(build_module, "APP_VERSION"))

    # A missing source must stop the build rather than silently substituting one.
    missing = REPO_ROOT / "tools" / "does_not_exist.py"
    try:
        build_module.read_app_version(missing)
        raised = False
    except FileNotFoundError:
        raised = True
    check("a missing source raises instead of falling back", raised)

    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    check("README names the current version", version in readme, f"{version} missing from README.md")


def test_version_title(app, tmp_dir: Path) -> None:
    """The window title reports the same version, offscreen."""
    from PySide6.QtWidgets import QApplication

    qapp = QApplication.instance() or QApplication(sys.argv)
    window = app.MainWindow(config_path=tmp_dir / "version" / "settings.json")
    check("the window title carries APP_VERSION", window.windowTitle().endswith(app.APP_VERSION), f"got {window.windowTitle()!r}")
    qapp.processEvents()
    window.close()


def test_config(app, tmp_dir: Path) -> None:
    """Config load/save. Pure functions, no Qt, never touches the real user config."""
    print("\nConfig persistence")

    path = tmp_dir / "nested" / "settings.json"

    seeded = app.load_config(path)
    names = [m["name"] for m in seeded["materials"]]
    check("missing file seeds the built-in materials", names == ["NiPS3", "CrSBr", "Graphene"], f"got {names}")

    nips3 = seeded["materials"][0]
    check("NiPS3 seeds red 2.35 / green 7.4", nips3["reference_values"] == {"red": 2.35, "green": 7.4}, f"got {nips3['reference_values']}")
    check("seeded presets omit blue", "blue" not in nips3["reference_values"])
    crsbr = seeded["materials"][1]
    check("CrSBr seeds red 4.45 / green 15.5", crsbr["reference_values"] == {"red": 4.45, "green": 15.5}, f"got {crsbr['reference_values']}")
    graphene = seeded["materials"][2]
    check("Graphene seeds green 3.0 only", graphene["reference_values"] == {"green": 3.0}, f"got {graphene['reference_values']}")

    check("save_config creates missing directories", app.save_config(seeded, path) is True)
    check("round-trip preserves materials", app.load_config(path)["materials"] == seeded["materials"])

    path.write_text("{not valid json", encoding="utf-8")
    recovered = app.load_config(path)
    check("corrupt config falls back to defaults without raising", [m["name"] for m in recovered["materials"]] == ["NiPS3", "CrSBr", "Graphene"])

    path.write_text(json.dumps({"schema_version": 2, "materials": [], "last_used": {"baseline_points": 7, "bogus": 1}, "extra": True}), encoding="utf-8")
    partial = app.load_config(path)
    check("an empty material list is honored, not reseeded", partial["materials"] == [], f"got {partial['materials']}")
    check("known last_used keys survive", partial["last_used"]["baseline_points"] == 7)
    check("unknown keys are ignored", "bogus" not in partial["last_used"])
    check("missing last_used keys fall back", partial["last_used"]["yaxis_max"] == 0.18)

    path.write_text(json.dumps({"schema_version": 2, "materials": [{"name": "Bad", "reference_values": {"red": "x", "green": 3}, "layer_count": "no"}]}), encoding="utf-8")
    salvaged = app.load_config(path)["materials"][0]
    check("bad channel values are dropped", salvaged["reference_values"] == {"green": 3.0}, f"got {salvaged['reference_values']}")
    check("bad layer_count falls back to 1", salvaged["layer_count"] == 1)


def test_config_migration(app, tmp_dir: Path) -> None:
    """Schema 1 -> 2 refreshes untouched built-ins without discarding user edits."""
    print("\nConfig migration (schema 1 -> 2)")

    path = tmp_dir / "migration" / "settings.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    def write_v1(materials: list) -> None:
        path.write_text(json.dumps({"schema_version": 1, "materials": materials}), encoding="utf-8")

    v1_nips3 = {"name": "NiPS3", "reference_values": {"red": 2.25, "green": 7.5}, "layer_count": 1}
    v1_crsbr = {"name": "CrSBr", "reference_values": {"red": 4.0, "green": 15.0}, "layer_count": 1}

    write_v1([v1_nips3, v1_crsbr])
    migrated = app.load_config(path)
    by_name = {m["name"]: m for m in migrated["materials"]}
    check("untouched NiPS3 is refreshed to 2.35 / 7.4", by_name["NiPS3"]["reference_values"] == {"red": 2.35, "green": 7.4}, f"got {by_name['NiPS3']['reference_values']}")
    check("untouched CrSBr is refreshed to 4.45 / 15.5", by_name["CrSBr"]["reference_values"] == {"red": 4.45, "green": 15.5}, f"got {by_name['CrSBr']['reference_values']}")
    check("migration adds the new Graphene built-in", by_name["Graphene"]["reference_values"] == {"green": 3.0}, f"got {sorted(by_name)}")
    check("migration writes the new schema version back", json.loads(path.read_text(encoding="utf-8"))["schema_version"] == 2)
    check("migration is not re-applied on the next load", app.load_config(path)["materials"] == migrated["materials"])

    # An edited built-in keeps the user's numbers.
    write_v1([{"name": "NiPS3", "reference_values": {"red": 9.9, "green": 7.5}, "layer_count": 3}, v1_crsbr])
    edited = {m["name"]: m for m in app.load_config(path)["materials"]}
    check("an edited built-in is left alone", edited["NiPS3"]["reference_values"] == {"red": 9.9, "green": 7.5}, f"got {edited['NiPS3']['reference_values']}")
    check("an edited built-in keeps its layer count", edited["NiPS3"]["layer_count"] == 3)

    # A built-in the user deleted under schema 1 stays deleted; only new ones arrive.
    write_v1([v1_crsbr])
    names = [m["name"] for m in app.load_config(path)["materials"]]
    check("a deleted built-in is not resurrected", "NiPS3" not in names, f"got {names}")
    check("Graphene still arrives alongside the deletion", "Graphene" in names, f"got {names}")

    # A file with no schema_version at all is treated as schema 1.
    path.write_text(json.dumps({"materials": [v1_nips3]}), encoding="utf-8")
    unversioned = {m["name"]: m for m in app.load_config(path)["materials"]}
    check("a missing schema_version migrates as schema 1", unversioned["NiPS3"]["reference_values"] == {"red": 2.35, "green": 7.4}, f"got {unversioned['NiPS3']['reference_values']}")


def test_materials(app, tmp_dir: Path) -> None:
    """Material presets: apply, dirty tracking, save, and cross-tab propagation."""
    print("\nMaterial presets (offscreen)")

    from PySide6.QtWidgets import QApplication

    qapp = QApplication.instance() or QApplication(sys.argv)
    path = tmp_dir / "materials" / "settings.json"

    window = app.MainWindow(config_path=path)
    image = make_test_image(app)
    window._create_tab_from_capture(make_pixmap(image), image)
    window._create_tab_from_capture(make_pixmap(image), image)
    tab1 = window.tabs.widget(0)
    tab2 = window.tabs.widget(1)
    panel1, panel2 = tab1.data_panel, tab2.data_panel

    check("dropdown lists the seeded materials", combo_materials(panel1) == ["NiPS3", "CrSBr", "Graphene"], f"got {combo_materials(panel1)}")

    # Apply through the real activation path so the itemData plumbing is covered.
    blue_before = panel1.reference_values["blue"]
    index = combo_materials(panel1).index("NiPS3")
    panel1._on_material_activated(index)
    check("applying NiPS3 sets red to 2.35", panel1.reference_values["red"] == 2.35, f"got {panel1.reference_values['red']}")
    check("applying NiPS3 sets green to 7.4", panel1.reference_values["green"] == 7.4, f"got {panel1.reference_values['green']}")
    check("applying NiPS3 sets the marked layer", panel1.layer_count == 1)
    check("a preset that omits blue leaves blue untouched", panel1.reference_values["blue"] == blue_before, f"got {panel1.reference_values['blue']}")
    check("a freshly applied preset is not dirty", panel1._material_dirty is False)
    check("the applied preset is selected", panel1.material_combo.currentText() == "NiPS3", f"got {panel1.material_combo.currentText()!r}")

    index = combo_materials(panel1).index("CrSBr")
    panel1._on_material_activated(index)
    check("applying CrSBr sets red to 4.45", panel1.reference_values["red"] == 4.45, f"got {panel1.reference_values['red']}")
    check("applying CrSBr sets green to 15.5", panel1.reference_values["green"] == 15.5, f"got {panel1.reference_values['green']}")

    # Graphene records green only, so red must survive the switch from CrSBr.
    index = combo_materials(panel1).index("Graphene")
    panel1._on_material_activated(index)
    check("applying Graphene sets green to 3.0", panel1.reference_values["green"] == 3.0, f"got {panel1.reference_values['green']}")
    check("Graphene leaves red where it was", panel1.reference_values["red"] == 4.45, f"got {panel1.reference_values['red']}")
    check("applying Graphene sets the marked layer", panel1.layer_count == 1)

    # The green arrow step matches red's, so stepping is usable at these values.
    check("the green spinbox steps by 0.15", panel1.ref_green_spinbox.singleStep() == 0.15, f"got {panel1.ref_green_spinbox.singleStep()}")
    panel1.ref_green_spinbox.setValue(15.5)
    panel1.ref_green_spinbox.stepBy(1)
    check("one green step up moves 15.50 to 15.65", panel1.ref_green_spinbox.value() == 15.65, f"got {panel1.ref_green_spinbox.value()}")
    panel1.ref_green_spinbox.stepBy(-1)
    check("one green step down returns to 15.50", panel1.ref_green_spinbox.value() == 15.5, f"got {panel1.ref_green_spinbox.value()}")

    index = combo_materials(panel1).index("NiPS3")
    panel1._on_material_activated(index)
    panel1.ref_red_spinbox.setValue(3.0)
    check("changing a value marks the preset dirty", panel1._material_dirty is True)
    check("a dirty preset is flagged in the dropdown", panel1.material_combo.currentText() == "NiPS3 *", f"got {panel1.material_combo.currentText()!r}")
    check("tweaking does not write back to the store", app.load_config(path)["materials"][0]["reference_values"]["red"] == 2.35)

    # Save under a new name with Blue off, so blue must not be recorded.
    dialog_text["value"] = ("TestMat", True)
    panel1._save_current_material()
    saved = app.load_config(path)["materials"][-1]
    check("saving adds the preset to the store", saved["name"] == "TestMat", f"got {saved['name']}")
    check("saving records the edited value", saved["reference_values"]["red"] == 3.0)
    check("saving with Blue off omits blue", "blue" not in saved["reference_values"], f"got {saved['reference_values']}")
    check("saving clears the dirty flag", panel1._material_dirty is False)
    check("a saved preset propagates to other tabs", "TestMat" in combo_materials(panel2), f"got {combo_materials(panel2)}")

    check("renaming succeeds", window.material_store.rename("TestMat", "Renamed") is True)
    check("renaming propagates to other tabs", "Renamed" in combo_materials(panel2))
    check("renaming keeps the selection in the origin tab", panel1.material_combo.currentText() == "Renamed", f"got {panel1.material_combo.currentText()!r}")
    check("renaming onto an existing name is refused", window.material_store.rename("Renamed", "NiPS3") is False)

    check("deleting succeeds", window.material_store.delete("Renamed") is True)
    check("deleting propagates to other tabs", "Renamed" not in combo_materials(panel2))
    check("a deleted preset clears the selection", panel1._material_name is None, f"got {panel1._material_name}")
    check("deleting leaves the on-screen values alone", panel1.reference_values["red"] == 3.0)

    window.material_store.delete("NiPS3")
    check("restore_builtins re-adds only what is missing", window.material_store.restore_builtins() == 1)
    check("restored built-ins appear in every tab", "NiPS3" in combo_materials(panel2))

    qapp.processEvents()
    window.close()


def test_inheritance(app, tmp_dir: Path) -> None:
    """New tabs inherit from the active tab, and settings survive a relaunch."""
    print("\nSettings inheritance (offscreen)")

    from PySide6.QtWidgets import QApplication

    qapp = QApplication.instance() or QApplication(sys.argv)
    path = tmp_dir / "inherit" / "settings.json"

    window = app.MainWindow(config_path=path)
    image = make_test_image(app)
    window._create_tab_from_capture(make_pixmap(image), image)

    panel1 = window.tabs.widget(0).data_panel
    panel1.blue_checkbox.setChecked(True)
    panel1.green_checkbox.setChecked(False)
    panel1.baseline_spinbox.setValue(42)
    panel1.ref_baseline_checkbox.setChecked(True)
    panel1.layer_spinbox.setValue(4)
    panel1.ref_red_spinbox.setValue(1.25)

    window._create_tab_from_capture(make_pixmap(image), image)
    panel2 = window.tabs.widget(1).data_panel
    check("new tab inherits channel visibility", (panel2.show_red, panel2.show_green, panel2.show_blue) == (True, False, True), f"got {(panel2.show_red, panel2.show_green, panel2.show_blue)}")
    check("new tab inherits baseline points", panel2.baseline_points == 42, f"got {panel2.baseline_points}")
    check("new tab inherits the calc-baseline toggle", panel2.use_calculated_ref_baseline is True)
    check("new tab inherits the marked layer", panel2.layer_count == 4, f"got {panel2.layer_count}")
    check("new tab inherits reference values", panel2.reference_values["red"] == 1.25, f"got {panel2.reference_values['red']}")
    check("inherited widgets match inherited state", panel2.baseline_spinbox.value() == 42 and panel2.blue_checkbox.isChecked())

    window.width_input.setValue(33)
    window.yaxis_checkbox.setChecked(True)
    window.close()  # closeEvent persists the last-used settings

    stored = app.load_config(path)["last_used"]
    check("closing persists panel settings", stored["baseline_points"] == 42, f"got {stored['baseline_points']}")
    check("closing persists toolbar settings", stored["linecut_width"] == 33 and stored["use_fixed_yaxis"] is True)

    # Relaunch: the first tab of a fresh session starts from the saved settings.
    window2 = app.MainWindow(config_path=path)
    check("toolbar is restored on relaunch", window2.width_input.value() == 33, f"got {window2.width_input.value()}")
    window2._create_tab_from_capture(make_pixmap(image), image)
    panel3 = window2.tabs.widget(0).data_panel
    check("the first tab of a session uses the saved settings", panel3.baseline_points == 42, f"got {panel3.baseline_points}")
    check("the first tab restores the marked layer", panel3.layer_count == 4, f"got {panel3.layer_count}")

    qapp.processEvents()
    window2.close()


def test_plot_style(app, tmp_dir: Path) -> None:
    """The calculated baseline is drawn only when it carries information."""
    print("\nPlot styling (offscreen)")

    from PySide6.QtWidgets import QApplication

    qapp = QApplication.instance() or QApplication(sys.argv)
    path = tmp_dir / "style" / "settings.json"

    window = app.MainWindow(config_path=path)
    image = make_test_image(app)
    window._create_tab_from_capture(make_pixmap(image), image)
    tab = window._current_tab()

    tab.canvas.start_background_mode()
    from PySide6.QtCore import QPointF

    tab.canvas.polygon_points = [QPointF(10, 10), QPointF(190, 10), QPointF(190, 60), QPointF(10, 60)]
    tab.canvas._finalize_polygon()
    tab._on_linecut_complete([((100, 10), (100, 190))])

    panel = tab.data_panel
    panel.ref_baseline_checkbox.setChecked(False)
    check("no baseline line is drawn when Calc Baseline is off", baseline_line_count(app, panel) == 0, f"got {baseline_line_count(app, panel)}")

    panel.ref_baseline_checkbox.setChecked(True)
    visible_axes = sum([panel.show_red, panel.show_green, panel.show_blue])
    check("one baseline line per visible axis when Calc Baseline is on", baseline_line_count(app, panel) == visible_axes, f"got {baseline_line_count(app, panel)} for {visible_axes} axes")
    check("the baseline is labeled on the plot", any(t.get_text() == "baseline" for ax in panel.figure.axes for t in ax.texts))
    check("the grid is drawn behind the data", all(ax.get_axisbelow() for ax in panel.figure.axes))

    # Tick labels must carry enough precision for the tick step: over the default
    # -5% .. 18% range the locator picks 2.5-point steps, which whole-percent
    # labels used to collapse into repeated values.
    panel.set_yaxis_limits(True, -0.05, 0.18)
    panel.figure.canvas.draw()
    ax = panel.figure.axes[0]
    labels = [t.get_text() for t in ax.get_yticklabels() if t.get_text()]
    check("y ticks keep their fractional part", any("2.5%" in label for label in labels), f"got {labels}")
    check("no two y tick labels read the same", len(set(labels)) == len(labels), f"got {labels}")

    qapp.processEvents()
    window.close()


def test_widgets(app, tmp_dir: Path) -> None:
    print("\nQt widgets (offscreen)")

    from PySide6.QtWidgets import QApplication

    qapp = QApplication.instance() or QApplication(sys.argv)

    warnings.clear()

    window = app.MainWindow(config_path=tmp_dir / "widgets" / "settings.json")
    check("MainWindow constructs", window is not None)

    image = make_test_image(app)
    window._create_tab_from_capture(make_pixmap(image), image)
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
    stub_dialogs(app)

    test_pure_functions(app)
    test_version(app)
    # Every suite that builds a MainWindow gets an explicit config path, so the
    # real user settings file is never read or written by the test run.
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        test_version_title(app, tmp_dir)
        test_config(app, tmp_dir)
        test_config_migration(app, tmp_dir)
        test_widgets(app, tmp_dir)
        test_materials(app, tmp_dir)
        test_inheritance(app, tmp_dir)
        test_plot_style(app, tmp_dir)

    print()
    if failures:
        print(f"FAILED ({len(failures)}): " + ", ".join(failures))
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
