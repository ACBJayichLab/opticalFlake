#!/usr/bin/env python3
"""
Build script to package opticalFlake into a standalone application.

Usage:
    python build_app.py

This creates a standalone .app bundle on macOS or .exe on Windows.
Requires PyInstaller: pip install pyinstaller
"""

import ast
import platform
import plistlib
import re
import subprocess
import sys
from pathlib import Path

# The source file no longer carries the version in its name, so this constant
# never has to change. The version itself is read out of the script below.
REPO_ROOT = Path(__file__).resolve().parent
MAIN_SCRIPT = REPO_ROOT / "opticalFlake.py"

_VERSION_PATTERN = re.compile(r"^APP_VERSION\s*=\s*(.+)$", re.MULTILINE)


def read_app_version(script: Path = MAIN_SCRIPT) -> str:
    """Read APP_VERSION out of the application source.

    Parsed as text rather than imported, so building never pulls in PySide6 or
    opens a Qt connection.

    Args:
        script: Path to the application source file.

    Returns:
        The version string, e.g. "V0.5.0".

    Raises:
        FileNotFoundError: The source file is missing.
        ValueError: The file has no readable APP_VERSION assignment.
    """
    if not script.exists():
        raise FileNotFoundError(
            f"Application source not found: {script}\n"
            "build_app.py builds exactly this file; it will not substitute another."
        )

    match = _VERSION_PATTERN.search(script.read_text(encoding="utf-8"))
    if match is None:
        raise ValueError(f"No APP_VERSION assignment found in {script.name}")
    try:
        version = ast.literal_eval(match.group(1).strip())
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"APP_VERSION in {script.name} is not a literal: {exc}") from exc
    if not isinstance(version, str) or not version.strip():
        raise ValueError(f"APP_VERSION in {script.name} is not a non-empty string")
    return version.strip()


def resolve_main_script() -> Path:
    """Return the source script for PyInstaller, or raise if it is missing.

    Deliberately has no fallback: silently building a different file than the
    one asked for is how the bundle and the source used to drift apart.

    Returns:
        Path to the application source.

    Raises:
        FileNotFoundError: The source file is missing.
    """
    if not MAIN_SCRIPT.exists():
        raise FileNotFoundError(f"Application source not found: {MAIN_SCRIPT}")
    return MAIN_SCRIPT


def stamp_bundle_version(app_path: Path, app_version: str) -> None:
    """Write the app version into a macOS bundle's Info.plist and re-sign it.

    PyInstaller has no CLI option for CFBundleShortVersionString, so a bundle
    otherwise reports 0.0.0 in Finder and there is no way to tell two builds
    apart once they are copied out of dist/. Editing Info.plist breaks the
    ad-hoc signature PyInstaller applied, so the bundle is re-sealed afterwards.

    Args:
        app_path: Path to the .app bundle.
        app_version: Version string from the source, e.g. "V0.5.0".

    Raises:
        subprocess.CalledProcessError: Re-signing or verification failed, which
            would leave a bundle macOS refuses to launch.
    """
    plist_path = app_path / "Contents" / "Info.plist"
    if not plist_path.exists():
        print(f"Warning: {plist_path} not found; version not stamped.")
        return

    # CFBundleShortVersionString must be numeric, so the "V" prefix is dropped.
    numeric_version = app_version.lstrip("Vv")

    with open(plist_path, "rb") as handle:
        plist = plistlib.load(handle)
    plist["CFBundleShortVersionString"] = numeric_version
    plist["CFBundleVersion"] = numeric_version
    with open(plist_path, "wb") as handle:
        plistlib.dump(plist, handle)

    # Only the outer seal is stale; the nested binaries keep their signatures.
    subprocess.run(
        ["codesign", "--force", "--sign", "-", str(app_path)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["codesign", "--verify", "--strict", str(app_path)],
        check=True,
        capture_output=True,
    )
    print(f"✓ Bundle stamped as version {numeric_version} and re-signed")


def check_pyinstaller():
    """Check if PyInstaller is installed, install if not."""
    try:
        import PyInstaller

        print(f"✓ PyInstaller {PyInstaller.__version__} found")
    except ImportError:
        print("PyInstaller not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
        print("✓ PyInstaller installed")


def build():
    """Build the application."""
    check_pyinstaller()

    # The bundle name follows the version recorded in the source, so the two can
    # never disagree about what was shipped.
    source_script = resolve_main_script()
    app_version = read_app_version(source_script)
    app_name = f"OpticalFlake_{app_version}"

    # Determine platform-specific settings
    system = platform.system()

    # Base PyInstaller command
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        f"--name={app_name}",
        "--windowed",  # No console window
        "--onedir",  # Single executable
        "--clean",  # Clean build
        # Replace this version's previous output instead of prompting, so
        # rebuilding the same version twice does not fail halfway through.
        "--noconfirm",
        # Hidden imports needed for PySide6/matplotlib
        "--hidden-import=PIL._tkinter_finder",
        "--hidden-import=matplotlib.backends.backend_qtagg",
        "--collect-submodules=matplotlib",
        "--collect-submodules=PIL",
        "--collect-submodules=mss",
    ]

    # Platform-specific options
    if system == "Darwin":  # macOS
        cmd.extend(
            [
                "--osx-bundle-identifier=com.opticalflake.app",
                # Request screen recording permission description
                "--osx-entitlements-file=entitlements.plist"
                if Path("entitlements.plist").exists()
                else "",
            ]
        )
        # Remove empty strings
        cmd = [c for c in cmd if c]

    elif system == "Windows":
        # Add Windows icon if available
        icon_path = Path("icon.ico")
        if icon_path.exists():
            cmd.append(f"--icon={icon_path}")

    # Add the main script
    cmd.append(str(source_script))

    print(f"\n{'=' * 60}")
    print(f"Building {app_name} for {system}")
    print(f"{'=' * 60}\n")
    print(f"Source: {source_script.name} (APP_VERSION = {app_version})")
    print("Command:", " ".join(cmd))
    print()

    # Run PyInstaller
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print(f"\n{'=' * 60}")
        print("✓ Build successful!")
        print(f"{'=' * 60}")

        dist_path = Path("dist")
        if system == "Darwin":
            app_path = dist_path / f"{app_name}.app"
            exe_path = dist_path / app_name
            if app_path.exists():
                stamp_bundle_version(app_path, app_version)
                print(f"\nApplication bundle: {app_path.absolute()}")
            elif exe_path.exists():
                print(f"\nExecutable: {exe_path.absolute()}")
        elif system == "Windows":
            print(f"\nExecutable: {dist_path / f'{app_name}.exe'}")
        else:
            print(f"\nExecutable: {dist_path / app_name}")

        print("\nNote: On macOS, the app needs Screen Recording permission.")
        print("Go to System Preferences > Privacy & Security > Screen Recording")
        print("and add the application.")
    else:
        print(f"\n✗ Build failed with code {result.returncode}")
        sys.exit(1)


if __name__ == "__main__":
    build()
