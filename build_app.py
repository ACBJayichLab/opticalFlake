#!/usr/bin/env python3
"""
Build script to package opticalFlake into a standalone application.

Usage:
    python build_app.py

This creates a standalone .app bundle on macOS or .exe on Windows.
Requires PyInstaller: pip install pyinstaller
"""

import platform
import subprocess
import sys
from pathlib import Path

APP_VERSION = "V0.4.2"
APP_NAME = f"OpticalFlake_{APP_VERSION}"
MAIN_SCRIPT = f"opticalFlake_{APP_VERSION}.py"


def _parse_version_tuple(version_text: str) -> tuple:
    """Parse version text like '0.4.1' into a sortable tuple."""
    parts = []
    for piece in version_text.split("."):
        try:
            parts.append(int(piece))
        except ValueError:
            parts.append(0)
    return tuple(parts)


def resolve_main_script() -> Path:
    """Resolve the source script for PyInstaller."""
    requested = Path(MAIN_SCRIPT)
    if requested.exists():
        return requested

    candidates = list(Path(".").glob("opticalFlake_V*.py"))
    if not candidates:
        raise FileNotFoundError(
            f"No source script found. Expected {MAIN_SCRIPT} or any opticalFlake_V*.py"
        )

    def version_key(path: Path) -> tuple:
        version_text = path.stem.removeprefix("opticalFlake_V")
        return _parse_version_tuple(version_text)

    fallback = max(candidates, key=version_key)
    print(
        f"Warning: {MAIN_SCRIPT} not found. Using source script {fallback.name} instead."
    )
    return fallback


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

    # Determine platform-specific settings
    system = platform.system()

    # Base PyInstaller command
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        f"--name={APP_NAME}",
        "--windowed",  # No console window
        "--onedir",  # Single executable
        "--clean",  # Clean build
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
    source_script = resolve_main_script()
    cmd.append(str(source_script))

    print(f"\n{'=' * 60}")
    print(f"Building {APP_NAME} for {system}")
    print(f"{'=' * 60}\n")
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
            app_path = dist_path / f"{APP_NAME}.app"
            exe_path = dist_path / APP_NAME
            if app_path.exists():
                print(f"\nApplication bundle: {app_path.absolute()}")
            elif exe_path.exists():
                print(f"\nExecutable: {exe_path.absolute()}")
        elif system == "Windows":
            print(f"\nExecutable: {dist_path / f'{APP_NAME}.exe'}")
        else:
            print(f"\nExecutable: {dist_path / APP_NAME}")

        print("\nNote: On macOS, the app needs Screen Recording permission.")
        print("Go to System Preferences > Privacy & Security > Screen Recording")
        print("and add the application.")
    else:
        print(f"\n✗ Build failed with code {result.returncode}")
        sys.exit(1)


if __name__ == "__main__":
    build()
