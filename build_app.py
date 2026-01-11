#!/usr/bin/env python3
"""
Build script to package opticalFlake into a standalone application.

Usage:
    python build_app.py

This creates a standalone .app bundle on macOS or .exe on Windows.
Requires PyInstaller: pip install pyinstaller
"""

import subprocess
import sys
import platform
from pathlib import Path


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
        sys.executable, "-m", "PyInstaller",
        "--name=OpticalFlake",
        "--windowed",  # No console window
        "--onefile",   # Single executable
        "--clean",     # Clean build
        # Hidden imports needed for PySide6/matplotlib
        "--hidden-import=PIL._tkinter_finder",
        "--hidden-import=matplotlib.backends.backend_qtagg",
        "--collect-submodules=matplotlib",
        "--collect-submodules=PIL",
        "--collect-submodules=mss",
    ]
    
    # Platform-specific options
    if system == "Darwin":  # macOS
        cmd.extend([
            "--osx-bundle-identifier=com.opticalflake.app",
            # Request screen recording permission description
            "--osx-entitlements-file=entitlements.plist" if Path("entitlements.plist").exists() else "",
        ])
        # Remove empty strings
        cmd = [c for c in cmd if c]
        
    elif system == "Windows":
        # Add Windows icon if available
        icon_path = Path("icon.ico")
        if icon_path.exists():
            cmd.append(f"--icon={icon_path}")
    
    # Add the main script
    cmd.append("opticalFlake_V0.3.py")
    
    print(f"\n{'='*60}")
    print(f"Building OpticalFlake for {system}")
    print(f"{'='*60}\n")
    print("Command:", " ".join(cmd))
    print()
    
    # Run PyInstaller
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print(f"\n{'='*60}")
        print("✓ Build successful!")
        print(f"{'='*60}")
        
        dist_path = Path("dist")
        if system == "Darwin":
            app_path = dist_path / "OpticalFlake.app"
            exe_path = dist_path / "OpticalFlake"
            if app_path.exists():
                print(f"\nApplication bundle: {app_path.absolute()}")
            elif exe_path.exists():
                print(f"\nExecutable: {exe_path.absolute()}")
        elif system == "Windows":
            print(f"\nExecutable: {dist_path / 'OpticalFlake.exe'}")
        else:
            print(f"\nExecutable: {dist_path / 'OpticalFlake'}")
            
        print("\nNote: On macOS, the app needs Screen Recording permission.")
        print("Go to System Preferences > Privacy & Security > Screen Recording")
        print("and add the application.")
    else:
        print(f"\n✗ Build failed with code {result.returncode}")
        sys.exit(1)


if __name__ == "__main__":
    build()
