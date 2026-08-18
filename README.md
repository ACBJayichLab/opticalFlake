# opticalFlake

Desktop tool for optical flake thickness characterization in materials science. Analyzes optical contrast of 2D materials (graphene flakes) by capturing screenshots, defining background regions, and computing RGB contrast along line cuts.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Platform](https://img.shields.io/badge/platform-macOS%20|%20Windows%20|%20Linux-lightgrey.svg)

## Features

- **Screen Capture**: Snip-tool style selection (drag or click-to-click)
- **Background Selection**: Rectangle drag or polygon click for custom regions
- **Linecut Analysis**: Multi-segment line cuts with directional arrows
- **RGB Contrast Plots**: Per-channel contrast visualization with baseline correction
- **Multiple Measurements**: Compare multiple line cuts on the same image
- **Adjustable Width**: Averaging width parameter for noise reduction
- **Multi-Tab Support**: Work with multiple captured images simultaneously

## Quick Start (Standalone App)

Step-by-step install instructions to hand to someone else live in **[INSTALL.md](INSTALL.md)**.

### macOS
1. Download `OpticalFlake_V0.5.0` from the [releases](../../releases)
2. Move to Applications folder
3. Clear the Gatekeeper block — see below. macOS refuses to open the app until you do
4. On first run, grant **Screen Recording** permission:
   - System Settings → Privacy & Security → Screen Recording → Add OpticalFlake

#### Unblocking the app on macOS

The app is ad-hoc signed and not notarized by Apple, so a Mac that downloaded it refuses to
launch it. Depending on the macOS version you get either:

> "OpticalFlake_V0.5.0" is damaged and can't be opened. You should move it to the Trash.

> "OpticalFlake_V0.5.0" cannot be opened because Apple cannot check it for malicious software.

Neither is true — macOS says this about any app without a paid Developer ID signature. Strip
the quarantine flag macOS attached at download and it opens normally:

```bash
xattr -dr com.apple.quarantine /Applications/OpticalFlake_V0.5.0.app
open /Applications/OpticalFlake_V0.5.0.app
```

If that reports `Operation not permitted`, repeat it with `sudo`. If the app still refuses
after the quarantine flag is gone, its signature was damaged in transit — re-seal it locally:

```bash
codesign --force --deep --sign - /Applications/OpticalFlake_V0.5.0.app
codesign --verify --deep --strict /Applications/OpticalFlake_V0.5.0.app
```

Without Terminal: launch the app, dismiss the warning, then **System Settings → Privacy &
Security**, scroll to the security section, click **"Open Anyway"** and authenticate. Right-
click → Open no longer works as a bypass on macOS 15 and later. Do not disable Gatekeeper
system-wide (`spctl --master-disable`) — the per-app commands above are enough.

You unblock once per installed copy; a new version needs it again, and needs re-adding under
Screen Recording, because macOS keys both to the exact app bundle.

#### Sharing a build with someone else

Send a `.zip`, never the unpacked `.app`. The bundle is built almost entirely out of symlinks
into the Qt frameworks, and uploading it as a folder to Drive or Dropbox drops them, which
breaks the app for real. Finder's **Compress** is safe; so is:

```bash
ditto -c -k --sequesterRsrc --keepParent \
  dist/OpticalFlake_V0.5.0.app dist/OpticalFlake_V0.5.0_macOS.zip
```

The build targets Apple Silicon only. On an Intel Mac (`uname -m` prints `x86_64`) it cannot
run at all, whatever the security settings — that needs a build made on an Intel Mac.

### Windows
1. Download `OpticalFlake_V0.5.0.exe` from the [releases](../../releases)
2. Run the executable

## Development Setup

### Prerequisites
- Python 3.10+
- macOS, Windows, or Linux

### Install

```bash
# Clone the repository
git clone https://github.com/yourusername/opticalFlake.git
cd opticalFlake

# Create virtual environment
python -m venv .venv

# Activate (macOS/Linux)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Run

```bash
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
python opticalFlake.py
```

### Build Standalone Application

```bash
python build_app.py
```

This creates:
- **macOS**: `dist/OpticalFlake_V0.5.0.app` (bundle) plus a plain executable
- **Windows**: `dist/OpticalFlake_V0.5.0.exe`

The version in the bundle name comes from `APP_VERSION` in `opticalFlake.py`, which is
the only place it is written.

## Usage

1. **Capture Image**: Click "Capture Image" then drag or click-to-click to select screen region
2. **Draw Background**: Select substrate/background region (drag for rectangle, click vertices for polygon)
3. **Draw Linecut**: Click points along the flake, double-click to finish
4. **Analyze**: View RGB contrast plots with automatic baseline correction

### Tips
- Background RGB values displayed in bottom-right of image
- Adjust "Width" parameter for line averaging (reduces noise)
- Use "Fixed Y-Axis" to compare measurements across tabs
- Red/Green/Blue checkboxes toggle channel visibility

## Dependencies

| Package | Purpose |
|---------|---------|
| PySide6 | Qt GUI framework |
| matplotlib | Plotting |
| numpy | Array operations |
| pillow | Image processing |
| mss | Cross-platform screen capture |

## Project Structure

```
opticalFlake/
├── opticalFlake.py        # Main application (APP_VERSION lives here)
├── build_app.py           # PyInstaller build script
├── requirements.txt       # Python dependencies
├── tools/smoke_test.py    # Headless checks
├── Old_Versions/          # Previous versions
└── dist/                  # Built executables (after build)
```

## License

MIT License
