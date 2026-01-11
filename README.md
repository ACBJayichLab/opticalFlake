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

### macOS
1. Download `OpticalFlake` from the [releases](../../releases)
2. Move to Applications folder
3. On first run, grant **Screen Recording** permission:
   - System Settings → Privacy & Security → Screen Recording → Add OpticalFlake

### Windows
1. Download `OpticalFlake.exe` from the [releases](../../releases)
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
python opticalFlake_V0.3.py
```

### Build Standalone Application

```bash
python build_app.py
```

This creates:
- **macOS**: `dist/OpticalFlake` (executable)
- **Windows**: `dist/OpticalFlake.exe`

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
├── opticalFlake_V0.3.py   # Main application
├── build_app.py           # PyInstaller build script
├── requirements.txt       # Python dependencies
├── Old_Versions/          # Previous versions
└── dist/                  # Built executables (after build)
```

## License

MIT License
