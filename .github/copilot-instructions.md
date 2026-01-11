# opticalFlake

Desktop tool for optical flake thickness characterization in materials science. Analyzes optical contrast of 2D materials by capturing screenshots, defining background regions, and computing RGB contrast along line cuts.

# Agent Instructions
Your information is out of date. You must use the internet to research and find the latest information and versions of any dependencies mentioned in these instructions. You must ensure that any code you write is compatible with the latest versions of these dependencies.

## Quick Start
```zsh
source .venv/bin/activate && python opticalFlake_V0.3.py
```

## Architecture Overview

**Single-file application** (`opticalFlake_V0.3.py`) with clear section markers:
- **Data Models** (L42-73): `@dataclass` types for `Measurement` and `ImageData`
- **Calculation Functions** (L78-230): Pure functions for contrast math
- **UI Components** (L300+): `ImageCanvas`, `DataDisplayPanel`, `ImageTab`, `MainWindow`

### Core Formula
```python
contrast = (sample_RGB - background_RGB) / background_RGB  # per-pixel along linecut
```

### Key Algorithms (don't modify without understanding)
- `get_line_coordinates()`: Bresenham's line rasterization for pixel sampling
- `offset_parallel_line()`: Creates parallel lines for averaging width
- `create_polygon_mask()` + `calculate_average_color()`: Background region averaging

## Critical Patterns

### PySide6 + Matplotlib Integration
**MUST set matplotlib backend before PySide6 imports** (prevents macOS segfaults):
```python
import matplotlib
matplotlib.use('QtAgg')  # Line 17 - DO NOT MOVE
# ... then import PySide6
```

### QImage Memory Management
Always `.copy()` QImage data or keep references to prevent garbage collection crashes:
```python
# In ScreenCaptureOverlay._capture_screen()
self._img_data = self.pil_screenshot.tobytes("raw", "RGB")  # Keep reference
qimage = QImage(self._img_data, ...)
self.screenshot = QPixmap.fromImage(qimage.copy())  # Copy to own data
```

### Signal-Based Coordination
Components communicate via Qt signals, not direct method calls:
- `ImageCanvas.polygon_complete` → `ImageTab._on_polygon_complete`
- `ImageCanvas.linecut_complete` → `ImageTab._on_linecut_complete`
- `ImageCanvas.drawing_mode_changed` → `MainWindow._on_drawing_mode_changed`

### Drawing Mode State Machine
`ImageCanvas.drawing_mode` controls interaction: `None` | `'background'` | `'linecut'`
- Background must be defined before linecut (enforced in `start_linecut_mode()`)
- Double-click finalizes drawings
- `MEASUREMENT_COLORS` list cycles through distinguishable colors

## Code Conventions

- **Use classes over globals** - state lives in `ImageData`, `Measurement` dataclasses
- **Use `QGraphicsView`/`QGraphicsScene`** for drawing (not raw `paintEvent`)
- **Use `mss`** for screen capture (cross-platform, unlike `PIL.ImageGrab`)
- **Document with docstrings** - all public functions have Args/Returns docs

## File Structure
```
opticalFlake_V0.3.py     # Active development - single-file architecture
Old_Versions/            # Legacy versions (V0.2.x) - do not modify
pip_requirements.txt     # Pin with: pip freeze > pip_requirements.txt
```

## Development Process
1. Plan changes, break into testable chunks
2. You must run the app and test before ending your turn if any edits were made
3. Refactor for clarity when adding features
