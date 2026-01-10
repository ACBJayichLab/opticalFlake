# opticalFlake

Desktop tool for optical flake thickness characterization in materials science. Analyzes optical contrast of 2D materials (graphene flakes) by capturing screenshots, defining background regions, and computing RGB contrast along line cuts.

## Quick Reference

### Run the Application
```zsh
source .venv/bin/activate
python opticalFlake_V0.3.py
```

### Dependencies
| Package | Purpose |
|---------|---------|
| PySide6 | GUI framework with `QGraphicsView` for drawing canvas |
| mss | Cross-platform screen capture |
| numpy | Array operations for contrast calculations |
| matplotlib | RGB contrast plotting via `FigureCanvasQTAgg` |
| pillow | Image manipulation for polygon masking |

## Architecture

### Core Calculation
```python
contrast = (sample_RGB - background_RGB) / background_RGB
```
Applied per-pixel along linecut, averaged across specified width. Contrast values are typically small (<10%).

### Key Algorithms
- **Bresenham's line rasterization** for pixel sampling along arbitrary angles
- **Parallel line offset** for averaging width calculation
- **Polygon mask creation** for background region averaging

### UI Components
- **Control Toolbar**: Capture Image, Draw Background, Draw Linecut, Width field
- **Tab Bar**: Multiple images supported, each with own data plot
- **Image Display**: `QGraphicsView`/`QGraphicsScene` canvas with annotations
- **Data Display**: Matplotlib RGB contrast plots + measurement list

## Code Style

- Use classes—avoid global variables
- Separate UI, data model, and calculation logic
- Use `QGraphicsView`/`QGraphicsScene` for drawing (avoids click registration issues)
- Prefer `mss` over `PIL.ImageGrab` for screen capture

```python
# Avoid
global background_rgb
background_rgb = calculate_background()

# Prefer
class ImageTab:
    def __init__(self):
        self.background_rgb = None
    
    def calculate_background(self, polygon_points):
        self.background_rgb = ...
```
# Code Process
- Do so thoroughly setting out tasks, testing, and repeating.
- Look up resources. 
- Do not hesitate to refactor code for clarity and efficiency.
- Maintain code structure and readability.
- Document functions and classes with docstrings.
- Continue until task is completed

# Testing and Validation
- Test unit chunks to verify functionality as the project grows.

## File Structure
- `opticalFlake_V0.3.py` - Current main implementation (PySide6 rewrite)
- `opticalFlake_V0.2.1.py` - Previous version (legacy)
- `opticalFlake_V0.2.py` - Earlier version (legacy)
