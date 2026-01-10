# opticalFlake V0.3 - Project Description

## Program Structure

### Overall Layout
1. **Control Toolbar** — Top of window
2. **Image & Data Display** — Multi-tabbed section below toolbar
    Each tab consists of:
   - Image panel (left)
   - Data panel (right)

---

## Control Toolbar

### Capture Image
- Click-and-drag selection (snip tool behavior)
- Opens new tab with captured screenshot
- Image can then be annotated with background region and linecuts

### Draw Background
- Click to place polygon vertices
- Double-click to finish and close polygon
- **Preview**: Lines and vertices shown while drawing
- **On completion**: Average RGB displayed in image corner

### Draw Linecut
- Single-click sets starting point (marked on image)
- Preview line follows cursor
- Click adds segment (can daisy-chain multiple connected segments)
- Double-click ends linecut selection → triggers contrast calculation

### Width Field
- Sets averaging width (parallel region to linecut)
- Width is previewed alongside linecut visualization
- Width value is stored per-measurement at creation time

---

## Data Display Section

### Plot Behavior
- On linecut completion (double-click), RGB contrast channels are plotted
- Additional linecuts on same image are co-plotted in the same data section

### Measurement List
Scrollable list containing each measurement with:
- Editable width field (updates visualization on change)
- Remove button (×) to delete measurement

---

## Data Persistence
- Data is calculated at creation time
- Tabs replace the old "Clear Screenshot" functionality
