"""
opticalFlake V0.3 - Optical Flake Thickness Characterizer

A desktop tool for optical flake thickness characterization in materials science.
Analyzes optical contrast of 2D materials (graphene flakes) by capturing screenshots,
defining background regions, and computing RGB contrast along line cuts.

Dependencies: PySide6, mss, numpy, matplotlib, pillow
"""

import sys
import math
from typing import Optional
from dataclasses import dataclass, field

# IMPORTANT: Set matplotlib backend before importing PySide6
# This prevents segmentation faults on macOS
import matplotlib
matplotlib.use('QtAgg')

import numpy as np
from PIL import Image, ImageDraw
import mss

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QToolBar, QPushButton, QSpinBox, QDoubleSpinBox, QLabel, QTabWidget, QSplitter,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsPolygonItem,
    QGraphicsLineItem, QGraphicsEllipseItem, QGraphicsTextItem,
    QScrollArea, QCheckBox, QMessageBox, QGroupBox
)
from PySide6.QtCore import Qt, QPointF, QRectF, Signal
from PySide6.QtGui import (
    QPixmap, QImage, QPen, QColor, QBrush, QPolygonF, QPainter, QFont
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator, FuncFormatter


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class Measurement:
    """Stores data for a single linecut measurement."""
    segments: list  # List of (start_point, end_point) tuples
    width: int
    red_contrast: np.ndarray
    green_contrast: np.ndarray
    blue_contrast: np.ndarray
    name: str = ""
    color: str = "#FFFFFF"  # Color for linecut visualization and plot traces


# Predefined colors for measurements (distinguishable on both image and plots)
MEASUREMENT_COLORS = [
    '#FF6B6B',  # Coral red
    '#4ECDC4',  # Teal
    '#45B7D1',  # Sky blue
    '#96CEB4',  # Sage green
    '#FFEAA7',  # Pale yellow
    '#DDA0DD',  # Plum
    '#98D8C8',  # Mint
    '#F7DC6F',  # Yellow
    '#BB8FCE',  # Light purple
    '#85C1E9',  # Light blue
]


@dataclass
class ImageData:
    """Stores all data associated with an image tab."""
    pixmap: QPixmap
    pil_image: Image.Image
    background_polygon: list = field(default_factory=list)
    background_rgb: tuple = (255, 255, 255)


# =============================================================================
# Calculation Functions
# =============================================================================

def get_line_coordinates(x1: int, y1: int, x2: int, y2: int) -> list:
    """
    Bresenham's Line Algorithm to generate points between two points.
    
    Args:
        x1, y1: Coordinates of the first point
        x2, y2: Coordinates of the second point
        
    Returns:
        List of (x, y) coordinates along the line
    """
    points = []
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    steep = dy > dx

    if steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
        dx, dy = dy, dx

    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    while True:
        if steep:
            points.append((y1, x1))
        else:
            points.append((x1, y1))

        if x1 == x2 and y1 == y2:
            break
        e2 = err * 2
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy

    return points


def offset_parallel_line(x1: float, y1: float, x2: float, y2: float, offset: float) -> tuple:
    """
    Create a parallel line offset by a perpendicular distance.
    
    Args:
        x1, y1: First endpoint
        x2, y2: Second endpoint
        offset: Perpendicular offset distance
        
    Returns:
        (new_x1, new_y1, new_x2, new_y2) for the offset line
    """
    dx = x2 - x1
    dy = y2 - y1
    length = math.sqrt(dx**2 + dy**2)
    
    if length == 0:
        return x1, y1, x2, y2
    
    # Unit perpendicular vector
    unit_perp_x = -dy / length
    unit_perp_y = dx / length

    offset_x = offset * unit_perp_x
    offset_y = offset * unit_perp_y

    return (
        int(round(x1 + offset_x)),
        int(round(y1 + offset_y)),
        int(round(x2 + offset_x)),
        int(round(y2 + offset_y))
    )


def create_polygon_mask(image_size: tuple, points: list) -> Image.Image:
    """
    Create a mask image where the polygon is white and the rest is black.
    
    Args:
        image_size: (width, height) of the image
        points: List of (x, y) polygon vertices
        
    Returns:
        PIL Image mask
    """
    mask = Image.new("L", image_size, 0)
    if len(points) >= 3:
        draw = ImageDraw.Draw(mask)
        draw.polygon(points, fill=255)
    return mask


def calculate_average_color(image: Image.Image, mask: Image.Image) -> tuple:
    """
    Calculate the average RGB color within the masked region.
    
    Args:
        image: PIL Image
        mask: Binary mask image
        
    Returns:
        (r, g, b) average color tuple
    """
    img_array = np.array(image)
    mask_array = np.array(mask)
    
    # Get pixels where mask is white
    masked_pixels = img_array[mask_array > 0]
    
    if len(masked_pixels) > 0:
        r = int(np.mean(masked_pixels[:, 0]))
        g = int(np.mean(masked_pixels[:, 1]))
        b = int(np.mean(masked_pixels[:, 2]))
        return (r, g, b)
    return (255, 255, 255)


def get_line_rgb_values(image: Image.Image, x1: int, y1: int, x2: int, y2: int) -> tuple:
    """
    Get RGB values of pixels along a line.
    
    Args:
        image: PIL Image
        x1, y1: Start point
        x2, y2: End point
        
    Returns:
        (red_array, green_array, blue_array) numpy arrays
    """
    coords = get_line_coordinates(x1, y1, x2, y2)
    width, height = image.size
    
    red, green, blue = [], [], []
    for x, y in coords:
        if 0 <= x < width and 0 <= y < height:
            pixel = image.getpixel((x, y))
            if pixel is None:
                continue
            if isinstance(pixel, (tuple, list)):
                r, g, b = pixel[0], pixel[1], pixel[2]
            else:
                r = g = b = int(pixel)  # Grayscale
            red.append(r)
            green.append(g)
            blue.append(b)
    
    return np.array(red), np.array(green), np.array(blue)


def calculate_contrast(image: Image.Image, segments: list, background_rgb: tuple, width: int) -> tuple:
    """
    Calculate RGB contrast along a multi-segment linecut with averaging width.
    
    Args:
        image: PIL Image
        segments: List of ((x1, y1), (x2, y2)) segment tuples
        background_rgb: Background (r, g, b) tuple
        width: Averaging width
        
    Returns:
        (red_contrast, green_contrast, blue_contrast) numpy arrays
    """
    all_red, all_green, all_blue = [], [], []
    
    for (x1, y1), (x2, y2) in segments:
        # Get center line values
        red, green, blue = get_line_rgb_values(image, x1, y1, x2, y2)
        
        if len(red) == 0:
            continue
            
        running_red = red.astype(float)
        running_green = green.astype(float)
        running_blue = blue.astype(float)
        num_lines = 1
        
        # Add parallel lines for averaging
        half_width = width // 2
        for offset in range(1, half_width + 1):
            # Positive offset
            nx1, ny1, nx2, ny2 = offset_parallel_line(x1, y1, x2, y2, offset)
            r, g, b = get_line_rgb_values(image, nx1, ny1, nx2, ny2)
            if len(r) == len(red):
                running_red += r
                running_green += g
                running_blue += b
                num_lines += 1
            
            # Negative offset
            nx1, ny1, nx2, ny2 = offset_parallel_line(x1, y1, x2, y2, -offset)
            r, g, b = get_line_rgb_values(image, nx1, ny1, nx2, ny2)
            if len(r) == len(red):
                running_red += r
                running_green += g
                running_blue += b
                num_lines += 1
        
        # Average and calculate contrast
        avg_red = running_red / num_lines
        avg_green = running_green / num_lines
        avg_blue = running_blue / num_lines
        
        bg_r, bg_g, bg_b = background_rgb
        red_contrast = (avg_red - bg_r) / bg_r if bg_r > 0 else np.zeros_like(avg_red)
        green_contrast = (avg_green - bg_g) / bg_g if bg_g > 0 else np.zeros_like(avg_green)
        blue_contrast = (avg_blue - bg_b) / bg_b if bg_b > 0 else np.zeros_like(avg_blue)
        
        all_red.extend(red_contrast)
        all_green.extend(green_contrast)
        all_blue.extend(blue_contrast)
    
    red_arr = np.array(all_red)
    green_arr = np.array(all_green)
    blue_arr = np.array(all_blue)

    def subtract_top3_median(arr: np.ndarray) -> np.ndarray:
        if arr.size == 0:
            return arr
        k = min(3, arr.size)
        # Get top-k largest values efficiently
        topk = np.partition(arr, arr.size - k)[-k:]
        offset = float(np.median(topk))
        return arr - offset

    red_arr = subtract_top3_median(red_arr)
    green_arr = subtract_top3_median(green_arr)
    blue_arr = subtract_top3_median(blue_arr)

    return red_arr, green_arr, blue_arr


# =============================================================================
# Screen Capture Overlay
# =============================================================================

class ScreenCaptureOverlay(QWidget):
    """
    Full-screen overlay for snip-tool style screenshot selection.
    Supports both click+drag and click-then-click methods.
    """
    capture_complete = Signal(QPixmap, Image.Image)
    
    def __init__(self):
        super().__init__()
        # Use borderless window instead of true fullscreen to avoid macOS conflicts
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | 
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool  # Prevents app switching issues
        )
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setCursor(Qt.CursorShape.CrossCursor)
        
        self.start_pos = None
        self.end_pos = None
        self.screenshot = None
        self.pil_screenshot = None
        self._img_data = None  # Keep reference to prevent garbage collection
        
        # Track interaction mode
        self.is_dragging = False  # True if user is click+dragging
        self.first_click_set = False  # True if first corner is set (click-then-click mode)
        
        self._capture_screen()
        
        # Size to cover screen after capture (so we have screenshot dimensions)
        if self.screenshot:
            self.setGeometry(0, 0, self.screenshot.width(), self.screenshot.height())
        
        # Enable mouse tracking for hover preview in click-then-click mode
        self.setMouseTracking(True)
    
    def _capture_screen(self):
        """Capture the entire screen using mss."""
        try:
            # Create a new mss instance for each capture to avoid conflicts
            sct = mss.mss()
            try:
                # Capture primary monitor (excludes menu bar on macOS)
                monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
                sct_img = sct.grab(monitor)
                
                # Convert to PIL Image - copy the data immediately
                self.pil_screenshot = Image.frombytes(
                    "RGB",
                    (sct_img.width, sct_img.height),
                    bytes(sct_img.rgb)  # Convert to bytes to copy the data
                )
                
                # Convert to QPixmap for display - keep reference to img_data
                self._img_data = self.pil_screenshot.tobytes("raw", "RGB")
                qimage = QImage(
                    self._img_data,
                    self.pil_screenshot.width,
                    self.pil_screenshot.height,
                    self.pil_screenshot.width * 3,
                    QImage.Format.Format_RGB888
                )
                self.screenshot = QPixmap.fromImage(qimage.copy())  # Copy to own the data
            finally:
                # Explicitly close mss to release resources
                sct.close()
        except Exception as e:
            print(f"Screen capture error: {e}")
            self.screenshot = None
            self.pil_screenshot = None
    
    def paintEvent(self, event):
        """Draw the screenshot with selection rectangle overlay."""
        painter = QPainter(self)
        
        if self.screenshot:
            painter.drawPixmap(0, 0, self.screenshot)
        
        # Draw semi-transparent overlay
        painter.fillRect(self.rect(), QColor(0, 0, 0, 100))
        
        # Draw selection rectangle
        if self.start_pos and self.end_pos:
            rect = QRectF(self.start_pos, self.end_pos).normalized()
            # Clear the selection area
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
            painter.fillRect(rect, Qt.GlobalColor.transparent)
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
            
            # Draw the screenshot in selection area
            if self.screenshot:
                source_rect = rect.toRect()
                painter.drawPixmap(rect.toRect(), self.screenshot, source_rect)
            
            # Draw selection border
            painter.setPen(QPen(QColor(0, 120, 215), 2))
            painter.drawRect(rect)
    
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if not self.first_click_set:
                # First click - start position
                self.start_pos = event.position()
                self.end_pos = event.position()
                self.is_dragging = False  # Will be set True in mouseMoveEvent if dragging
            else:
                # Second click - finalize selection (click-then-click mode)
                self.end_pos = event.position()
                self._finalize_capture()
            self.update()
    
    def mouseMoveEvent(self, event):
        if self.start_pos:
            if event.buttons() & Qt.MouseButton.LeftButton:
                # User is dragging with button held
                self.is_dragging = True
                self.end_pos = event.position()
            elif self.first_click_set:
                # Click-then-click mode: show preview rectangle following mouse
                self.end_pos = event.position()
            self.update()
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.start_pos:
            if self.is_dragging:
                # Drag mode - finalize on release
                self.end_pos = event.position()
                self._finalize_capture()
            else:
                # Click mode - first click sets corner, wait for second click
                self.first_click_set = True
                self.end_pos = event.position()
            self.update()
    
    def _finalize_capture(self):
        """Process the selection and emit the captured region."""
        if not self.start_pos or not self.end_pos:
            self.close()
            return
            
        rect = QRectF(self.start_pos, self.end_pos).normalized().toRect()
        
        # Only process if we have a valid screenshot and reasonable selection size
        if self.pil_screenshot is not None and rect.width() > 10 and rect.height() > 10:
            try:
                # Crop the captured region
                cropped_pil = self.pil_screenshot.crop((
                    rect.x(), rect.y(),
                    rect.x() + rect.width(),
                    rect.y() + rect.height()
                ))
                
                # Convert to QPixmap - keep reference to prevent GC during QImage creation
                self._cropped_img_data = cropped_pil.tobytes("raw", "RGB")
                qimage = QImage(
                    self._cropped_img_data,
                    cropped_pil.width,
                    cropped_pil.height,
                    cropped_pil.width * 3,
                    QImage.Format.Format_RGB888
                )
                # Copy to own the pixel data before emitting
                cropped_pixmap = QPixmap.fromImage(qimage.copy())
                
                self.capture_complete.emit(cropped_pixmap, cropped_pil)
            except Exception as e:
                print(f"Error processing capture: {e}")
        
        self.close()
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close()


# =============================================================================
# Image Canvas
# =============================================================================

class ImageCanvas(QGraphicsView):
    """
    Graphics view for displaying and annotating captured images.
    """
    polygon_complete = Signal(list)  # Emits polygon points
    linecut_complete = Signal(list)  # Emits list of segments
    invalid_action = Signal(str)  # Emits error message for invalid actions
    drawing_mode_changed = Signal(bool)  # Emits True when entering drawing mode, False when exiting
    
    def __init__(self):
        super().__init__()
        self._scene = QGraphicsScene()
        self.setScene(self._scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        
        self.pixmap_item: Optional[QGraphicsPixmapItem] = None
        self.drawing_mode = None  # 'background' or 'linecut'
        self.has_background = False  # Track if background is defined
        
        # Background polygon state
        self.polygon_points = []
        self.polygon_preview_items = []
        self.polygon_item: Optional[QGraphicsPolygonItem] = None
        
        # Linecut state
        self.linecut_segments = []
        self.linecut_points = []
        self.linecut_preview_items = []
        self.current_preview_line: Optional[QGraphicsLineItem] = None
        self.width_preview_items = []
        
        # Persistent linecut graphics (stay after linecut is complete)
        self.persistent_linecut_items = []  # List of graphics items for completed linecuts
        self.measurement_count = 0  # For assigning colors
        
        # Background rectangle preview
        self.bg_preview_rect: Optional[QGraphicsPolygonItem] = None
        self.bg_preview_line: Optional[QGraphicsLineItem] = None  # Preview line for polygon mode
        
        # Background drawing state
        self.bg_drag_start: Optional[QPointF] = None  # Track drag start position
        self.bg_is_dragging = False  # True if user is click+dragging
        
        # RGB text display
        self.rgb_text_item: Optional[QGraphicsTextItem] = None
        
        self.averaging_width = 10
        self.current_linecut_color = '#FFFFFF'
        
        self.setMouseTracking(True)
    
    def set_image(self, pixmap: QPixmap):
        """Set the image to display."""
        self._scene.clear()
        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self._scene.addItem(self.pixmap_item)
        self.setSceneRect(self.pixmap_item.boundingRect())
        self.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
        
        # Reset state
        self.polygon_points = []
        self.polygon_preview_items = []
        self.polygon_item = None
        self.has_background = False
        self.linecut_segments = []
        self.linecut_points = []
        self.linecut_preview_items = []
        self.current_preview_line = None
        self.persistent_linecut_items = []
        self.measurement_count = 0
        self.bg_preview_rect = None
        self.bg_preview_line = None
        self.bg_drag_start = None
        self.bg_is_dragging = False
        self.rgb_text_item = None
    
    def start_background_mode(self):
        """Enter background polygon drawing mode."""
        self.drawing_mode = 'background'
        self.polygon_points = []
        self.bg_drag_start = None
        self.bg_is_dragging = False
        self._clear_polygon_preview()
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.drawing_mode_changed.emit(True)
    
    def start_linecut_mode(self) -> bool:
        """Enter linecut drawing mode. Returns False if background not defined."""
        if not self.has_background:
            self.invalid_action.emit("Define background region first")
            self.setCursor(Qt.CursorShape.ForbiddenCursor)
            from PySide6.QtCore import QTimer
            QTimer.singleShot(1000, lambda: self.setCursor(Qt.CursorShape.ArrowCursor))
            return False
        
        self.drawing_mode = 'linecut'
        self.linecut_points = []
        self._clear_linecut_preview()
        # Assign color for this linecut
        self.current_linecut_color = MEASUREMENT_COLORS[self.measurement_count % len(MEASUREMENT_COLORS)]
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.drawing_mode_changed.emit(True)
        return True
    
    def get_current_color(self) -> str:
        """Get the color assigned to the current/next linecut."""
        return self.current_linecut_color
    
    def stop_drawing(self):
        """Exit drawing mode."""
        was_drawing = self.drawing_mode is not None
        self.drawing_mode = None
        self.setCursor(Qt.CursorShape.ArrowCursor)
        if was_drawing:
            self.drawing_mode_changed.emit(False)
    
    def set_averaging_width(self, width: int):
        """Update the averaging width."""
        self.averaging_width = width
    
    def _clear_polygon_preview(self):
        """Clear polygon preview graphics."""
        for item in self.polygon_preview_items:
            self._scene.removeItem(item)
        self.polygon_preview_items = []
        
        if self.bg_preview_rect:
            self._scene.removeItem(self.bg_preview_rect)
            self.bg_preview_rect = None
        
        if self.bg_preview_line:
            self._scene.removeItem(self.bg_preview_line)
            self.bg_preview_line = None
    
    def _draw_background_preview(self, mouse_pos: QPointF):
        """Draw preview for background selection (rectangle or polygon edge)."""
        # Clear previous rectangle preview
        if self.bg_preview_rect:
            self._scene.removeItem(self.bg_preview_rect)
            self.bg_preview_rect = None
        
        # Clear previous line preview
        if self.bg_preview_line:
            self._scene.removeItem(self.bg_preview_line)
            self.bg_preview_line = None
        
        if self.bg_is_dragging and self.bg_drag_start:
            # Rectangle drag mode - show rectangle from drag start to current position
            p1 = self.bg_drag_start
            rect_points = [
                QPointF(p1.x(), p1.y()),
                QPointF(mouse_pos.x(), p1.y()),
                QPointF(mouse_pos.x(), mouse_pos.y()),
                QPointF(p1.x(), mouse_pos.y())
            ]
            polygon = QPolygonF(rect_points)
            self.bg_preview_rect = QGraphicsPolygonItem(polygon)
            self.bg_preview_rect.setPen(QPen(QColor(0, 255, 0), 2, Qt.PenStyle.DashLine))
            self.bg_preview_rect.setBrush(QBrush(QColor(0, 255, 0, 40)))
            self._scene.addItem(self.bg_preview_rect)
        elif len(self.polygon_points) > 0:
            # Polygon click mode - show line from last point to mouse
            last_point = self.polygon_points[-1]
            self.bg_preview_line = QGraphicsLineItem(
                last_point.x(), last_point.y(),
                mouse_pos.x(), mouse_pos.y()
            )
            self.bg_preview_line.setPen(QPen(QColor(0, 255, 0, 150), 2, Qt.PenStyle.DashLine))
            self._scene.addItem(self.bg_preview_line)
    
    def _clear_linecut_preview(self):
        """Clear linecut preview graphics."""
        for item in self.linecut_preview_items:
            self._scene.removeItem(item)
        self.linecut_preview_items = []
        
        if self.current_preview_line:
            self._scene.removeItem(self.current_preview_line)
            self.current_preview_line = None
        
        for item in self.width_preview_items:
            self._scene.removeItem(item)
        self.width_preview_items = []
    
    def _draw_polygon_preview(self):
        """Draw polygon vertices and edges preview."""
        self._clear_polygon_preview()
        
        pen = QPen(QColor(0, 255, 0), 2, Qt.PenStyle.DashLine)
        brush = QBrush(QColor(0, 255, 0))
        
        # Draw vertices
        for point in self.polygon_points:
            ellipse = QGraphicsEllipseItem(point.x() - 4, point.y() - 4, 8, 8)
            ellipse.setPen(pen)
            ellipse.setBrush(brush)
            self._scene.addItem(ellipse)
            self.polygon_preview_items.append(ellipse)
        
        # Draw edges
        for i in range(len(self.polygon_points) - 1):
            p1 = self.polygon_points[i]
            p2 = self.polygon_points[i + 1]
            line = QGraphicsLineItem(p1.x(), p1.y(), p2.x(), p2.y())
            line.setPen(pen)
            self._scene.addItem(line)
            self.polygon_preview_items.append(line)
    
    def _finalize_polygon(self, is_rectangle: bool = False):
        """Complete the polygon and display it.
        
        Args:
            is_rectangle: If True, treat 2 points as rectangle corners.
                         If False, require at least 3 points for polygon.
        """
        min_points = 2 if is_rectangle else 3
        if len(self.polygon_points) < min_points:
            # Not enough points, just cancel
            self.polygon_points = []
            self.bg_drag_start = None
            self.bg_is_dragging = False
            self._clear_polygon_preview()
            self.stop_drawing()
            return
        
        self._clear_polygon_preview()
        
        # Remove old polygon
        if self.polygon_item:
            self._scene.removeItem(self.polygon_item)
        
        # If exactly 2 points and is_rectangle, create rectangle from opposite corners
        if len(self.polygon_points) == 2 and is_rectangle:
            p1 = self.polygon_points[0]
            p2 = self.polygon_points[1]
            # Create rectangle corners (clockwise from p1)
            self.polygon_points = [
                QPointF(p1.x(), p1.y()),
                QPointF(p2.x(), p1.y()),
                QPointF(p2.x(), p2.y()),
                QPointF(p1.x(), p2.y())
            ]
        
        # Create completed polygon
        polygon = QPolygonF(self.polygon_points)
        self.polygon_item = QGraphicsPolygonItem(polygon)
        self.polygon_item.setPen(QPen(QColor(255, 0, 0), 2, Qt.PenStyle.DashLine))
        self.polygon_item.setBrush(QBrush(QColor(255, 0, 0, 30)))
        self._scene.addItem(self.polygon_item)
        
        self.has_background = True
        
        # Emit signal with points as tuples
        points_as_tuples = [(int(p.x()), int(p.y())) for p in self.polygon_points]
        self.polygon_complete.emit(points_as_tuples)
        
        self.polygon_points = []
        self.bg_drag_start = None
        self.bg_is_dragging = False
        self.stop_drawing()
    
    def _draw_linecut_preview(self, mouse_pos: Optional[QPointF] = None):
        """Draw linecut segments and preview."""
        # Clear previous preview line
        if self.current_preview_line:
            self._scene.removeItem(self.current_preview_line)
            self.current_preview_line = None
        
        for item in self.width_preview_items:
            self._scene.removeItem(item)
        self.width_preview_items = []
        
        # Use the assigned color for this linecut
        linecut_color = QColor(self.current_linecut_color)
        pen = QPen(linecut_color, 2)
        width_pen = QPen(linecut_color.lighter(150), 1, Qt.PenStyle.DashLine)
        point_brush = QBrush(linecut_color)
        
        # Draw existing points
        for item in self.linecut_preview_items:
            self._scene.removeItem(item)
        self.linecut_preview_items = []
        
        # Only draw a small dot at the starting point
        if len(self.linecut_points) > 0:
            start_point = self.linecut_points[0]
            ellipse = QGraphicsEllipseItem(start_point.x() - 2.5, start_point.y() - 2.5, 5, 5)
            ellipse.setPen(pen)
            ellipse.setBrush(point_brush)
            self._scene.addItem(ellipse)
            self.linecut_preview_items.append(ellipse)
        
        # Draw confirmed segments with direction arrows
        for i in range(len(self.linecut_points) - 1):
            p1 = self.linecut_points[i]
            p2 = self.linecut_points[i + 1]
            line = QGraphicsLineItem(p1.x(), p1.y(), p2.x(), p2.y())
            line.setPen(pen)
            self._scene.addItem(line)
            self.linecut_preview_items.append(line)
            
            # Add arrowhead to show direction
            arrow = self._create_arrowhead(p1.x(), p1.y(), p2.x(), p2.y(), linecut_color)
            if arrow:
                self._scene.addItem(arrow)
                self.linecut_preview_items.append(arrow)
            
            # Draw width preview for confirmed segments
            self._draw_width_preview(p1, p2, width_pen)
        
        # Draw preview line to mouse position
        if mouse_pos and len(self.linecut_points) > 0:
            last_point = self.linecut_points[-1]
            self.current_preview_line = QGraphicsLineItem(
                last_point.x(), last_point.y(),
                mouse_pos.x(), mouse_pos.y()
            )
            preview_pen = QPen(QColor(255, 255, 255, 150), 2, Qt.PenStyle.DashLine)
            self.current_preview_line.setPen(preview_pen)
            self._scene.addItem(self.current_preview_line)
            
            # Preview width for potential segment
            self._draw_width_preview(last_point, mouse_pos, width_pen, preview=True)
    
    def _create_arrowhead(self, x1: float, y1: float, x2: float, y2: float, 
                          color: QColor, size: float = 8) -> Optional[QGraphicsPolygonItem]:
        """Create an arrowhead pointing from (x1,y1) to (x2,y2)."""
        # Calculate direction vector
        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx*dx + dy*dy)
        
        if length == 0:
            return None
        
        # Unit vector in direction of line
        ux = dx / length
        uy = dy / length
        
        # Perpendicular unit vector
        px = -uy
        py = ux
        
        # Arrow tip at (x2, y2), base points offset back and to the sides
        tip = QPointF(x2, y2)
        base_center_x = x2 - ux * size
        base_center_y = y2 - uy * size
        
        left = QPointF(base_center_x + px * size * 0.5, base_center_y + py * size * 0.5)
        right = QPointF(base_center_x - px * size * 0.5, base_center_y - py * size * 0.5)
        
        # Create triangle polygon
        arrow_polygon = QPolygonF([tip, left, right])
        arrow_item = QGraphicsPolygonItem(arrow_polygon)
        arrow_item.setPen(QPen(color, 1))
        arrow_item.setBrush(QBrush(color))
        
        return arrow_item
    
    def _draw_width_preview(self, p1: QPointF, p2: QPointF, pen: QPen, preview: bool = False):
        """Draw parallel lines showing averaging width."""
        half_width = self.averaging_width // 2
        if half_width < 1:
            return
        
        x1, y1 = p1.x(), p1.y()
        x2, y2 = p2.x(), p2.y()
        
        # Upper parallel line
        nx1, ny1, nx2, ny2 = offset_parallel_line(x1, y1, x2, y2, half_width)
        upper = QGraphicsLineItem(nx1, ny1, nx2, ny2)
        upper.setPen(pen)
        self._scene.addItem(upper)
        self.width_preview_items.append(upper)
        
        # Lower parallel line
        nx1, ny1, nx2, ny2 = offset_parallel_line(x1, y1, x2, y2, -half_width)
        lower = QGraphicsLineItem(nx1, ny1, nx2, ny2)
        lower.setPen(pen)
        self._scene.addItem(lower)
        self.width_preview_items.append(lower)
    
    def _finalize_linecut(self):
        """Complete the linecut and emit signal."""
        if len(self.linecut_points) >= 2:
            # Build segments list
            segments = []
            for i in range(len(self.linecut_points) - 1):
                p1 = self.linecut_points[i]
                p2 = self.linecut_points[i + 1]
                segments.append((
                    (int(p1.x()), int(p1.y())),
                    (int(p2.x()), int(p2.y()))
                ))
            
            # Add persistent graphics for this linecut
            self._add_persistent_linecut(self.linecut_points, self.current_linecut_color)
            self.measurement_count += 1
            
            self.linecut_segments.extend(segments)
            self.linecut_complete.emit(segments)
        
        self.linecut_points = []
        self._clear_linecut_preview()
        self.stop_drawing()
    
    def _add_persistent_linecut(self, points: list, color: str):
        """Add permanent linecut graphics to the scene."""
        qcolor = QColor(color)
        pen = QPen(qcolor, 2, Qt.PenStyle.DashLine)
        point_brush = QBrush(qcolor)
        width_pen = QPen(qcolor.lighter(150), 1, Qt.PenStyle.DashLine)
        
        items = []
        
        # Draw only the starting point (small dot)
        if len(points) > 0:
            start_point = points[0]
            ellipse = QGraphicsEllipseItem(start_point.x() - 2.5, start_point.y() - 2.5, 5, 5)
            ellipse.setPen(pen)
            ellipse.setBrush(point_brush)
            self._scene.addItem(ellipse)
            items.append(ellipse)
        
        # Draw line segments with direction arrows
        for i in range(len(points) - 1):
            p1 = points[i]
            p2 = points[i + 1]
            line = QGraphicsLineItem(p1.x(), p1.y(), p2.x(), p2.y())
            line.setPen(pen)
            self._scene.addItem(line)
            items.append(line)
            
            # Add arrowhead to show direction
            arrow = self._create_arrowhead(p1.x(), p1.y(), p2.x(), p2.y(), qcolor)
            if arrow:
                self._scene.addItem(arrow)
                items.append(arrow)
            
            # Draw width indicators
            half_width = self.averaging_width // 2
            if half_width >= 1:
                x1, y1, x2, y2 = p1.x(), p1.y(), p2.x(), p2.y()
                nx1, ny1, nx2, ny2 = offset_parallel_line(x1, y1, x2, y2, half_width)
                upper = QGraphicsLineItem(nx1, ny1, nx2, ny2)
                upper.setPen(width_pen)
                self._scene.addItem(upper)
                items.append(upper)
                
                nx1, ny1, nx2, ny2 = offset_parallel_line(x1, y1, x2, y2, -half_width)
                lower = QGraphicsLineItem(nx1, ny1, nx2, ny2)
                lower.setPen(width_pen)
                self._scene.addItem(lower)
                items.append(lower)
        
        self.persistent_linecut_items.append(items)
    
    def remove_persistent_linecut(self, index: int):
        """Remove a persistent linecut by index."""
        if 0 <= index < len(self.persistent_linecut_items):
            for item in self.persistent_linecut_items[index]:
                self._scene.removeItem(item)
            del self.persistent_linecut_items[index]
    
    def update_persistent_linecut_width(self, index: int, new_width: int, segments: list, color: str):
        """Update the width indicator graphics for a persistent linecut."""
        if 0 <= index < len(self.persistent_linecut_items):
            # Remove old graphics
            for item in self.persistent_linecut_items[index]:
                self._scene.removeItem(item)
            
            # Recreate with new width
            qcolor = QColor(color)
            pen = QPen(qcolor, 2)
            point_brush = QBrush(qcolor)
            width_pen = QPen(qcolor.lighter(150), 1, Qt.PenStyle.DashLine)
            
            items = []
            half_width = new_width // 2
            
            # Draw points and lines for each segment
            for i, ((x1, y1), (x2, y2)) in enumerate(segments):
                # Draw only the starting point (small dot) for the first segment
                if i == 0:
                    ellipse = QGraphicsEllipseItem(x1 - 2.5, y1 - 2.5, 5, 5)
                    ellipse.setPen(pen)
                    ellipse.setBrush(point_brush)
                    self._scene.addItem(ellipse)
                    items.append(ellipse)
                
                # Draw line segment
                line = QGraphicsLineItem(x1, y1, x2, y2)
                line.setPen(pen)
                self._scene.addItem(line)
                items.append(line)
                
                # Add arrowhead to show direction
                arrow = self._create_arrowhead(x1, y1, x2, y2, qcolor)
                if arrow:
                    self._scene.addItem(arrow)
                    items.append(arrow)
                
                # Draw width indicators
                if half_width >= 1:
                    nx1, ny1, nx2, ny2 = offset_parallel_line(x1, y1, x2, y2, half_width)
                    upper = QGraphicsLineItem(nx1, ny1, nx2, ny2)
                    upper.setPen(width_pen)
                    self._scene.addItem(upper)
                    items.append(upper)
                    
                    nx1, ny1, nx2, ny2 = offset_parallel_line(x1, y1, x2, y2, -half_width)
                    lower = QGraphicsLineItem(nx1, ny1, nx2, ny2)
                    lower.setPen(width_pen)
                    self._scene.addItem(lower)
                    items.append(lower)
            
            self.persistent_linecut_items[index] = items
    
    def display_rgb_text(self, rgb: tuple):
        """Display average RGB in corner of image."""
        if self.rgb_text_item:
            self._scene.removeItem(self.rgb_text_item)
        
        text = f"Background: R={rgb[0]}, G={rgb[1]}, B={rgb[2]}"
        self.rgb_text_item = QGraphicsTextItem(text)
        self.rgb_text_item.setDefaultTextColor(QColor(255, 255, 255))
        self.rgb_text_item.setFont(QFont("Arial", 10))
        
        # Position in bottom-right corner of the image
        if self.pixmap_item:
            img_rect = self.pixmap_item.boundingRect()
            text_width = self.rgb_text_item.boundingRect().width()
            text_height = self.rgb_text_item.boundingRect().height()
            self.rgb_text_item.setPos(
                img_rect.right() - text_width - 5,
                img_rect.bottom() - text_height - 5
            )
        else:
            self.rgb_text_item.setPos(5, 5)
        
        self._scene.addItem(self.rgb_text_item)
    
    def mousePressEvent(self, event):
        if not self.drawing_mode:
            super().mousePressEvent(event)
            return
        
        scene_pos = self.mapToScene(event.position().toPoint())
        
        if self.drawing_mode == 'background':
            if len(self.polygon_points) == 0:
                # First click - could be start of drag or polygon
                self.bg_drag_start = scene_pos
                self.bg_is_dragging = False  # Will be set to True in mouseMoveEvent if dragging
            else:
                # Subsequent click - adding polygon vertex
                self.polygon_points.append(scene_pos)
                self._draw_polygon_preview()
        
        elif self.drawing_mode == 'linecut':
            self.linecut_points.append(scene_pos)
            self._draw_linecut_preview()
    
    def mouseDoubleClickEvent(self, event):
        scene_pos = self.mapToScene(event.position().toPoint())
        
        if self.drawing_mode == 'background':
            # Double-click finalizes polygon; add current point as final vertex
            self.polygon_points.append(scene_pos)
            self._finalize_polygon(is_rectangle=False)
        
        elif self.drawing_mode == 'linecut':
            self.linecut_points.append(scene_pos)
            self._finalize_linecut()
    
    def mouseMoveEvent(self, event):
        scene_pos = self.mapToScene(event.position().toPoint())
        
        if self.drawing_mode == 'background':
            if self.bg_drag_start and event.buttons() & Qt.MouseButton.LeftButton:
                # User is dragging - switch to drag mode
                self.bg_is_dragging = True
                self._draw_background_preview(scene_pos)
            elif len(self.polygon_points) > 0:
                # Polygon mode - show preview line
                self._draw_background_preview(scene_pos)
        elif self.drawing_mode == 'linecut' and len(self.linecut_points) > 0:
            self._draw_linecut_preview(scene_pos)
        
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        if self.drawing_mode == 'background' and self.bg_drag_start:
            scene_pos = self.mapToScene(event.position().toPoint())
            
            if self.bg_is_dragging:
                # Dragged - create rectangle
                self.polygon_points = [self.bg_drag_start, scene_pos]
                self._finalize_polygon(is_rectangle=True)
            else:
                # Just clicked (no drag) - start polygon mode
                self.polygon_points.append(self.bg_drag_start)
                self.bg_drag_start = None
                self._draw_polygon_preview()
        
        super().mouseReleaseEvent(event)
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.pixmap_item:
            self.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)


# =============================================================================
# Data Display Panel
# =============================================================================

class MeasurementListItem(QWidget):
    """Widget for displaying a measurement in the list."""
    width_changed = Signal(int, int)  # measurement_index, new_width
    remove_clicked = Signal(int)  # measurement_index
    
    def __init__(self, index: int, name: str, width: int, color: str):
        super().__init__()
        self.index = index
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 2, 5, 2)
        
        # Color indicator
        color_label = QLabel("●")
        color_label.setStyleSheet(f"color: {color}; font-size: 16px;")
        layout.addWidget(color_label)
        
        # Name label
        self.name_label = QLabel(name)
        self.name_label.setMinimumWidth(80)
        layout.addWidget(self.name_label)
        
        # Width input
        layout.addWidget(QLabel("Width:"))
        self.width_input = QSpinBox()
        self.width_input.setRange(1, 999)
        self.width_input.setValue(width)
        self.width_input.valueChanged.connect(self._on_width_changed)
        layout.addWidget(self.width_input)
        
        # Remove button
        self.remove_btn = QPushButton("×")
        self.remove_btn.setFixedSize(25, 25)
        self.remove_btn.clicked.connect(self._on_remove)
        layout.addWidget(self.remove_btn)
    
    def _on_width_changed(self, value):
        self.width_changed.emit(self.index, value)
    
    def _on_remove(self):
        self.remove_clicked.emit(self.index)


class DataDisplayPanel(QWidget):
    """Panel for displaying RGB contrast plots and measurement list."""
    measurement_removed = Signal(int)  # Signal when measurement is removed
    width_change_requested = Signal(int, int)  # (measurement_index, new_width)
    
    def __init__(self):
        super().__init__()
        self.measurements: list[Measurement] = []
        
        # Channel visibility state
        self.show_red = True
        self.show_green = True
        self.show_blue = False
        
        # Y-axis limit settings (fraction units; plotting scales to %)
        self.use_fixed_yaxis = False
        self.yaxis_min = -0.2
        self.yaxis_max = 0.05
        
        layout = QVBoxLayout(self)
        
        # Channel selection checkboxes
        channel_group = QGroupBox("Channels")
        channel_layout = QHBoxLayout(channel_group)
        
        self.red_checkbox = QCheckBox("Red")
        self.red_checkbox.setChecked(self.show_red)
        self.red_checkbox.setStyleSheet("QCheckBox { color: #cc0000; font-weight: bold; }")
        self.red_checkbox.stateChanged.connect(self._on_channel_changed)
        channel_layout.addWidget(self.red_checkbox)
        
        self.green_checkbox = QCheckBox("Green")
        self.green_checkbox.setChecked(self.show_green)
        self.green_checkbox.setStyleSheet("QCheckBox { color: #008800; font-weight: bold; }")
        self.green_checkbox.stateChanged.connect(self._on_channel_changed)
        channel_layout.addWidget(self.green_checkbox)
        
        self.blue_checkbox = QCheckBox("Blue")
        self.blue_checkbox.setChecked(self.show_blue)
        self.blue_checkbox.setStyleSheet("QCheckBox { color: #0066cc; font-weight: bold; }")
        self.blue_checkbox.stateChanged.connect(self._on_channel_changed)
        channel_layout.addWidget(self.blue_checkbox)
        
        layout.addWidget(channel_group)
        
        # Matplotlib figure
        self.figure = Figure(figsize=(6, 8))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas, stretch=3)
        
        # Measurement list
        layout.addWidget(QLabel("Measurements:"))
        
        self.list_scroll = QScrollArea()
        self.list_scroll.setWidgetResizable(True)
        self.list_widget = QWidget()
        self.list_layout = QVBoxLayout(self.list_widget)
        self.list_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.list_scroll.setWidget(self.list_widget)
        self.list_scroll.setMaximumHeight(150)
        layout.addWidget(self.list_scroll, stretch=1)
        
        self._update_plots()
    
    def _on_channel_changed(self):
        """Handle channel checkbox changes."""
        self.show_red = self.red_checkbox.isChecked()
        self.show_green = self.green_checkbox.isChecked()
        self.show_blue = self.blue_checkbox.isChecked()
        self._update_plots()
    
    def _setup_plots(self):
        """Initialize the matplotlib plots based on visible channels."""
        self.figure.clear()
        
        # Count visible channels
        visible_channels = []
        if self.show_red:
            visible_channels.append(('red', 'Red Channel'))
        if self.show_green:
            visible_channels.append(('green', 'Green Channel'))
        if self.show_blue:
            visible_channels.append(('blue', 'Blue Channel'))
        
        if not visible_channels:
            # No channels selected, show empty plot
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, 'Select at least one channel', 
                   ha='center', va='center', transform=ax.transAxes)
            self.figure.tight_layout()
            self.canvas.draw()
            return {}
        
        # Create subplots for visible channels
        axes = {}
        n_plots = len(visible_channels)
        for i, (color, title) in enumerate(visible_channels):
            ax = self.figure.add_subplot(n_plots, 1, i + 1)
            ax.set_xlabel('Pixels')
            ax.set_ylabel('Contrast (%)', color=color, fontweight='bold')
            ax.tick_params(axis='y', labelcolor=color)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
            # Increase number of tick marks
            ax.yaxis.set_major_locator(MaxNLocator(nbins=10))
            # Format ticks as percent
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y:.0f}%"))
            axes[color] = ax
        
        self.figure.tight_layout()
        
        # Add baseline shift note in bottom left
        self.figure.text(0.02, 0.01, '(Shifted baseline)', 
                        fontsize=8, color='gray', style='italic',
                        ha='left', va='bottom')
        
        return axes
    
    def add_measurement(self, measurement: Measurement):
        """Add a new measurement and update display."""
        measurement.name = f"Linecut {len(self.measurements) + 1}"
        self.measurements.append(measurement)
        self._update_plots()
        self._update_list()
    
    def set_yaxis_limits(self, use_fixed: bool, y_min: float, y_max: float):
        """Set Y-axis limit parameters and refresh plots."""
        self.use_fixed_yaxis = use_fixed
        self.yaxis_min = y_min
        self.yaxis_max = y_max
        self._update_plots()
    
    def remove_measurement(self, index: int):
        """Remove a measurement by index."""
        if 0 <= index < len(self.measurements):
            del self.measurements[index]
            # Update names
            for i, m in enumerate(self.measurements):
                m.name = f"Linecut {i + 1}"
            self._update_plots()
            self._update_list()
            self.measurement_removed.emit(index)
    
    def update_measurement_width(self, index: int, new_width: int):
        """Request recalculation with new width (actual update happens in ImageTab)."""
        if 0 <= index < len(self.measurements):
            self.width_change_requested.emit(index, new_width)
    
    def _update_plots(self):
        """Redraw all plots with current measurements."""
        axes = self._setup_plots()
        
        if not axes:
            return
        
        for i, m in enumerate(self.measurements):
            # Use the measurement's assigned color for the trace
            color = m.color
            alpha = 0.9
            
            if len(m.red_contrast) > 0:
                x = np.arange(len(m.red_contrast))
                
                if self.show_red and 'red' in axes:
                    axes['red'].plot(x, m.red_contrast * 100.0, color=color, alpha=alpha, 
                                   label=m.name, linewidth=1.5)
                if self.show_green and 'green' in axes:
                    axes['green'].plot(x, m.green_contrast * 100.0, color=color, alpha=alpha,
                                     label=m.name, linewidth=1.5)
                if self.show_blue and 'blue' in axes:
                    axes['blue'].plot(x, m.blue_contrast * 100.0, color=color, alpha=alpha,
                                    label=m.name, linewidth=1.5)
        
        # Add legends to first visible axis
        if self.measurements and axes:
            first_ax = list(axes.values())[0]
            first_ax.legend(fontsize=8, loc='upper right')
        
        # Apply fixed Y-axis limits if enabled
        if self.use_fixed_yaxis:
            for ax in axes.values():
                ax.set_ylim(self.yaxis_min * 100.0, self.yaxis_max * 100.0)
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def update_measurement_data(self, index: int, red: np.ndarray, green: np.ndarray, blue: np.ndarray):
        """Update a measurement's contrast data and refresh plots."""
        if 0 <= index < len(self.measurements):
            self.measurements[index].red_contrast = red
            self.measurements[index].green_contrast = green
            self.measurements[index].blue_contrast = blue
            self._update_plots()
    
    def _update_list(self):
        """Update the measurement list widget."""
        # Clear existing items
        while self.list_layout.count():
            item = self.list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Add items for each measurement
        for i, m in enumerate(self.measurements):
            item = MeasurementListItem(i, m.name, m.width, m.color)
            item.width_changed.connect(self.update_measurement_width)
            item.remove_clicked.connect(self.remove_measurement)
            self.list_layout.addWidget(item)


# =============================================================================
# Image Tab
# =============================================================================

class ImageTab(QWidget):
    """Tab containing image canvas and data display for one captured image."""
    
    def __init__(self, pixmap: QPixmap, pil_image: Image.Image):
        super().__init__()
        self.data = ImageData(pixmap=pixmap, pil_image=pil_image)
        
        layout = QHBoxLayout(self)
        
        # Splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Image canvas
        self.canvas = ImageCanvas()
        self.canvas.set_image(pixmap)
        self.canvas.polygon_complete.connect(self._on_polygon_complete)
        self.canvas.linecut_complete.connect(self._on_linecut_complete)
        self.canvas.invalid_action.connect(self._on_invalid_action)
        # Expose drawing_mode_changed signal for parent to connect
        self.drawing_mode_changed = self.canvas.drawing_mode_changed
        splitter.addWidget(self.canvas)
        
        # Data display
        self.data_panel = DataDisplayPanel()
        self.data_panel.measurement_removed.connect(self._on_measurement_removed)
        self.data_panel.width_change_requested.connect(self._on_width_change_requested)
        splitter.addWidget(self.data_panel)
        
        splitter.setSizes([500, 400])
        layout.addWidget(splitter)
    
    def start_background(self):
        """Start background polygon drawing."""
        self.canvas.start_background_mode()
    
    def start_linecut(self, width: int) -> bool:
        """Start linecut drawing. Returns False if not allowed."""
        self.canvas.set_averaging_width(width)
        return self.canvas.start_linecut_mode()
    
    def _on_invalid_action(self, message: str):
        """Handle invalid action notification."""
        # Show status message (could be enhanced with a status bar)
        QMessageBox.warning(self, "Invalid Action", message)
    
    def _on_measurement_removed(self, index: int):
        """Handle measurement removal - also remove from canvas."""
        self.canvas.remove_persistent_linecut(index)

    def _on_width_change_requested(self, index: int, new_width: int):
        """Recalculate measurement data when width changes."""
        measurements = self.data_panel.measurements
        if 0 <= index < len(measurements):
            measurement = measurements[index]
            # Recalculate contrast with new width
            red, green, blue = calculate_contrast(
                self.data.pil_image,
                measurement.segments,
                self.data.background_rgb,
                new_width
            )
            # Update the measurement data
            measurement.red_contrast = red
            measurement.green_contrast = green
            measurement.blue_contrast = blue
            measurement.width = new_width
            # Update the display
            self.data_panel.update_measurement_data(index, red, green, blue)
            # Update the linecut graphics on the canvas
            self.canvas.update_persistent_linecut_width(
                index, new_width, measurement.segments, measurement.color
            )
    
    def _recalculate_all_measurements(self):
        """Recalculate all measurements with current background."""
        for i, measurement in enumerate(self.data_panel.measurements):
            red, green, blue = calculate_contrast(
                self.data.pil_image,
                measurement.segments,
                self.data.background_rgb,
                measurement.width
            )
            measurement.red_contrast = red
            measurement.green_contrast = green
            measurement.blue_contrast = blue
            self.data_panel.update_measurement_data(i, red, green, blue)
    
    def _on_polygon_complete(self, points: list):
        """Handle completed background polygon."""
        self.data.background_polygon = points
        
        # Calculate average color
        mask = create_polygon_mask(self.data.pil_image.size, points)
        self.data.background_rgb = calculate_average_color(self.data.pil_image, mask)
        
        # Display RGB on canvas
        self.canvas.display_rgb_text(self.data.background_rgb)
        
        # Recalculate all existing measurements with new background
        if self.data_panel.measurements:
            self._recalculate_all_measurements()
    
    def _on_linecut_complete(self, segments: list):
        """Handle completed linecut."""
        if not self.data.background_polygon:
            # No background defined yet
            return
        
        # Get the color assigned to this linecut
        linecut_color = self.canvas.get_current_color()
        
        # Calculate contrast
        red, green, blue = calculate_contrast(
            self.data.pil_image,
            segments,
            self.data.background_rgb,
            self.canvas.averaging_width
        )
        
        # Create measurement with matching color
        measurement = Measurement(
            segments=segments,
            width=self.canvas.averaging_width,
            red_contrast=red,
            green_contrast=green,
            blue_contrast=blue,
            color=linecut_color
        )
        
        self.data_panel.add_measurement(measurement)


# =============================================================================
# Main Window
# =============================================================================

class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Optical Flake Thickness Characterizer V0.3")
        self.setMinimumSize(1200, 800)
        
        # Track selection mode state
        self._in_selection_mode = False
        
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Toolbar
        self.toolbar = QToolBar()
        self.toolbar.setMovable(False)
        self.toolbar.setStyleSheet("QToolBar { spacing: 8px; padding: 4px; }")
        self.addToolBar(self.toolbar)
        
        # Capture button
        self.capture_btn = QPushButton("Capture Image")
        self.capture_btn.clicked.connect(self._start_capture)
        self.toolbar.addWidget(self.capture_btn)
        
        self.toolbar.addSeparator()
        
        # Draw Background button
        self.bg_btn = QPushButton("Draw Background")
        self.bg_btn.clicked.connect(self._start_background)
        self.bg_btn.setEnabled(False)
        self.toolbar.addWidget(self.bg_btn)
        
        # Draw Linecut button
        self.linecut_btn = QPushButton("Draw Linecut")
        self.linecut_btn.clicked.connect(self._start_linecut)
        self.linecut_btn.setEnabled(False)
        self.toolbar.addWidget(self.linecut_btn)
        
        # Cancel button (only visible during selection mode)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._cancel_selection)
        self.cancel_btn.setVisible(False)
        self.toolbar.addWidget(self.cancel_btn)
        
        self.toolbar.addSeparator()
        
        # Width input
        self.width_label = QLabel(" Width:")
        self.toolbar.addWidget(self.width_label)
        self.width_input = QSpinBox()
        self.width_input.setRange(1, 999)
        self.width_input.setValue(10)
        self.width_input.setToolTip("Averaging width for linecut")
        self.toolbar.addWidget(self.width_input)
        
        self.toolbar.addSeparator()
        
        # Y-axis controls
        self.yaxis_checkbox = QCheckBox(" Fixed Y-Axis")
        self.yaxis_checkbox.setChecked(False)
        self.yaxis_checkbox.setToolTip("Use fixed Y-axis limits across all plots")
        self.yaxis_checkbox.stateChanged.connect(self._on_yaxis_settings_changed)
        self.toolbar.addWidget(self.yaxis_checkbox)
        
        self.yaxis_min_label = QLabel(" Min:")
        self.toolbar.addWidget(self.yaxis_min_label)
        self.yaxis_min_input = QDoubleSpinBox()
        self.yaxis_min_input.setRange(-10.0, 10.0)
        self.yaxis_min_input.setValue(-0.2)
        self.yaxis_min_input.setSingleStep(0.1)
        self.yaxis_min_input.setDecimals(2)
        self.yaxis_min_input.setToolTip("Minimum Y-axis value")
        self.yaxis_min_input.valueChanged.connect(self._on_yaxis_settings_changed)
        self.toolbar.addWidget(self.yaxis_min_input)
        
        self.yaxis_max_label = QLabel(" Max:")
        self.toolbar.addWidget(self.yaxis_max_label)
        self.yaxis_max_input = QDoubleSpinBox()
        self.yaxis_max_input.setRange(-10.0, 10.0)
        self.yaxis_max_input.setValue(0.05)
        self.yaxis_max_input.setSingleStep(0.1)
        self.yaxis_max_input.setDecimals(2)
        self.yaxis_max_input.setToolTip("Maximum Y-axis value")
        self.yaxis_max_input.valueChanged.connect(self._on_yaxis_settings_changed)
        self.toolbar.addWidget(self.yaxis_max_input)
        
        # Spacer to push mode indicator to the right
        spacer = QWidget()
        spacer.setSizePolicy(spacer.sizePolicy().horizontalPolicy(), spacer.sizePolicy().verticalPolicy())
        spacer.setMinimumWidth(20)
        self.toolbar.addWidget(spacer)
        
        # Selection mode indicator
        self.mode_indicator = QLabel("")
        self.mode_indicator.setStyleSheet("font-weight: bold;")
        self.toolbar.addWidget(self.mode_indicator)
        
        # Tab widget
        self.tabs = QTabWidget()
        self.tabs.setTabsClosable(True)
        self.tabs.tabCloseRequested.connect(self._close_tab)
        self.tabs.currentChanged.connect(self._tab_changed)
        layout.addWidget(self.tabs)
        
        # Capture overlay reference
        self.capture_overlay: Optional[ScreenCaptureOverlay] = None
    
    def _start_capture(self):
        """Start screen capture mode."""
        self.hide()
        QApplication.processEvents()
        
        # Small delay to ensure window is hidden
        from PySide6.QtCore import QTimer
        QTimer.singleShot(200, self._show_capture_overlay)
    
    def _show_capture_overlay(self):
        """Show the capture overlay after delay."""
        try:
            self.capture_overlay = ScreenCaptureOverlay()
            # Check if capture was successful
            if self.capture_overlay.screenshot is None:
                print("Screen capture failed - could not grab screen")
                self.capture_overlay.close()
                self.capture_overlay = None
                self.show()
                return
            self.capture_overlay.capture_complete.connect(self._on_capture_complete)
            self.capture_overlay.destroyed.connect(self._on_capture_cancelled)
            self.capture_overlay.show()
        except Exception as e:
            print(f"Error creating capture overlay: {e}")
            self.show()
    
    def _on_capture_complete(self, pixmap: QPixmap, pil_image: Image.Image):
        """Handle completed screen capture."""
        self.show()
        
        # Create new tab
        tab = ImageTab(pixmap, pil_image)
        tab.drawing_mode_changed.connect(self._on_drawing_mode_changed)
        
        # Apply current Y-axis settings to new tab
        tab.data_panel.set_yaxis_limits(
            self.yaxis_checkbox.isChecked(),
            self.yaxis_min_input.value(),
            self.yaxis_max_input.value()
        )
        
        index = self.tabs.addTab(tab, f"Image {self.tabs.count() + 1}")
        self.tabs.setCurrentIndex(index)
        
        self._update_button_states()
    
    def _on_capture_cancelled(self):
        """Handle cancelled capture."""
        self.show()
    
    def _close_tab(self, index: int):
        """Close a tab."""
        self.tabs.removeTab(index)
        self._update_button_states()
    
    def _tab_changed(self, index: int):
        """Handle tab change."""
        self._update_button_states()
    
    def _update_button_states(self):
        """Update button enabled states based on current tab."""
        has_tab = self.tabs.count() > 0
        # Don't enable buttons if in selection mode
        if not self._in_selection_mode:
            self.bg_btn.setEnabled(has_tab)
            self.linecut_btn.setEnabled(has_tab)
            self.capture_btn.setEnabled(True)
            self.width_input.setEnabled(True)
            self.width_label.setEnabled(True)
    
    def _current_tab(self) -> Optional[ImageTab]:
        """Get the current image tab."""
        widget = self.tabs.currentWidget()
        if isinstance(widget, ImageTab):
            return widget
        return None
    
    def _start_background(self):
        """Start background drawing on current tab."""
        tab = self._current_tab()
        if tab:
            tab.start_background()
    
    def _start_linecut(self):
        """Start linecut drawing on current tab."""
        tab = self._current_tab()
        if tab:
            tab.start_linecut(self.width_input.value())
    
    def _on_yaxis_settings_changed(self):
        """Update Y-axis settings for all tabs."""
        use_fixed = self.yaxis_checkbox.isChecked()
        y_min = self.yaxis_min_input.value()
        y_max = self.yaxis_max_input.value()
        
        # Apply to all tabs
        for i in range(self.tabs.count()):
            tab = self.tabs.widget(i)
            if isinstance(tab, ImageTab):
                tab.data_panel.set_yaxis_limits(use_fixed, y_min, y_max)
    
    def _on_drawing_mode_changed(self, is_drawing: bool):
        """Handle drawing mode state changes - gray out UI when in selection mode."""
        self._in_selection_mode = is_drawing
        
        if is_drawing:
            # Disable toolbar controls during selection
            self.capture_btn.setEnabled(False)
            self.bg_btn.setEnabled(False)
            self.linecut_btn.setEnabled(False)
            self.width_input.setEnabled(False)
            self.width_label.setEnabled(False)
            self.yaxis_checkbox.setEnabled(False)
            self.yaxis_min_input.setEnabled(False)
            self.yaxis_max_input.setEnabled(False)
            self.yaxis_min_label.setEnabled(False)
            self.yaxis_max_label.setEnabled(False)
            self.tabs.tabBar().setEnabled(False)
            self.cancel_btn.setVisible(True)
            
            # Style cancel button prominently (works in light/dark)
            self.cancel_btn.setStyleSheet(
                "QPushButton { background-color: #dc3545; color: white; font-weight: bold; "
                "border: none; padding: 4px 12px; border-radius: 3px; }"
                "QPushButton:hover { background-color: #c82333; }"
            )
            
            # Update mode indicator with theme-aware orange
            self.mode_indicator.setStyleSheet("color: #ff8c00; font-weight: bold;")
            
            # Update mode indicator
            tab = self._current_tab()
            if tab and tab.canvas.drawing_mode == 'background':
                self.mode_indicator.setText("🎯 Drag for rectangle, or click vertices then double-click to finish polygon")
            elif tab and tab.canvas.drawing_mode == 'linecut':
                self.mode_indicator.setText("🎯 Click to draw linecut, double-click to finish")
        else:
            # Re-enable toolbar controls
            self.tabs.tabBar().setEnabled(True)
            self.cancel_btn.setVisible(False)
            self.cancel_btn.setStyleSheet("")  # Reset cancel button style
            self.mode_indicator.setText("")
            self.mode_indicator.setStyleSheet("font-weight: bold;")  # Reset mode indicator
            self.yaxis_checkbox.setEnabled(True)
            self.yaxis_min_input.setEnabled(True)
            self.yaxis_max_input.setEnabled(True)
            self.yaxis_min_label.setEnabled(True)
            self.yaxis_max_label.setEnabled(True)
            self._update_button_states()
    
    def _cancel_selection(self):
        """Cancel the current selection/drawing mode."""
        tab = self._current_tab()
        if tab:
            tab.canvas.stop_drawing()
            tab.canvas._clear_polygon_preview()
            tab.canvas._clear_linecut_preview()
            tab.canvas.polygon_points = []
            tab.canvas.linecut_points = []
            tab.canvas.bg_drag_start = None
            tab.canvas.bg_is_dragging = False


# =============================================================================
# Entry Point
# =============================================================================

def main():
    # Ensure only one QApplication instance
    app = QApplication.instance()
    if not isinstance(app, QApplication):
        app = QApplication(sys.argv)
    
    app.setStyle('Fusion')
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

