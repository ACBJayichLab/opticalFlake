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

import numpy as np
from PIL import Image, ImageDraw
import mss
import mss.tools

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QToolBar, QPushButton, QSpinBox, QLabel, QTabWidget, QSplitter,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsPolygonItem,
    QGraphicsLineItem, QGraphicsEllipseItem, QGraphicsTextItem,
    QScrollArea, QListWidget, QListWidgetItem, QLineEdit
)
from PySide6.QtCore import Qt, QPointF, QRectF, Signal, QObject
from PySide6.QtGui import (
    QPixmap, QImage, QPen, QColor, QBrush, QPolygonF, QPainter, QFont
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


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


@dataclass
class ImageData:
    """Stores all data associated with an image tab."""
    pixmap: QPixmap
    pil_image: Image.Image
    background_polygon: list = field(default_factory=list)
    background_rgb: tuple = (255, 255, 255)
    measurements: list = field(default_factory=list)


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
            r, g, b = image.getpixel((x, y))[:3]
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
    
    return np.array(all_red), np.array(all_green), np.array(all_blue)


# =============================================================================
# Screen Capture Overlay
# =============================================================================

class ScreenCaptureOverlay(QWidget):
    """
    Full-screen overlay for snip-tool style screenshot selection.
    """
    capture_complete = Signal(QPixmap, Image.Image)
    
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setWindowState(Qt.WindowFullScreen)
        self.setCursor(Qt.CrossCursor)
        
        self.start_pos = None
        self.end_pos = None
        self.screenshot = None
        self.pil_screenshot = None
        
        self._capture_screen()
    
    def _capture_screen(self):
        """Capture the entire screen using mss."""
        with mss.mss() as sct:
            # Capture all monitors
            monitor = sct.monitors[0]  # All monitors combined
            sct_img = sct.grab(monitor)
            
            # Convert to PIL Image
            self.pil_screenshot = Image.frombytes(
                "RGB",
                (sct_img.width, sct_img.height),
                sct_img.rgb
            )
            
            # Convert to QPixmap for display
            img_data = self.pil_screenshot.tobytes("raw", "RGB")
            qimage = QImage(
                img_data,
                self.pil_screenshot.width,
                self.pil_screenshot.height,
                self.pil_screenshot.width * 3,
                QImage.Format_RGB888
            )
            self.screenshot = QPixmap.fromImage(qimage)
    
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
            painter.setCompositionMode(QPainter.CompositionMode_Clear)
            painter.fillRect(rect, Qt.transparent)
            painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
            
            # Draw the screenshot in selection area
            if self.screenshot:
                source_rect = rect.toRect()
                painter.drawPixmap(rect.toRect(), self.screenshot, source_rect)
            
            # Draw selection border
            painter.setPen(QPen(QColor(0, 120, 215), 2))
            painter.drawRect(rect)
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.start_pos = event.position()
            self.end_pos = event.position()
            self.update()
    
    def mouseMoveEvent(self, event):
        if self.start_pos:
            self.end_pos = event.position()
            self.update()
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.start_pos and self.end_pos:
            rect = QRectF(self.start_pos, self.end_pos).normalized().toRect()
            
            if rect.width() > 10 and rect.height() > 10:
                # Crop the captured region
                cropped_pil = self.pil_screenshot.crop((
                    rect.x(), rect.y(),
                    rect.x() + rect.width(),
                    rect.y() + rect.height()
                ))
                
                # Convert to QPixmap
                img_data = cropped_pil.tobytes("raw", "RGB")
                qimage = QImage(
                    img_data,
                    cropped_pil.width,
                    cropped_pil.height,
                    cropped_pil.width * 3,
                    QImage.Format_RGB888
                )
                cropped_pixmap = QPixmap.fromImage(qimage)
                
                self.capture_complete.emit(cropped_pixmap, cropped_pil)
            
            self.close()
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
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
    
    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.NoDrag)
        
        self.pixmap_item: Optional[QGraphicsPixmapItem] = None
        self.drawing_mode = None  # 'background' or 'linecut'
        
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
        
        # RGB text display
        self.rgb_text_item: Optional[QGraphicsTextItem] = None
        
        self.averaging_width = 10
        
        self.setMouseTracking(True)
    
    def set_image(self, pixmap: QPixmap):
        """Set the image to display."""
        self.scene.clear()
        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.pixmap_item)
        self.setSceneRect(self.pixmap_item.boundingRect())
        self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
        
        # Reset state
        self.polygon_points = []
        self.polygon_preview_items = []
        self.polygon_item = None
        self.linecut_segments = []
        self.linecut_points = []
        self.linecut_preview_items = []
        self.current_preview_line = None
        self.rgb_text_item = None
    
    def start_background_mode(self):
        """Enter background polygon drawing mode."""
        self.drawing_mode = 'background'
        self.polygon_points = []
        self._clear_polygon_preview()
        self.setCursor(Qt.CrossCursor)
    
    def start_linecut_mode(self):
        """Enter linecut drawing mode."""
        self.drawing_mode = 'linecut'
        self.linecut_points = []
        self._clear_linecut_preview()
        self.setCursor(Qt.CrossCursor)
    
    def stop_drawing(self):
        """Exit drawing mode."""
        self.drawing_mode = None
        self.setCursor(Qt.ArrowCursor)
    
    def set_averaging_width(self, width: int):
        """Update the averaging width."""
        self.averaging_width = width
    
    def _clear_polygon_preview(self):
        """Clear polygon preview graphics."""
        for item in self.polygon_preview_items:
            self.scene.removeItem(item)
        self.polygon_preview_items = []
    
    def _clear_linecut_preview(self):
        """Clear linecut preview graphics."""
        for item in self.linecut_preview_items:
            self.scene.removeItem(item)
        self.linecut_preview_items = []
        
        if self.current_preview_line:
            self.scene.removeItem(self.current_preview_line)
            self.current_preview_line = None
        
        for item in self.width_preview_items:
            self.scene.removeItem(item)
        self.width_preview_items = []
    
    def _draw_polygon_preview(self):
        """Draw polygon vertices and edges preview."""
        self._clear_polygon_preview()
        
        pen = QPen(QColor(0, 255, 0), 2)
        brush = QBrush(QColor(0, 255, 0))
        
        # Draw vertices
        for point in self.polygon_points:
            ellipse = QGraphicsEllipseItem(point.x() - 4, point.y() - 4, 8, 8)
            ellipse.setPen(pen)
            ellipse.setBrush(brush)
            self.scene.addItem(ellipse)
            self.polygon_preview_items.append(ellipse)
        
        # Draw edges
        for i in range(len(self.polygon_points) - 1):
            p1 = self.polygon_points[i]
            p2 = self.polygon_points[i + 1]
            line = QGraphicsLineItem(p1.x(), p1.y(), p2.x(), p2.y())
            line.setPen(pen)
            self.scene.addItem(line)
            self.polygon_preview_items.append(line)
    
    def _finalize_polygon(self):
        """Complete the polygon and display it."""
        if len(self.polygon_points) >= 3:
            self._clear_polygon_preview()
            
            # Remove old polygon
            if self.polygon_item:
                self.scene.removeItem(self.polygon_item)
            
            # Create completed polygon
            polygon = QPolygonF(self.polygon_points)
            self.polygon_item = QGraphicsPolygonItem(polygon)
            self.polygon_item.setPen(QPen(QColor(255, 0, 0), 2))
            self.polygon_item.setBrush(QBrush(QColor(255, 0, 0, 30)))
            self.scene.addItem(self.polygon_item)
            
            # Emit signal with points as tuples
            points_as_tuples = [(int(p.x()), int(p.y())) for p in self.polygon_points]
            self.polygon_complete.emit(points_as_tuples)
        
        self.polygon_points = []
        self.stop_drawing()
    
    def _draw_linecut_preview(self, mouse_pos: QPointF = None):
        """Draw linecut segments and preview."""
        # Clear previous preview line
        if self.current_preview_line:
            self.scene.removeItem(self.current_preview_line)
            self.current_preview_line = None
        
        for item in self.width_preview_items:
            self.scene.removeItem(item)
        self.width_preview_items = []
        
        pen = QPen(QColor(255, 255, 255), 2)
        width_pen = QPen(QColor(255, 255, 0), 1, Qt.DashLine)
        point_brush = QBrush(QColor(255, 255, 255))
        
        # Draw existing points
        for item in self.linecut_preview_items:
            self.scene.removeItem(item)
        self.linecut_preview_items = []
        
        for point in self.linecut_points:
            ellipse = QGraphicsEllipseItem(point.x() - 4, point.y() - 4, 8, 8)
            ellipse.setPen(pen)
            ellipse.setBrush(point_brush)
            self.scene.addItem(ellipse)
            self.linecut_preview_items.append(ellipse)
        
        # Draw confirmed segments
        for i in range(len(self.linecut_points) - 1):
            p1 = self.linecut_points[i]
            p2 = self.linecut_points[i + 1]
            line = QGraphicsLineItem(p1.x(), p1.y(), p2.x(), p2.y())
            line.setPen(pen)
            self.scene.addItem(line)
            self.linecut_preview_items.append(line)
            
            # Draw width preview for confirmed segments
            self._draw_width_preview(p1, p2, width_pen)
        
        # Draw preview line to mouse position
        if mouse_pos and len(self.linecut_points) > 0:
            last_point = self.linecut_points[-1]
            self.current_preview_line = QGraphicsLineItem(
                last_point.x(), last_point.y(),
                mouse_pos.x(), mouse_pos.y()
            )
            preview_pen = QPen(QColor(255, 255, 255, 150), 2, Qt.DashLine)
            self.current_preview_line.setPen(preview_pen)
            self.scene.addItem(self.current_preview_line)
            
            # Preview width for potential segment
            self._draw_width_preview(last_point, mouse_pos, width_pen, preview=True)
    
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
        self.scene.addItem(upper)
        self.width_preview_items.append(upper)
        
        # Lower parallel line
        nx1, ny1, nx2, ny2 = offset_parallel_line(x1, y1, x2, y2, -half_width)
        lower = QGraphicsLineItem(nx1, ny1, nx2, ny2)
        lower.setPen(pen)
        self.scene.addItem(lower)
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
            
            self.linecut_segments.extend(segments)
            self.linecut_complete.emit(segments)
        
        self.linecut_points = []
        self._clear_linecut_preview()
        self.stop_drawing()
    
    def display_rgb_text(self, rgb: tuple):
        """Display average RGB in corner of image."""
        if self.rgb_text_item:
            self.scene.removeItem(self.rgb_text_item)
        
        text = f"Background: R={rgb[0]}, G={rgb[1]}, B={rgb[2]}"
        self.rgb_text_item = QGraphicsTextItem(text)
        self.rgb_text_item.setDefaultTextColor(QColor(255, 255, 255))
        self.rgb_text_item.setFont(QFont("Arial", 10))
        
        # Position in top-left corner
        self.rgb_text_item.setPos(5, 5)
        
        # Add background rectangle for visibility
        self.scene.addItem(self.rgb_text_item)
    
    def mousePressEvent(self, event):
        if not self.drawing_mode:
            super().mousePressEvent(event)
            return
        
        scene_pos = self.mapToScene(event.position().toPoint())
        
        if self.drawing_mode == 'background':
            self.polygon_points.append(scene_pos)
            self._draw_polygon_preview()
        
        elif self.drawing_mode == 'linecut':
            self.linecut_points.append(scene_pos)
            self._draw_linecut_preview()
    
    def mouseDoubleClickEvent(self, event):
        scene_pos = self.mapToScene(event.position().toPoint())
        
        if self.drawing_mode == 'background':
            self.polygon_points.append(scene_pos)
            self._finalize_polygon()
        
        elif self.drawing_mode == 'linecut':
            self.linecut_points.append(scene_pos)
            self._finalize_linecut()
    
    def mouseMoveEvent(self, event):
        if self.drawing_mode == 'linecut' and len(self.linecut_points) > 0:
            scene_pos = self.mapToScene(event.position().toPoint())
            self._draw_linecut_preview(scene_pos)
        super().mouseMoveEvent(event)
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.pixmap_item:
            self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)


# =============================================================================
# Data Display Panel
# =============================================================================

class MeasurementListItem(QWidget):
    """Widget for displaying a measurement in the list."""
    width_changed = Signal(int, int)  # measurement_index, new_width
    remove_clicked = Signal(int)  # measurement_index
    
    def __init__(self, index: int, name: str, width: int):
        super().__init__()
        self.index = index
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 2, 5, 2)
        
        # Name label
        self.name_label = QLabel(name)
        self.name_label.setMinimumWidth(100)
        layout.addWidget(self.name_label)
        
        # Width input
        layout.addWidget(QLabel("Width:"))
        self.width_input = QSpinBox()
        self.width_input.setRange(1, 100)
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
    
    def __init__(self):
        super().__init__()
        self.measurements: list[Measurement] = []
        
        layout = QVBoxLayout(self)
        
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
        self.list_layout.setAlignment(Qt.AlignTop)
        self.list_scroll.setWidget(self.list_widget)
        self.list_scroll.setMaximumHeight(150)
        layout.addWidget(self.list_scroll, stretch=1)
        
        self._setup_plots()
    
    def _setup_plots(self):
        """Initialize the matplotlib plots."""
        self.figure.clear()
        self.ax_red = self.figure.add_subplot(311)
        self.ax_green = self.figure.add_subplot(312)
        self.ax_blue = self.figure.add_subplot(313)
        
        for ax, color, title in [
            (self.ax_red, 'red', 'Red Channel'),
            (self.ax_green, 'green', 'Green Channel'),
            (self.ax_blue, 'blue', 'Blue Channel')
        ]:
            ax.set_title(title)
            ax.set_xlabel('Pixels')
            ax.set_ylabel('Contrast')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def add_measurement(self, measurement: Measurement):
        """Add a new measurement and update display."""
        measurement.name = f"Linecut {len(self.measurements) + 1}"
        self.measurements.append(measurement)
        self._update_plots()
        self._update_list()
    
    def remove_measurement(self, index: int):
        """Remove a measurement by index."""
        if 0 <= index < len(self.measurements):
            del self.measurements[index]
            # Update names
            for i, m in enumerate(self.measurements):
                m.name = f"Linecut {i + 1}"
            self._update_plots()
            self._update_list()
    
    def update_measurement_width(self, index: int, new_width: int):
        """Update a measurement's width (note: doesn't recalculate contrast)."""
        if 0 <= index < len(self.measurements):
            self.measurements[index].width = new_width
    
    def _update_plots(self):
        """Redraw all plots with current measurements."""
        self._setup_plots()
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        
        for i, m in enumerate(self.measurements):
            color = colors[i % len(colors)]
            alpha = 0.8
            
            if len(m.red_contrast) > 0:
                x = np.arange(len(m.red_contrast))
                self.ax_red.plot(x, m.red_contrast, color='red', alpha=alpha, 
                               label=m.name, linewidth=1.5)
                self.ax_green.plot(x, m.green_contrast, color='green', alpha=alpha,
                                 label=m.name, linewidth=1.5)
                self.ax_blue.plot(x, m.blue_contrast, color='blue', alpha=alpha,
                                label=m.name, linewidth=1.5)
        
        if self.measurements:
            self.ax_red.legend(fontsize=8)
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def _update_list(self):
        """Update the measurement list widget."""
        # Clear existing items
        while self.list_layout.count():
            item = self.list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Add items for each measurement
        for i, m in enumerate(self.measurements):
            item = MeasurementListItem(i, m.name, m.width)
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
        splitter = QSplitter(Qt.Horizontal)
        
        # Image canvas
        self.canvas = ImageCanvas()
        self.canvas.set_image(pixmap)
        self.canvas.polygon_complete.connect(self._on_polygon_complete)
        self.canvas.linecut_complete.connect(self._on_linecut_complete)
        splitter.addWidget(self.canvas)
        
        # Data display
        self.data_panel = DataDisplayPanel()
        splitter.addWidget(self.data_panel)
        
        splitter.setSizes([500, 400])
        layout.addWidget(splitter)
    
    def start_background(self):
        """Start background polygon drawing."""
        self.canvas.start_background_mode()
    
    def start_linecut(self, width: int):
        """Start linecut drawing."""
        self.canvas.set_averaging_width(width)
        self.canvas.start_linecut_mode()
    
    def _on_polygon_complete(self, points: list):
        """Handle completed background polygon."""
        self.data.background_polygon = points
        
        # Calculate average color
        mask = create_polygon_mask(self.data.pil_image.size, points)
        self.data.background_rgb = calculate_average_color(self.data.pil_image, mask)
        
        # Display RGB on canvas
        self.canvas.display_rgb_text(self.data.background_rgb)
    
    def _on_linecut_complete(self, segments: list):
        """Handle completed linecut."""
        if not self.data.background_polygon:
            # No background defined yet
            return
        
        # Calculate contrast
        red, green, blue = calculate_contrast(
            self.data.pil_image,
            segments,
            self.data.background_rgb,
            self.canvas.averaging_width
        )
        
        # Create measurement
        measurement = Measurement(
            segments=segments,
            width=self.canvas.averaging_width,
            red_contrast=red,
            green_contrast=green,
            blue_contrast=blue
        )
        
        self.data.measurements.append(measurement)
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
        
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Toolbar
        toolbar = QToolBar()
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        
        # Capture button
        self.capture_btn = QPushButton("Capture Image")
        self.capture_btn.clicked.connect(self._start_capture)
        toolbar.addWidget(self.capture_btn)
        
        toolbar.addSeparator()
        
        # Draw Background button
        self.bg_btn = QPushButton("Draw Background")
        self.bg_btn.clicked.connect(self._start_background)
        self.bg_btn.setEnabled(False)
        toolbar.addWidget(self.bg_btn)
        
        # Draw Linecut button
        self.linecut_btn = QPushButton("Draw Linecut")
        self.linecut_btn.clicked.connect(self._start_linecut)
        self.linecut_btn.setEnabled(False)
        toolbar.addWidget(self.linecut_btn)
        
        toolbar.addSeparator()
        
        # Width input
        toolbar.addWidget(QLabel("Width:"))
        self.width_input = QSpinBox()
        self.width_input.setRange(1, 100)
        self.width_input.setValue(10)
        self.width_input.setToolTip("Averaging width for linecut")
        toolbar.addWidget(self.width_input)
        
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
        self.capture_overlay = ScreenCaptureOverlay()
        self.capture_overlay.capture_complete.connect(self._on_capture_complete)
        self.capture_overlay.destroyed.connect(self._on_capture_cancelled)
        self.capture_overlay.show()
    
    def _on_capture_complete(self, pixmap: QPixmap, pil_image: Image.Image):
        """Handle completed screen capture."""
        self.show()
        
        # Create new tab
        tab = ImageTab(pixmap, pil_image)
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
        self.bg_btn.setEnabled(has_tab)
        self.linecut_btn.setEnabled(has_tab)
    
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


# =============================================================================
# Entry Point
# =============================================================================

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
