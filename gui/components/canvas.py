"""Canvas components."""

from __future__ import annotations

from typing import Optional
import numpy as np

from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QMouseEvent
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pyqtgraph.opengl as gl


class TrajectoryCanvas(FigureCanvas):
    """3D trajectory canvas."""
    
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        self._fig = Figure(figsize=(5, 4))
        super().__init__(self._fig)
        self.setParent(parent)
        self._ax = self._fig.add_subplot(111, projection="3d")
        self._ax.set_xlabel("X (m)")
        self._ax.set_ylabel("Y (m)")
        self._ax.set_zlabel("Z (m)")

    def plot_traj(self, ref_xyz: Optional[np.ndarray], est_xyz: Optional[np.ndarray]) -> None:
        self._ax.clear()
        self._ax.set_xlabel("X (m)")
        self._ax.set_ylabel("Y (m)")
        self._ax.set_zlabel("Z (m)")
        if ref_xyz is not None and len(ref_xyz) > 0:
            self._ax.plot(ref_xyz[:, 0], ref_xyz[:, 1], ref_xyz[:, 2], color="tab:blue", label="reference")
        if est_xyz is not None and len(est_xyz) > 0:
            self._ax.plot(est_xyz[:, 0], est_xyz[:, 1], est_xyz[:, 2], color="tab:orange", label="estimate")
        if self._ax.has_data():
            self._ax.legend()
        self.draw()


class ThreeViewCanvas(FigureCanvas):
    """Three-view canvas: projections on XY, YZ and XZ planes."""
    
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        # Reduce figure height to make the canvas more compact
        self._fig = Figure(figsize=(12, 6))
        super().__init__(self._fig)
        self.setParent(parent)
        # Set a smaller minimum height to better fit 2K screens
        self.setMinimumHeight(200)
        # Create three subplots: XY, YZ, XZ
        self._ax_xy = self._fig.add_subplot(131)
        self._ax_yz = self._fig.add_subplot(132)
        self._ax_xz = self._fig.add_subplot(133)
        self._fig.tight_layout(pad=3.0)
        
    def plot_trajectories(self, ref_xyz: Optional[np.ndarray], est_xyz_list: list[tuple[np.ndarray, str]]) -> None:
        """Plot three orthogonal views of trajectories.

        Args:
            ref_xyz: XYZ coordinates of the reference trajectory (N, 3)
            est_xyz_list: list of estimated trajectories, each as (xyz, label)
        """
        # Clear all subplots
        self._ax_xy.clear()
        self._ax_yz.clear()
        self._ax_xz.clear()
        
        has_data = False
        
        # Plot reference trajectory
        if ref_xyz is not None and len(ref_xyz) > 0:
            self._ax_xy.plot(ref_xyz[:, 0], ref_xyz[:, 1], color="tab:blue", label="reference", linewidth=1.5)
            self._ax_yz.plot(ref_xyz[:, 1], ref_xyz[:, 2], color="tab:blue", label="reference", linewidth=1.5)
            self._ax_xz.plot(ref_xyz[:, 0], ref_xyz[:, 2], color="tab:blue", label="reference", linewidth=1.5)
            has_data = True
        
        # Plot estimated trajectories
        colors = ["tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink"]
        for i, (est_xyz, label) in enumerate(est_xyz_list):
            if est_xyz is not None and len(est_xyz) > 0:
                color = colors[i % len(colors)]
                self._ax_xy.plot(est_xyz[:, 0], est_xyz[:, 1], color=color, label=label, linewidth=1.2, alpha=0.8)
                self._ax_yz.plot(est_xyz[:, 1], est_xyz[:, 2], color=color, label=label, linewidth=1.2, alpha=0.8)
                self._ax_xz.plot(est_xyz[:, 0], est_xyz[:, 2], color=color, label=label, linewidth=1.2, alpha=0.8)
                has_data = True
        
        # Set labels and titles
        self._ax_xy.set_xlabel("X (m)")
        self._ax_xy.set_ylabel("Y (m)")
        self._ax_xy.set_title("XY view")
        self._ax_xy.grid(True, alpha=0.3)
        if has_data:
            self._ax_xy.axis("equal")
            self._ax_xy.legend(fontsize=12, loc="best")
        
        self._ax_yz.set_xlabel("Y (m)")
        self._ax_yz.set_ylabel("Z (m)")
        self._ax_yz.set_title("YZ view")
        self._ax_yz.grid(True, alpha=0.3)
        if has_data:
            self._ax_yz.axis("equal")
            self._ax_yz.legend(fontsize=12, loc="best")
        
        self._ax_xz.set_xlabel("X (m)")
        self._ax_xz.set_ylabel("Z (m)")
        self._ax_xz.set_title("XZ view")
        self._ax_xz.grid(True, alpha=0.3)
        if has_data:
            self._ax_xz.axis("equal")
            self._ax_xz.legend(fontsize=12, loc="best")
        
        self.draw()


class PointCloudView(gl.GLViewWidget):
    """3D point cloud view."""
    
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent=parent)
        self.opts["distance"] = 10
        grid = gl.GLGridItem()
        grid.scale(1, 1, 1)
        self.addItem(grid)
        self._scatter: Optional[gl.GLScatterPlotItem] = None
        
        # State for right-button panning
        self._panning = False
        self._last_pan_pos = None

    def set_points(self, points: np.ndarray, point_size: float = 3.0) -> None:
        """Set point cloud data.

        Args:
            points: point cloud coordinates (N, 3)
            point_size: point size (default 3.0)
        """
        if self._scatter is not None:
            self.removeItem(self._scatter)
            self._scatter = None
        if points is None or len(points) == 0:
            return
        pos = points.astype(np.float32)
        size = np.full((pos.shape[0],), float(point_size), dtype=np.float32)
        color = np.tile(np.array([1.0, 1.0, 1.0, 0.8], dtype=np.float32), (pos.shape[0], 1))
        self._scatter = gl.GLScatterPlotItem(pos=pos, size=size, color=color, pxMode=False)
        self.addItem(self._scatter)
    
    def mousePressEvent(self, ev: QMouseEvent) -> None:
        """Handle mouse press events, detect right button to start panning."""
        if ev.button() == Qt.MouseButton.RightButton:
            self._panning = True
            self._last_pan_pos = ev.position()
            ev.accept()
        else:
            # Other buttons (e.g. left-button rotation) are handled by the parent
            super().mousePressEvent(ev)
    
    def mouseMoveEvent(self, ev: QMouseEvent) -> None:
        """Handle mouse move events, implementing right-button panning."""
        if self._panning and self._last_pan_pos is not None:
            # Compute mouse movement offset
            current_pos = ev.position()
            dx = current_pos.x() - self._last_pan_pos.x()
            dy = current_pos.y() - self._last_pan_pos.y()
            
            # Get current camera parameters (from opts dict)
            distance = self.opts.get('distance', 10.0)
            azimuth = self.opts.get('azimuth', 0.0)
            elevation = self.opts.get('elevation', 0.0)
            center = self.opts.get('center', [0.0, 0.0, 0.0])
            
            # Compute translation vector.
            # Translation speed is proportional to camera distance so that panning feels natural.
            pan_speed = distance * 0.001  # Tweak pan speed factor
            
            # Compute translation in camera coordinates.
            # In screen space, x is camera-right and y is camera-up.
            import math
            elev_rad = math.radians(elevation)
            azim_rad = math.radians(azimuth)
            
            # Compute camera right and up vectors (in camera coordinates)
            # Right vector: perpendicular to view direction in XY plane
            right_x = -math.sin(azim_rad)
            right_y = math.cos(azim_rad)
            right_z = 0.0
            
            # Up vector: perpendicular to view direction and right vector
            up_x = -math.cos(azim_rad) * math.sin(elev_rad)
            up_y = -math.sin(azim_rad) * math.sin(elev_rad)
            up_z = math.cos(elev_rad)
            
            # Compute translation vector (world coordinates)
            pan_x = (right_x * dx - up_x * dy) * pan_speed
            pan_y = (right_y * dx - up_y * dy) * pan_speed
            pan_z = (right_z * dx - up_z * dy) * pan_speed
            
            # Update camera center position
            # center may be QVector3D or list, need to handle
            if hasattr(center, 'x'):
                # QVector3D对象
                from PyQt6.QtGui import QVector3D
                new_center = QVector3D(
                    center.x() + pan_x,
                    center.y() + pan_y,
                    center.z() + pan_z
                )
            else:
                # List or tuple
                new_center = [
                    center[0] + pan_x,
                    center[1] + pan_y,
                    center[2] + pan_z
                ]
            
            # Update center in opts dictionary
            self.opts['center'] = new_center
            self.update()  # Trigger redraw
            
            self._last_pan_pos = current_pos
            ev.accept()
        else:
            # Non-panning state, pass to parent class (e.g. left-button rotation)
            super().mouseMoveEvent(ev)
    
    def mouseReleaseEvent(self, ev: QMouseEvent) -> None:
        """Handle mouse release event, end panning."""
        if ev.button() == Qt.MouseButton.RightButton:
            self._panning = False
            self._last_pan_pos = None
            ev.accept()
        else:
            super().mouseReleaseEvent(ev)

