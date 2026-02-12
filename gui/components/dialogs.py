"""Dialog components."""

from __future__ import annotations

import os
import json
from typing import Optional
import numpy as np

from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QComboBox,
    QWidget,
)
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from evo.gui_interface import (
    load_point_cloud,
    load_image,
    compute_pcd_transform,
    apply_pcd_transform,
    color_points_by_height,
    compute_mlsd,
)


class PcdAlignDialog(QDialog):
    """Dialog for aligning point clouds with satellite imagery."""
    
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("MLSD evaluation")
        self.resize(1400, 600)
        
        # Data
        self.pcd_points: Optional[np.ndarray] = None
        self.sat_image: Optional[np.ndarray] = None
        self.picked_sat: list[list[float]] = []  # [(u, v), ...]
        self.picked_pcd: list[list[float]] = []  # [(x, y), ...]
        self.transform_matrix: Optional[np.ndarray] = None
        self.method = "affine"  # "affine" or "homo"
        self.click_state = "sat"  # "sat" or "pcd"
        self.mlsd_value: Optional[float] = None
        # Store point cloud display data (for redrawing)
        self.pcd_xy_display: Optional[np.ndarray] = None
        self.pcd_colors_display: Optional[np.ndarray] = None
        self.pcd_yaxis_inverted = False  # Whether y-axis should be inverted (default: not inverted)
        # Cached cropped overlay image
        self.overlay_cropped: Optional[np.ndarray] = None
        self.overlay_crop_bounds: Optional[tuple] = None  # (u_min, v_min, u_max, v_max)
        # Cached deformation visualization data
        self.deformation_vis_data: Optional[dict] = None  # {uv, dev, H, W, max_dev}
        
        # Layout
        main_layout = QVBoxLayout()
        
        # Control bar
        ctrl_layout = QHBoxLayout()
        self.btn_load_sat = QPushButton("Load satellite image")
        self.btn_load_sat.clicked.connect(self._load_satellite)
        self.btn_load_sat.setObjectName("primary_btn")
        self.btn_load_pcd = QPushButton("Load point cloud")
        self.btn_load_pcd.clicked.connect(self._load_pointcloud)
        self.btn_load_pcd.setObjectName("primary_btn")
        self.combo_method = QComboBox()
        self.combo_method.addItems(["Affine transform (3+ pairs)", "Homography (4+ pairs)"])
        self.combo_method.currentIndexChanged.connect(self._change_method)
        self.btn_compute = QPushButton("Compute transform")
        self.btn_compute.clicked.connect(self._compute_transform)
        self.btn_compute.setObjectName("primary_btn")
        self.btn_clear = QPushButton("Clear point pairs")
        self.btn_clear.clicked.connect(self._clear_pairs)
        self.btn_save_points = QPushButton("Save point pairs")
        self.btn_save_points.clicked.connect(self._save_points)
        self.btn_load_points = QPushButton("Load point pairs")
        self.btn_load_points.clicked.connect(self._load_points)
        self.btn_save = QPushButton("Save results")
        self.btn_save.clicked.connect(self._save_result)
        self.label_status = QLabel("Status: please load satellite image and point cloud first")
        self.label_mlsd = QLabel("MLSD: -")
        
        ctrl_layout.addWidget(self.btn_load_sat)
        ctrl_layout.addWidget(self.btn_load_pcd)
        ctrl_layout.addWidget(QLabel("Transformation method:"))
        ctrl_layout.addWidget(self.combo_method)
        ctrl_layout.addWidget(self.btn_compute)
        ctrl_layout.addWidget(self.btn_clear)
        ctrl_layout.addWidget(self.btn_save_points)
        ctrl_layout.addWidget(self.btn_load_points)
        ctrl_layout.addWidget(self.btn_save)
        ctrl_layout.addWidget(self.label_status)
        ctrl_layout.addWidget(self.label_mlsd)
        ctrl_layout.addStretch()
        
        # Canvas area
        canvas_layout = QHBoxLayout()
        
        # Satellite image canvas and toolbar
        sat_widget = QWidget()
        sat_widget_layout = QVBoxLayout()
        self.sat_canvas = FigureCanvas(Figure(figsize=(6, 6)))
        self.sat_ax = self.sat_canvas.figure.add_subplot(111)
        self.sat_ax.set_title("Satellite image (click here first)")
        self.sat_ax.set_axis_off()
        self.sat_canvas.mpl_connect("button_press_event", self._on_sat_click)
        self.sat_toolbar = NavigationToolbar(self.sat_canvas, self)
        self.sat_toolbar.setMaximumHeight(30)
        sat_widget_layout.addWidget(self.sat_toolbar)
        sat_widget_layout.addWidget(self.sat_canvas)
        sat_widget.setLayout(sat_widget_layout)
        
        # Top-view point cloud canvas and toolbar
        pcd_widget = QWidget()
        pcd_widget_layout = QVBoxLayout()
        self.pcd_canvas = FigureCanvas(Figure(figsize=(6, 6)))
        self.pcd_ax = self.pcd_canvas.figure.add_subplot(111)
        self.pcd_ax.set_title("Top view of point cloud (click here next)")
        # self.pcd_ax.set_aspect("equal")
        # Don't invert y-axis, keep normal point cloud top-view direction
        self.pcd_canvas.mpl_connect("button_press_event", self._on_pcd_click)
        self.pcd_toolbar = NavigationToolbar(self.pcd_canvas, self)
        self.pcd_toolbar.setMaximumHeight(30)
        pcd_widget_layout.addWidget(self.pcd_toolbar)
        pcd_widget_layout.addWidget(self.pcd_canvas)
        pcd_widget.setLayout(pcd_widget_layout)
        
        # Overlay result canvas and title
        overlay_widget = QWidget()
        overlay_widget_layout = QVBoxLayout()
        overlay_title = QLabel("Overlay result")
        overlay_title.setFixedHeight(30)
        overlay_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        overlay_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.overlay_canvas = FigureCanvas(Figure(figsize=(6, 6)))
        self.overlay_ax = self.overlay_canvas.figure.add_subplot(111)
        # Don't set title in figure to avoid affecting resolution
        self.overlay_ax.set_axis_off()
        overlay_widget_layout.addWidget(overlay_title)
        overlay_widget_layout.addWidget(self.overlay_canvas)
        overlay_widget.setLayout(overlay_widget_layout)
        
        canvas_layout.addWidget(sat_widget)
        canvas_layout.addWidget(pcd_widget)
        canvas_layout.addWidget(overlay_widget)
        
        main_layout.addLayout(ctrl_layout)
        main_layout.addLayout(canvas_layout)
        self.setLayout(main_layout)
    
    def _load_satellite(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select satellite image",
            "",
            "Image files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All files (*.*)",
        )
        if path:
            try:
                self.sat_image = load_image(path)
                self._redraw_satellite()
                self._update_status()
            except Exception as exc:
                QMessageBox.critical(self, "Error", f"Failed to load satellite image:\n{exc}")
    
    def _redraw_satellite(self) -> None:
        """Redraw satellite image and picked points."""
        # Save current view limits
        if self.sat_image is not None and len(self.sat_ax.get_images()) > 0:
            xlim = self.sat_ax.get_xlim()
            ylim = self.sat_ax.get_ylim()
        else:
            xlim = ylim = None
        
        self.sat_ax.clear()
        if self.sat_image is not None:
            self.sat_ax.imshow(self.sat_image)
            # Restore view limits
            if xlim is not None and ylim is not None:
                self.sat_ax.set_xlim(xlim)
                self.sat_ax.set_ylim(ylim)
        self.sat_ax.set_title("Satellite image (click here first)")
        self.sat_ax.set_axis_off()
        # Redraw picked points
        if self.picked_sat:
            sat_pts = np.array(self.picked_sat)
            self.sat_ax.scatter(sat_pts[:, 0], sat_pts[:, 1], s=100, c="yellow", 
                              marker="x", linewidths=2, zorder=10)
        self.sat_canvas.draw()
    
    def _load_pointcloud(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select point cloud file", "", 
            "Point cloud files (*.pcd *.ply *.npy *.npz);;All files (*.*)"
        )
        if path:
            try:
                points = load_point_cloud(path)
                if points.shape[1] > 3:
                    points = points[:, :3]
                self.pcd_points = points
                
                # Extract XY and Z
                xy = points[:, :2]
                z = points[:, 2]
                colors = color_points_by_height(z)
                
                # Limit displayed points
                max_pts = 200000
                if len(xy) > max_pts:
                    idx = np.random.choice(len(xy), max_pts, replace=False)
                    xy = xy[idx]
                    colors = colors[idx]
                
                # Save display data
                self.pcd_xy_display = xy
                self.pcd_colors_display = colors
                
                self._redraw_pointcloud(reset_view=True)  # Reset view when loading new point cloud
                self._update_status()
            except Exception as exc:
                QMessageBox.critical(self, "Error", f"Failed to load point cloud:\n{exc}")
    
    def _redraw_pointcloud(self, reset_view: bool = False) -> None:
        """Redraw point cloud and markers
        
        Args:
            reset_view: If True, reset view range to fit new data; if False, keep current view range
        """
        # Save current view range (only when not resetting)
        if not reset_view:
            if len(self.pcd_ax.collections) > 0 or len(self.pcd_ax.patches) > 0:
                xlim = self.pcd_ax.get_xlim()
                ylim = self.pcd_ax.get_ylim()
            else:
                xlim = ylim = None
        else:
            xlim = ylim = None
        
        self.pcd_ax.clear()
        if self.pcd_xy_display is not None and self.pcd_colors_display is not None:
            self.pcd_ax.scatter(self.pcd_xy_display[:, 0], self.pcd_xy_display[:, 1], 
                              s=1, c=self.pcd_colors_display / 255.0, marker=".")
        self.pcd_ax.set_aspect("equal")
        self.pcd_ax.set_title("Point cloud top-view (click here again)")
        # Restore y-axis inversion state (using flag to ensure consistency)
        if self.pcd_yaxis_inverted:
            self.pcd_ax.invert_yaxis()
        # Restore view range (must be after setting aspect and invert)
        # If reset_view is True, let matplotlib automatically adjust view range
        if not reset_view and xlim is not None and ylim is not None:
            self.pcd_ax.set_xlim(xlim)
            self.pcd_ax.set_ylim(ylim)
        # Redraw markers
        if self.picked_pcd:
            pcd_pts = np.array(self.picked_pcd)
            self.pcd_ax.scatter(pcd_pts[:, 0], pcd_pts[:, 1], s=100, c="yellow",
                              marker="x", linewidths=2, zorder=10)
        self.pcd_canvas.draw()
    
    def _change_method(self, index: int) -> None:
        self.method = "homo" if index == 1 else "affine"
        self._update_status()
    
    def _on_sat_click(self, event) -> None:
        if event.inaxes != self.sat_ax or self.sat_image is None:
            return
        # Check if toolbar is in zoom/pan mode (to avoid accidental clicks)
        if self.sat_toolbar.mode != '':
            return
        # Only handle left-click
        if event.button != 1:
            return
        if self.click_state == "sat":
            u, v = event.xdata, event.ydata
            if u is None or v is None:  # Skip if coordinates are invalid
                return
            self.picked_sat.append([u, v])
            self.click_state = "pcd"
            self._update_markers()
    
    def _on_pcd_click(self, event) -> None:
        if event.inaxes != self.pcd_ax or self.pcd_points is None:
            return
        # Check if toolbar is in zoom/pan mode (to avoid accidental clicks)
        if self.pcd_toolbar.mode != '':
            return
        # Only handle left-click
        if event.button != 1:
            return
        if self.click_state == "pcd":
            x, y = event.xdata, event.ydata
            if x is None or y is None:  # Skip if coordinates are invalid
                return
            self.picked_pcd.append([x, y])
            self.click_state = "sat"
            self._update_markers()
    
    def _update_markers(self) -> None:
        # Redraw satellite image and markers
        self._redraw_satellite()
        # Redraw point cloud and markers
        self._redraw_pointcloud()
        self._update_status()
    
    def _update_status(self) -> None:
        status = f"Status: {len(self.picked_sat)} point pairs selected | Method: {self.method}"
        if self.sat_image is None:
            status += " | Please load satellite image"
        if self.pcd_points is None:
            status += " | Please load point cloud"
        if self.click_state == "pcd":
            status += " | Please click corresponding points on the point cloud plot"
        else:
            status += " | Please click on the satellite image"
        self.label_status.setText(status)
    
    def _clear_pairs(self) -> None:
        self.picked_sat.clear()
        self.picked_pcd.clear()
        self.click_state = "sat"
        self.transform_matrix = None
        self.mlsd_value = None
        self.label_mlsd.setText("MLSD: -")
        # Reset clipping related variables
        self.overlay_cropped = None
        self.overlay_crop_bounds = None
        # Redraw
        self._redraw_satellite()
        self._redraw_pointcloud()
        self.overlay_ax.clear()
        # Don't set title in figure, title is displayed in GUI
        self.overlay_ax.set_axis_off()
        self.overlay_canvas.draw()
        self._update_status()
    
    def _save_points(self) -> None:
        """Save point pairs to JSON file"""
        if len(self.picked_sat) == 0:
            QMessageBox.warning(self, "Warning", "No point pairs to save")
            return
        
        path, _ = QFileDialog.getSaveFileName(
            self, "Save point pairs", "", 
            "JSON files (*.json);;All files (*.*)"
        )
        if not path:
            return
        
        try:
            data = {
                "picked_sat": self.picked_sat,
                "picked_pcd": self.picked_pcd,
                "method": self.method,
                "num_pairs": len(self.picked_sat)
            }
            
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            QMessageBox.information(
                self, "Save successful", 
                f"Point pairs saved to:\n{path}\n\nTotal {len(self.picked_sat)} point pairs"
            )
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to save point pairs:\n{exc}")
    
    def _load_points(self) -> None:
        """Load point pairs from JSON file"""
        if self.sat_image is None or self.pcd_points is None:
            QMessageBox.warning(
                self, "Warning", 
                "Please load satellite image and point cloud first, then import point pairs"
            )
            return
        
        path, _ = QFileDialog.getOpenFileName(
            self, "Import point pairs", "", 
            "JSON files (*.json);;All files (*.*)"
        )
        if not path:
            return
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Validate data format
            if not isinstance(data, dict):
                raise ValueError("JSON file format error: root object must be a dictionary")
            
            if "picked_sat" not in data or "picked_pcd" not in data:
                raise ValueError("JSON file format error: missing required fields")
            
            picked_sat = data["picked_sat"]
            picked_pcd = data["picked_pcd"]
            
            if not isinstance(picked_sat, list) or not isinstance(picked_pcd, list):
                raise ValueError("JSON file format error: point pairs must be a list")
            
            if len(picked_sat) != len(picked_pcd):
                raise ValueError("JSON file format error: satellite image points and point cloud points do not match")
            
            if len(picked_sat) < 3:
                raise ValueError("At least 3 point pairs are required")
            
            # Validate each point pair format
            for i, (sat_pt, pcd_pt) in enumerate(zip(picked_sat, picked_pcd)):
                if not isinstance(sat_pt, list) or not isinstance(pcd_pt, list):
                    raise ValueError(f"Point pair {i+1} format error: must be a list")
                if len(sat_pt) != 2 or len(pcd_pt) != 2:
                    raise ValueError(f"Point pair {i+1} format error: must contain 2 coordinate values")
            
            # Import point pairs
            self.picked_sat = picked_sat
            self.picked_pcd = picked_pcd
            
            # Import method (if exists)
            if "method" in data:
                method = data["method"]
                if method == "homo":
                    self.combo_method.setCurrentIndex(1)
                else:
                    self.combo_method.setCurrentIndex(0)
                self.method = method
            
            # Reset transformation matrix (needs to be recalculated)
            self.transform_matrix = None
            self.mlsd_value = None
            self.label_mlsd.setText("MLSD: -")
            self.overlay_cropped = None
            self.overlay_crop_bounds = None
            
            # Update display
            self.click_state = "sat"  # 重置点击状态
            self._update_markers()
            
            QMessageBox.information(
                self, "Import successful", 
                f"Successfully imported {len(picked_sat)} point pairs\n\n"
                f"Method: {self.method}\n"
                f"Please click 'Compute transform' button to calculate transformation matrix"
            )
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to import point pairs:\n{exc}")
    
    def _compute_transform(self) -> None:
        if len(self.picked_sat) < 3:
            QMessageBox.warning(self, "Warning", "At least 3 point pairs are required to compute transformation")
            return
        if self.method == "homo" and len(self.picked_sat) < 4:
            QMessageBox.warning(self, "Warning", "Homography transformation requires at least 4 point pairs")
            return
        
        try:
            src_pts = np.array(self.picked_pcd, dtype=np.float32)
            dst_pts = np.array(self.picked_sat, dtype=np.float32)
            
            self.transform_matrix, inliers = compute_pcd_transform(
                src_pts, dst_pts, self.method
            )
            
            if self.transform_matrix is None:
                QMessageBox.warning(self, "Error", "Transformation calculation failed, please try selecting more dispersed points")
                return
            
            num_inliers = int(inliers.sum()) if inliers is not None else len(src_pts)
            
            # Calculate MLSD metric
            try:
                self.mlsd_value = compute_mlsd(
                    self.transform_matrix,
                    self.method,
                    src_pts,
                    num_samples=1000
                )
                self.label_mlsd.setText(f"MLSD: {self.mlsd_value:.6f}")
            except Exception as exc:
                self.mlsd_value = None
                self.label_mlsd.setText(f"MLSD: calculation failed ({exc})")
            
            QMessageBox.information(
                self, "Success", 
                f"Transformation calculation successful!\nMethod: {self.method}\nInliers: {num_inliers}/{len(src_pts)}\n"
                f"MLSD: {self.mlsd_value:.6f}" if self.mlsd_value is not None else "MLSD: calculation failed"
            )
            
            # Generate overlay image
            self._generate_overlay()
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to compute transformation:\n{exc}")
    
    def _generate_overlay(self) -> None:
        if self.transform_matrix is None or self.pcd_points is None or self.sat_image is None:
            return
        
        try:
            xy = self.pcd_points[:, :2]
            z = self.pcd_points[:, 2]
            colors = color_points_by_height(z)
            
            # Apply transformation
            uv = apply_pcd_transform(xy, self.transform_matrix, self.method).astype(int)

            # Calculate local deformation for each point (same as MLSD: remove overall scale, only preserve anisotropy)
            deformation_values = self._compute_point_deformation(
                xy, self.transform_matrix, self.method
            )
            
            # Create overlay image
            overlay = self.sat_image.copy()
            H, W = overlay.shape[:2]
            
            # Draw point cloud (limit point count to improve performance)
            step = max(1, int(len(uv) / 500000))
            alpha = 0.4
            valid_uv = []
            valid_dev = []
            for (u, v), c, dev in zip(uv[::step], colors[::step], deformation_values[::step]):
                if 0 <= u < W and 0 <= v < H:
                    overlay[v, u] = (1 - alpha) * overlay[v, u] + alpha * c
                    valid_uv.append([u, v])
                    valid_dev.append(dev)
            
            # Calculate bounding box of point cloud coverage area (for clipping)
            if len(valid_uv) > 0:
                valid_uv = np.array(valid_uv)
                valid_dev = np.array(valid_dev)
                # Add margin (5% of image size)
                padding = int(min(H, W) * 0.05)
                u_min = max(0, int(valid_uv[:, 0].min()) - padding)
                u_max = min(W, int(valid_uv[:, 0].max()) + padding)
                v_min = max(0, int(valid_uv[:, 1].min()) - padding)
                v_max = min(H, int(valid_uv[:, 1].max()) + padding)
                
                # Crop image
                overlay_cropped = overlay[v_min:v_max, u_min:u_max]
                # Save cropped boundaries
                self.overlay_crop_bounds = (u_min, v_min, u_max, v_max)
            else:
                # If no valid points, don't crop
                overlay_cropped = overlay
                self.overlay_crop_bounds = None
            
            # Save cropped image for subsequent saving
            self.overlay_cropped = overlay_cropped
            # Save deformation visualization data (for generating deformation color map)
            if len(valid_uv) > 0 and len(valid_dev) > 0:
                self.deformation_vis_data = {
                    "uv": valid_uv,
                    "dev": valid_dev,
                    "H": H,
                    "W": W,
                    "max_dev": float(np.nanmax(valid_dev)) if np.any(np.isfinite(valid_dev)) else 0.0,
                }
            else:
                self.deformation_vis_data = None
            
            # Display overlay result
            self.overlay_ax.clear()
            self.overlay_ax.imshow(overlay_cropped)
            # Don't set title in figure, title is displayed in GUI
            self.overlay_ax.set_axis_off()
            self.overlay_canvas.draw()
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to generate overlay image:\n{exc}")
    
    def _compute_point_deformation(
        self,
        points_xy: np.ndarray,
        transform_matrix: np.ndarray,
        method: str,
    ) -> np.ndarray:
        """
        Calculate local deformation for each point (same as MLSD: remove overall scale, only preserve anisotropy)
        Return deviation array corresponding to points_xy, if cannot be calculated then return nan.
        """
        n = len(points_xy)
        if n == 0:
            return np.array([])

        dev = np.full(n, np.nan, dtype=float)

        if method == "affine":
            J = transform_matrix[:2, :2]
            try:
                J_inv = np.linalg.inv(J)
                sigma = np.linalg.svd(J_inv, compute_uv=False)
                sigma1, sigma2 = sigma[0], sigma[1]
                scale = np.sqrt(abs(sigma1 * sigma2))
                if scale > 1e-9:
                    sigma1_norm = sigma1 / scale
                    sigma2_norm = sigma2 / scale
                    deviation = max(abs(sigma1_norm - 1.0), abs(sigma2_norm - 1.0))
                else:
                    deviation = 1e6
                dev[:] = deviation
            except np.linalg.LinAlgError:
                dev[:] = np.nan
        else:  # homography
            H = transform_matrix
            h11, h12, h13 = H[0, 0], H[0, 1], H[0, 2]
            h21, h22, h23 = H[1, 0], H[1, 1], H[1, 2]
            h31, h32, h33 = H[2, 0], H[2, 1], H[2, 2]

            for idx, (x, y) in enumerate(points_xy):
                p_homogeneous = np.array([x, y, 1.0])
                p_transformed = H @ p_homogeneous
                w = p_transformed[2]
                if abs(w) < 1e-9:
                    continue

                numerator_u = h11 * x + h12 * y + h13
                numerator_v = h21 * x + h22 * y + h23

                du_dx = (h11 * w - numerator_u * h31) / (w * w)
                du_dy = (h12 * w - numerator_u * h32) / (w * w)
                dv_dx = (h21 * w - numerator_v * h31) / (w * w)
                dv_dy = (h22 * w - numerator_v * h32) / (w * w)

                J = np.array([[du_dx, du_dy], [dv_dx, dv_dy]])
                try:
                    J_inv = np.linalg.inv(J)
                    sigma = np.linalg.svd(J_inv, compute_uv=False)
                    sigma1, sigma2 = sigma[0], sigma[1]
                    scale = np.sqrt(abs(sigma1 * sigma2))
                    if scale > 1e-9:
                        sigma1_norm = sigma1 / scale
                        sigma2_norm = sigma2 / scale
                        deviation = max(abs(sigma1_norm - 1.0), abs(sigma2_norm - 1.0))
                    else:
                        deviation = 1e6
                    dev[idx] = deviation
                except np.linalg.LinAlgError:
                    continue

        return dev

    def _save_result(self) -> None:
        if self.transform_matrix is None:
            QMessageBox.warning(self, "Warning", "Please compute transformation and generate overlay result first")
            return
        
        save_dir = QFileDialog.getExistingDirectory(self, "Select save directory", "")
        if not save_dir:
            return
        
        try:
            # Save overlay image (300dpi, high resolution and no white edges)
            overlay_path = os.path.join(save_dir, "overlay_result.png")
            self.overlay_canvas.figure.savefig(
                overlay_path,
                dpi=300,
                bbox_inches="tight",
                pad_inches=0,
            )
            
            # Save transformation matrix
            matrix_path = os.path.join(save_dir, f"transform_{self.method}.npy")
            np.save(matrix_path, self.transform_matrix)
            
            # Save MLSD metric
            mlsd_path = os.path.join(save_dir, "mlsd_metric.txt")
            with open(mlsd_path, "w", encoding="utf-8") as f:
                if self.mlsd_value is not None:
                    f.write(f"MLSD (Map Local Scale Deviation): {self.mlsd_value:.6f}\n")
                else:
                    f.write("MLSD (Map Local Scale Deviation): calculation failed\n")
                f.write(f"Transformation method: {self.method}\n")
                f.write(f"Calculation formula: MLSD = max_{{p∈Ω}} (max(|σ₁(J_p⁻¹) - 1|, |σ₂(J_p⁻¹) - 1|))\n")
                f.write(f"Where J_p is the Jacobian matrix of the transformation at point p, σ₁ and σ₂ are the maximum and minimum singular values respectively.\n")
            
            save_msg = f"Result saved to:\n{save_dir}\n\n- overlay_result.png\n- transform_{self.method}.npy\n- mlsd_metric.txt"
            if self.mlsd_value is not None:
                save_msg += f"\n\nMLSD: {self.mlsd_value:.6f}"

            # Save deformation color map (if exists)
            if self.deformation_vis_data is not None:
                try:
                    from matplotlib import pyplot as plt
                    from matplotlib import colors as mcolors

                    data = self.deformation_vis_data
                    uv = data["uv"]
                    dev = data["dev"]
                    max_dev = data.get("max_dev", float(np.nanmax(dev)))

                    # Filter valid values
                    mask = np.isfinite(dev)
                    uv_valid = uv[mask]
                    dev_valid = dev[mask]

                    if len(dev_valid) > 0:
                        # Adapt image scale based on aspect ratio of point cloud in image, avoid color bar being too high
                        u_span = float(uv_valid[:, 0].max() - uv_valid[:, 0].min())
                        v_span = float(uv_valid[:, 1].max() - uv_valid[:, 1].min())
                        if u_span <= 0 or v_span <= 0:
                            fig_w, fig_h = 8.0, 8.0
                        else:
                            aspect = v_span / u_span
                            base = 6.0
                            fig_w = base
                            fig_h = base * aspect
                            # Limit height to avoid extreme long bar
                            fig_h = max(4.0, min(10.0, fig_h))

                        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
                        norm = mcolors.Normalize(vmin=0, vmax=max_dev if max_dev > 0 else 1.0)
                        sc = ax.scatter(uv_valid[:, 0], uv_valid[:, 1], s=1, c=dev_valid,
                                        cmap="plasma", norm=norm, alpha=0.7)
                        ax.set_aspect("equal")
                        ax.set_xlabel("u (px)")
                        ax.set_ylabel("v (px)")
                        ax.grid(True, alpha=0.3)
                        title_max = f"{max_dev:.6f}" if np.isfinite(max_dev) else "nan"
                        ax.set_title(
                            f"Deformation heatmap (method: {self.method})\n"
                            f"max deviation: {title_max}",
                            fontsize=11,
                        )
                        # Make color bar height follow current coordinate axis, not the entire rectangular canvas
                        cbar = plt.colorbar(sc, ax=ax, fraction=0.05, pad=0.02)
                        cbar.set_label("Local deformation (anisotropy)", fontsize=10)
                        deform_path = os.path.join(save_dir, "overlay_deformation.png")
                        plt.savefig(deform_path, dpi=300, bbox_inches="tight", pad_inches=0)
                        plt.close(fig)
                        save_msg += f"\n- overlay_deformation.png"
                except Exception as exc:
                    save_msg += f"\n- overlay_deformation.png (Failed to save: {exc})"
            
            QMessageBox.information(self, "Save successful", save_msg)
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Save failed:\n{exc}")

