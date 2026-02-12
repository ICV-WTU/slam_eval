import os
from typing import Optional
import importlib.util

import numpy as np

from PyQt6 import QtWidgets, QtCore
from PyQt6.QtCore import QThread, pyqtSignal as Signal
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QCheckBox,
    QScrollArea,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QMessageBox,
    QTabWidget,
)

import matplotlib.pyplot as plt

# Configure matplotlib fonts with common CJK-safe families and a default size
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # Ensure minus signs render correctly
plt.rcParams['font.size'] = 14  # Set default font size to 14px

from evo.gui_interface import (
    load_trajectory,
    sync_if_possible,
    extract_positions_xyz,
    align_estimations_to_reference,
    compute_ape,
    compute_rpe,
    load_point_cloud,
    compute_trajectory_statistics,
    evaluate_map_quality,
    compute_map_mme,
)

try:
    import yaml
except ImportError:  # noqa: WPS440
    yaml = None  # type: ignore[assignment]

# Import GUI components
from gui.components import (
    MapEvalConfigModel,
    TrajectoryCanvas,
    ThreeViewCanvas,
    PointCloudView,
    PcdAlignDialog,
    TrajectoryControlPanel,
    PointCloudControlPanel,
    TrajectoryVisualizationTab,
    MetricsTab,
)

class MapEvaluationWorker(QThread):
    """Worker thread for map evaluation."""
    progress_signal = Signal(str)  # Progress message signal
    finished_signal = Signal(dict)  # Completion signal carrying results
    error_signal = Signal(str)  # Error signal
    
    def __init__(
        self,
        est_points: np.ndarray,
        gt_points: Optional[np.ndarray],
        config: dict,
        config_model: MapEvalConfigModel,
    ) -> None:
        super().__init__()
        self.est_points = est_points
        self.gt_points = gt_points
        self.config = config
        self.config_model = config_model
    
    def run(self) -> None:
        """Execute map evaluation inside the worker thread."""
        import time
        try:
            # If no ground-truth map is provided, only compute MME for the estimated map
            if self.gt_points is None:
                if not self.config_model.evaluate_mme:
                    self.error_signal.emit("No ground truth map provided, but selected metrics require one.")
                    return
                
                self.progress_signal.emit("Ground truth map missing, computing MME for the estimated map only...")
                mme_value, _ = compute_map_mme(self.est_points, radius=self.config_model.mme_radius)
                
                result = {
                    "type": "single_map",
                    "mme": mme_value,
                }
                self.finished_signal.emit(result)
                return
            
            # Full evaluation when ground-truth is available
            self.progress_signal.emit("Starting map evaluation...")
            
            # Run the evaluation
            start_time = time.time()
            results = evaluate_map_quality(self.est_points, self.gt_points, self.config)
            
            # Record evaluation time
            elapsed_time = time.time() - start_time
            results["evaluation_time"] = elapsed_time
            
            self.finished_signal.emit(results)
            
        except Exception as exc:  # noqa: BLE001
            self.error_signal.emit(str(exc))
    


class MainWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SLAM Evaluation Tool - Light Ver.")
        # Reduce default window height to fit 2K screens (2560x1440) without overflow.
        # Usable height on 2K screens is usually about 1300-1400 (minus task bar), so set 900px to leave headroom.
        default_width, default_height = 1400, 900
        screen = QApplication.primaryScreen()
        if screen is not None:
            available_rect = screen.availableGeometry()
            # Use conservative ratios to ensure the window stays fully visible
            max_width = max(1024, int(available_rect.width() * 0.95))
            max_height = max(600, int(available_rect.height() * 0.8))
            target_width = min(default_width, max_width)
            target_height = min(default_height, max_height)
        else:
            target_width, target_height = default_width, default_height
        
        # Force window size to avoid Qt restoring outdated geometry
        self.resize(target_width, target_height)
        
        # Apply global stylesheet (QSS) - light theme
        self.apply_light_stylesheet()

        # --- Main layout: horizontal splitter dividing left/right areas ---
        splitter_h = QSplitter(QtCore.Qt.Orientation.Horizontal)
        
        # --- Left: visualization area (70%) ---
        viz_container = QWidget()
        viz_layout = QVBoxLayout(viz_container)
        viz_layout.setContentsMargins(10, 10, 10, 10)
        viz_layout.setSpacing(10)
        
        # Top visualization: trajectory + point cloud (side by side)
        top_viz_widget = QWidget()
        top_viz_layout = QHBoxLayout(top_viz_widget)
        top_viz_layout.setContentsMargins(0, 0, 0, 0)
        top_viz_layout.setSpacing(10)
        
        # Trajectory view
        self.traj_canvas = TrajectoryCanvas(self)
        traj_group = QGroupBox("Trajectory")
        traj_vbox = QVBoxLayout()
        traj_vbox.addWidget(self.traj_canvas)
        traj_group.setLayout(traj_vbox)
        traj_group.setMinimumHeight(350)
        traj_group.setMinimumWidth(400)
        
        # Point cloud view
        self.pcd_view = PointCloudView(self)
        pcd_group = QGroupBox("Point Cloud")
        pcd_vbox = QVBoxLayout()
        pcd_vbox.addWidget(self.pcd_view)
        pcd_group.setLayout(pcd_vbox)
        pcd_group.setMinimumHeight(350)
        pcd_group.setMinimumWidth(400)
        
        top_viz_layout.addWidget(traj_group)
        top_viz_layout.addWidget(pcd_group)
        
        # Bottom tab area
        self.bottom_tabs = QTabWidget()
        # Tab 1: trajectory visualization
        self.traj_viz_tab = TrajectoryVisualizationTab(self)
        self.bottom_tabs.addTab(self.traj_viz_tab, "Trajectory three views")
        # Tab 2: metrics (trajectory + map evaluation)
        self.metrics_tab = MetricsTab(self)
        self.bottom_tabs.addTab(self.metrics_tab, "Metrics")
        
        # Vertical splitter dividing top visualization from bottom tabs
        splitter_v = QSplitter(QtCore.Qt.Orientation.Vertical)
        splitter_v.addWidget(top_viz_widget)
        splitter_v.addWidget(self.bottom_tabs)
        splitter_v.setStretchFactor(0, 6)  # Top takes 60%
        splitter_v.setStretchFactor(1, 4)  # Bottom takes 40%
        
        viz_layout.addWidget(splitter_v)
        
        # --- Right: control panel (30%) ---
        control_scroll = QScrollArea()
        control_scroll.setWidgetResizable(True)
        control_scroll.setFrameShape(QFrame.Shape.NoFrame)  # Remove frame for a cleaner look
        
        control_content = QWidget()
        control_layout = QVBoxLayout(control_content)
        control_layout.setSpacing(15)
        control_layout.setContentsMargins(15, 15, 15, 15)
        
        # Component-based trajectory control panel
        self.traj_ctrl = TrajectoryControlPanel(self)
        # Connect signals
        self.traj_ctrl.btn_est.clicked.connect(self._choose_multi_est)
        self.traj_ctrl.btn_est_remove.clicked.connect(self._remove_selected_est)
        self.traj_ctrl.btn_est_up.clicked.connect(self._move_est_up)
        self.traj_ctrl.btn_est_down.clicked.connect(self._move_est_down)
        self.traj_ctrl.btn_est_clear.clicked.connect(self._clear_multi_est)
        self.traj_ctrl.btn_plot_traj.clicked.connect(self.on_plot_traj)
        self.traj_ctrl.btn_compute_ape.clicked.connect(self.on_compute_ape)
        self.traj_ctrl.btn_compute_rpe.clicked.connect(self.on_compute_rpe)
        self.traj_ctrl.btn_compute_stats.clicked.connect(self.on_compute_stats)
        self.traj_ctrl.btn_save_plots.clicked.connect(self.on_save_plots)
        self.traj_ctrl.list_est.itemChanged.connect(self._update_est_summary)
        self.traj_ctrl.list_est.currentItemChanged.connect(self._update_est_summary)
        
        # Component-based point cloud control panel (map evaluation included)
        self.pcd_ctrl = PointCloudControlPanel(self)
        # Point cloud operation signals
        self.pcd_ctrl.btn_load_pcd.clicked.connect(self.on_load_pcd)
        self.pcd_ctrl.btn_pcd_align.clicked.connect(self.on_open_pcd_align)
        # Map evaluation signals
        self.pcd_ctrl.btn_map_est.clicked.connect(lambda: self._choose_file(self.pcd_ctrl.edit_map_est))
        self.pcd_ctrl.btn_map_gt.clicked.connect(lambda: self._choose_file(self.pcd_ctrl.edit_map_gt))
        self.pcd_ctrl.btn_map_load_cfg.clicked.connect(self.on_load_map_config)
        self.pcd_ctrl.btn_map_save_cfg.clicked.connect(self.on_save_map_config)
        self.pcd_ctrl.btn_eval_map.clicked.connect(self.on_evaluate_map)
        
        # Add panels to the control layout
        control_layout.addWidget(self.traj_ctrl)
        control_layout.addWidget(self.pcd_ctrl)
        
        # Footer label
        logo_label = QLabel("SLAM Evaluation Tool | ICV Group Ver 1.0")
        logo_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        logo_label.setStyleSheet("color: #888; font-size: 11px; margin-top: 20px;")
        control_layout.addWidget(logo_label)
        
        control_layout.addStretch()
        control_scroll.setWidget(control_content)
        control_scroll.setMinimumWidth(380)
        
        # Add both areas to the horizontal splitter
        splitter_h.addWidget(viz_container)
        splitter_h.addWidget(control_scroll)
        splitter_h.setStretchFactor(0, 7)  # Left takes 70%
        splitter_h.setStretchFactor(1, 3)  # Right takes 30%
        
        # Main layout
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.addWidget(splitter_h)
        self.setLayout(main_layout)

        # Data cache
        self.map_eval_config = MapEvalConfigModel()
        
        # Map evaluation worker thread
        self._map_eval_worker: Optional[MapEvaluationWorker] = None
        
        # Store target size so showEvent can enforce it
        self._target_width = target_width
        self._target_height = target_height

    def apply_light_stylesheet(self) -> None:
        """Define modern light theme style"""
        style = """
        QMainWindow, QScrollArea {
            background-color: #f3f3f3; /* Overall grey-white background, eye protection */
        }
        QWidget {
            color: #333333;
            font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
            font-size: 13px;
        }
        /* GroupBox Style */
        QGroupBox {
            border: 1px solid #d0d0d0;
            border-radius: 6px;
            margin-top: 20px;
            background-color: #ffffff; /* Group background pure white */
            font-weight: bold;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 5px;
            left: 10px;
            color: #005a9e; /* Title deep blue */
        }
        /* Input boxes and combo boxes */
        QLineEdit, QComboBox {
            background-color: #ffffff;
            border: 1px solid #c0c0c0;
            border-radius: 4px;
            padding: 4px;
            color: #333;
        }
        QLineEdit:focus, QComboBox:focus {
            border: 1px solid #0078d4; /* Highlight blue frame on focus */
            background-color: #ffffff;
        }
        /* General button style */
        QPushButton {
            background-color: #ffffff;
            border: 1px solid #cccccc;
            border-radius: 4px;
            padding: 6px 12px;
            color: #333;
        }
        QPushButton:hover {
            background-color: #e1e1e1;
            border-color: #adadad;
        }
        QPushButton:pressed {
            background-color: #cfcfcf;
        }
        /* Emphasis button (primary operation) */
        QPushButton#primary_btn {
            background-color: #0078d4;
            color: white;
            border: 1px solid #0078d4;
            font-weight: bold;
        }
        QPushButton#primary_btn:hover {
            background-color: #106ebe;
        }
        /* List widget */
        QListWidget {
            background-color: #ffffff;
            border: 1px solid #c0c0c0;
            border-radius: 4px;
        }
        /* Tab Widget */
        QTabWidget::pane {
            border: 1px solid #d0d0d0;
            background-color: #ffffff;
            top: -1px; 
        }
        QTabBar::tab {
            background: #e0e0e0;
            border: 1px solid #c0c0c0;
            padding: 8px 20px;
            color: #555;
            margin-right: 2px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }
        QTabBar::tab:selected {
            background: #ffffff;
            border-bottom-color: #ffffff; /* Merge with panel */
            color: #005a9e;
            font-weight: bold;
        }
        /* Splitter handle */
        QSplitter::handle {
            background-color: #d0d0d0;
        }
        """
        self.setStyleSheet(style)

    def showEvent(self, event) -> None:
        """Override showEvent to ensure correct size when window is shown."""
        super().showEvent(event)
        # Check and fix size again when window is shown
        screen = QApplication.primaryScreen()
        if screen is not None:
            available_rect = screen.availableGeometry()
            max_height = max(600, int(available_rect.height() * 0.8))
            current_height = self.height()
            if current_height > max_height:
                self.resize(self.width(), max_height)
        # Ensure window size does not exceed target size
        if self.height() > self._target_height:
            self.resize(self.width(), self._target_height)
    
    def closeEvent(self, event) -> None:
        """Clean up resources when window closes."""
        # Stop map evaluation worker thread
        if self._map_eval_worker is not None and self._map_eval_worker.isRunning():
            self._map_eval_worker.terminate()
            self._map_eval_worker.wait()
        
        # Stop progress timer
        if hasattr(self, '_progress_timer'):
            self._progress_timer.stop()
        
        super().closeEvent(event)

    def _choose_file(self, target: QLineEdit) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select File", "", "All Files (*.*)")
        if path:
            target.setText(path)
            if target is self.pcd_ctrl.edit_map_est or target is self.pcd_ctrl.edit_map_gt:
                self._update_map_eval_config_from_ui()

    def _choose_multi_est(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Select Estimated Trajectory (Multi-select)", "", "All Files (*.*)")
        if paths:
            # Get all paths in current list (use absolute path)
            existing_paths = set()
            for i in range(self.traj_ctrl.list_est.count()):
                item = self.traj_ctrl.list_est.item(i)
                if item:
                    existing_paths.add(os.path.abspath(item.data(QtCore.Qt.ItemDataRole.UserRole)))
            
            # Add new paths (deduplicate, use absolute path)
            added_count = 0
            for p in paths:
                abs_path = os.path.abspath(p)
                if abs_path not in existing_paths:
                    # Display format: filename (absolute path)
                    display_text = f"{os.path.basename(p)} ({abs_path})"
                    item = QtWidgets.QListWidgetItem(display_text)
                    item.setData(QtCore.Qt.ItemDataRole.UserRole, abs_path)
                    item.setToolTip(abs_path)  # Show full path on mouse hover
                    self.traj_ctrl.list_est.addItem(item)
                    existing_paths.add(abs_path)
                    added_count += 1
            
            if added_count == 0:
                QtWidgets.QMessageBox.information(self, "Info", "Selected files are already in the list")
            else:
                self._update_est_summary()

    def _remove_selected_est(self) -> None:
        selected_items = self.traj_ctrl.list_est.selectedItems()
        if not selected_items:
            QtWidgets.QMessageBox.information(self, "Info", "Please select items to remove first")
            return
        for item in selected_items:
            row = self.traj_ctrl.list_est.row(item)
            self.traj_ctrl.list_est.takeItem(row)
        self._update_est_summary()

    def _clear_multi_est(self) -> None:
        self.traj_ctrl.list_est.clear()
        self._update_est_summary()

    def _update_est_summary(self) -> None:
        """Update trajectory summary (Summary textbox removed in new UI, method kept empty for compatibility)."""
        # Trajectory info displayed directly in list in new UI, no separate summary box needed
        pass

    def _move_est_up(self) -> None:
        current_row = self.traj_ctrl.list_est.currentRow()
        if current_row > 0:
            item = self.traj_ctrl.list_est.takeItem(current_row)
            self.traj_ctrl.list_est.insertItem(current_row - 1, item)
            self.traj_ctrl.list_est.setCurrentRow(current_row - 1)
            self._update_est_summary()

    def _move_est_down(self) -> None:
        current_row = self.traj_ctrl.list_est.currentRow()
        if current_row >= 0 and current_row < self.traj_ctrl.list_est.count() - 1:
            item = self.traj_ctrl.list_est.takeItem(current_row)
            self.traj_ctrl.list_est.insertItem(current_row + 1, item)
            self.traj_ctrl.list_est.setCurrentRow(current_row + 1)
            self._update_est_summary()

    def _get_est_paths_from_list(self) -> list[str]:
        """Get absolute paths of all estimated trajectories from the list."""
        paths = []
        for i in range(self.traj_ctrl.list_est.count()):
            item = self.traj_ctrl.list_est.item(i)
            if item:
                path = item.data(QtCore.Qt.ItemDataRole.UserRole)
                if path:
                    paths.append(path)
        return paths

    def on_plot_traj(self) -> None:
        fmt = self.traj_ctrl.combo_fmt.currentText().lower()
        
        # Get all paths from list
        all_paths = self._get_est_paths_from_list()
        
        if len(all_paths) < 1:
            QtWidgets.QMessageBox.warning(self, "Info", "Please select at least one trajectory file first")
            return
        
        # First trajectory as reference
        ref_path = all_paths[0]
        # Remaining trajectories as estimates
        est_paths = all_paths[1:] if len(all_paths) > 1 else []
        
        try:
            # Get synchronization parameters
            try:
                t_max_diff = float(self.traj_ctrl.edit_t_max_diff.text().strip() or "0.01")
            except ValueError:
                t_max_diff = 0.01
            try:
                t_offset = float(self.traj_ctrl.edit_t_offset.text().strip() or "0.0")
            except ValueError:
                t_offset = 0.0
            skip_on_failure = True  # Skip sync failure by default
            
            # Load reference trajectory
            traj_ref = load_trajectory(ref_path, fmt)  # type: ignore[assignment]
            
            # Load all estimated trajectories
            traj_ests = []
            for p in est_paths:
                t = load_trajectory(p, fmt)
                # Attempt to sync with reference
                try:
                    t_ref, t_est = sync_if_possible(
                        traj_ref, t, t_max_diff=t_max_diff, t_offset=t_offset, skip_on_failure=skip_on_failure
                    )
                    # Use synchronized estimated trajectory
                    traj_ests.append(t_est)
                except Exception as sync_exc:
                    if skip_on_failure:
                        # Sync failed but skip allowed, use original trajectory
                        traj_ests.append(t)
                        QtWidgets.QMessageBox.warning(
                            self, "Sync Warning", f"Trajectory {os.path.basename(p)} sync failed, using original:\n{sync_exc}"
                        )
                    else:
                        raise
            # Alignment (optional)
            ests_to_plot = traj_ests
            if self.traj_ctrl.chk_align.isChecked() and len(ests_to_plot) > 0:
                try:
                    ests_to_plot, alignment_warnings = align_estimations_to_reference(
                        traj_ref, traj_ests, correct_scale=self.traj_ctrl.chk_correct_scale.isChecked(), skip_on_failure=True
                    )
                    # Collect all warning messages
                    warning_messages = []
                    for i, (warning, path) in enumerate(zip(alignment_warnings, est_paths)):
                        if warning:
                            warning_messages.append(f"{os.path.basename(path)}: {warning}")
                    
                    # If there are warnings, show to user
                    if warning_messages:
                        QtWidgets.QMessageBox.warning(
                            self, "Alignment Warning",
                            "Some trajectories have alignment issues:\n\n" + "\n".join(warning_messages) +
                            "\n\nThese trajectories might not be aligned correctly, please check data."
                        )
                except Exception as align_exc:
                    QtWidgets.QMessageBox.warning(
                        self, "Alignment Warning", f"Alignment failed, using unaligned trajectories:\n{align_exc}"
                    )
                    # Use original trajectories
                    ests_to_plot = traj_ests
            # Plot
            self.traj_canvas._ax.clear()
            # Plot reference trajectory
            ref_xyz = extract_positions_xyz(traj_ref)
            ref_label = os.path.basename(ref_path) + " (Ref)"
            self.traj_canvas._ax.plot(ref_xyz[:, 0], ref_xyz[:, 1], ref_xyz[:, 2], color="tab:blue", label=ref_label)
            # Plot estimated trajectories
            labels = [ref_label]
            est_xyz_list = []  # For three-view
            for i, est in enumerate(ests_to_plot):
                est_xyz = extract_positions_xyz(est)
                est_label = os.path.basename(est_paths[i]) if i < len(est_paths) else f"estimate_{i+1}"
                self.traj_canvas._ax.plot(est_xyz[:, 0], est_xyz[:, 1], est_xyz[:, 2], label=est_label)
                labels.append(est_label)
                est_xyz_list.append((est_xyz, est_label))
            if self.traj_canvas._ax.has_data() and labels:
                self.traj_canvas._ax.legend(labels)
            self.traj_canvas._ax.set_xlabel("X (m)")
            self.traj_canvas._ax.set_ylabel("Y (m)")
            self.traj_canvas._ax.set_zlabel("Z (m)")
            self.traj_canvas.draw()
            
            # Update three-view
            self.traj_viz_tab.plot_trajectories(ref_xyz, est_xyz_list)
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to plot trajectory:\n{exc}")

    def on_compute_ape(self) -> None:
        # Get all paths from list
        all_paths = self._get_est_paths_from_list()
        
        if len(all_paths) < 2:
            QtWidgets.QMessageBox.warning(self, "Info", "Please select at least two trajectories (first as reference, others as estimates)")
            return
        
        # First trajectory as reference
        ref_path = all_paths[0]
        # Remaining trajectories as estimates
        est_paths = all_paths[1:]
        
        fmt = self.traj_ctrl.combo_fmt.currentText().lower()
        # Get synchronization parameters
        try:
            t_max_diff = float(self.traj_ctrl.edit_t_max_diff.text().strip() or "0.01")
        except ValueError:
            t_max_diff = 0.01
        try:
            t_offset = float(self.traj_ctrl.edit_t_offset.text().strip() or "0.0")
        except ValueError:
            t_offset = 0.0
        skip_on_failure = True  # Skip sync failure by default
        
        try:
            # Load reference trajectory
            traj_ref = load_trajectory(ref_path, fmt)
            
            # Compute one by one
            for idx, path in enumerate(est_paths):
                est = load_trajectory(path, fmt)
                # Initialize synced reference trajectory (use original if sync fails)
                traj_ref_sync = traj_ref
                # Attempt to sync with reference
                try:
                    traj_ref_sync, est = sync_if_possible(
                        traj_ref, est, t_max_diff=t_max_diff, t_offset=t_offset, skip_on_failure=skip_on_failure
                    )
                except Exception as sync_exc:
                    if skip_on_failure:
                        # Sync failed but skip allowed, use original trajectory
                        QtWidgets.QMessageBox.warning(
                            self, "Sync Warning", f"Trajectory {os.path.basename(path)} sync failed, using original for calculation:\n{sync_exc}"
                        )
                        traj_ref_sync = traj_ref  # Ensure original trajectory is used
                    else:
                        raise
                # Internal alignment during computation, using defaults; enable if scale correction is selected
                # Use synchronized reference trajectory to ensure point set correspondence
                res = compute_ape(
                    traj_ref_sync,
                    est,
                    align=self.traj_ctrl.chk_align.isChecked(),
                    correct_scale=self.traj_ctrl.chk_correct_scale.isChecked(),
                )
                label = os.path.basename(path)
                self._show_metrics(f"APE[{label}]", res)
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to compute APE:\n{exc}")

    def on_compute_rpe(self) -> None:
        # Get all paths from list
        all_paths = self._get_est_paths_from_list()
        
        if len(all_paths) < 2:
            QtWidgets.QMessageBox.warning(self, "Info", "Please select at least two trajectories (first as reference, others as estimates)")
            return
        
        # First trajectory as reference
        ref_path = all_paths[0]
        # Remaining trajectories as estimates
        est_paths = all_paths[1:]
        
        fmt = self.traj_ctrl.combo_fmt.currentText().lower()
        # Get synchronization parameters
        try:
            t_max_diff = float(self.traj_ctrl.edit_t_max_diff.text().strip() or "0.01")
        except ValueError:
            t_max_diff = 0.01
        try:
            t_offset = float(self.traj_ctrl.edit_t_offset.text().strip() or "0.0")
        except ValueError:
            t_offset = 0.0
        skip_on_failure = True  # Skip sync failure by default
        
        try:
            # Load reference trajectory
            traj_ref = load_trajectory(ref_path, fmt)
            
            for idx, path in enumerate(est_paths):
                est = load_trajectory(path, fmt)
                # Initialize synced reference trajectory (use original if sync fails)
                traj_ref_sync = traj_ref
                # Attempt to sync with reference
                try:
                    traj_ref_sync, est = sync_if_possible(
                        traj_ref, est, t_max_diff=t_max_diff, t_offset=t_offset, skip_on_failure=skip_on_failure
                    )
                except Exception as sync_exc:
                    if skip_on_failure:
                        # Sync failed but skip allowed, use original trajectory
                        QtWidgets.QMessageBox.warning(
                            self, "Sync Warning", f"Trajectory {os.path.basename(path)} sync failed, using original for calculation:\n{sync_exc}"
                        )
                        traj_ref_sync = traj_ref  # Ensure original trajectory is used
                    else:
                        raise
                # Use synchronized reference trajectory to ensure point set correspondence
                res = compute_rpe(
                    traj_ref_sync,
                    est,
                    align=self.traj_ctrl.chk_align.isChecked(),
                    correct_scale=self.traj_ctrl.chk_correct_scale.isChecked(),
                )
                label = os.path.basename(path)
                self._show_metrics(f"RPE[{label}]", res)
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to compute RPE:\n{exc}")

    def _show_metrics(self, name: str, result: dict) -> None:
        stats = result.get("stats", {})
        lines = [f"[{name}] {result.get('title', '')}"]
        for k in ("rmse", "mean", "median", "std", "min", "max", "sse"):
            if k in stats:
                lines.append(f"{k}: {stats[k]}")
        self.metrics_tab.append_metrics("\n".join(lines) + "\n")

    def on_compute_stats(self) -> None:
        """Compute trajectory statistics: start/end distance and path length."""
        # Get all paths from list
        all_paths = self._get_est_paths_from_list()
        
        if len(all_paths) < 1:
            QtWidgets.QMessageBox.warning(self, "Info", "Please select at least one trajectory")
            return
        
        fmt = self.traj_ctrl.combo_fmt.currentText().lower()
        
        try:
            # Compute statistics for each trajectory
            for path in all_paths:
                traj = load_trajectory(path, fmt)
                stats = compute_trajectory_statistics(traj)
                
                if "error" in stats:
                    QtWidgets.QMessageBox.warning(self, "Warning", f"Trajectory {os.path.basename(path)}: {stats['error']}")
                    continue
                
                label = os.path.basename(path)
                lines = [f"[Trajectory Stats - {label}]"]
                lines.append(f"Start Position: ({stats['start_position'][0]:.3f}, {stats['start_position'][1]:.3f}, {stats['start_position'][2]:.3f}) m")
                lines.append(f"End Position: ({stats['end_position'][0]:.3f}, {stats['end_position'][1]:.3f}, {stats['end_position'][2]:.3f}) m")
                lines.append("")
                lines.append("Start-End Distance:")
                lines.append(f"  XY Plane: {stats['distance_xy']:.3f} m")
                lines.append(f"  YZ Plane: {stats['distance_yz']:.3f} m")
                lines.append(f"  XZ Plane: {stats['distance_xz']:.3f} m")
                lines.append(f"  3D Space: {stats['distance_3d']:.3f} m")
                lines.append("")
                lines.append(f"Path Length (Arc Length): {stats['path_length']:.3f} m")
                
                self.metrics_tab.append_metrics("\n".join(lines) + "\n\n")
                
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to compute trajectory statistics:\n{exc}")

    def on_load_pcd(self) -> None:
        path = self.pcd_ctrl.edit_map_est.text().strip()
        if not path:
            QtWidgets.QMessageBox.warning(self, "Info", "Please select a point cloud file first")
            return
        try:
            pts = load_point_cloud(path)
            if pts.shape[1] > 3:
                pts = pts[:, :3]
            # Get point size setting
            try:
                point_size = float(self.pcd_ctrl.edit_point_size.text().strip() or "3.0")
            except ValueError:
                point_size = 3.0
            self.pcd_view.set_points(pts, point_size=point_size)
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load point cloud:\n{exc}")

    def on_save_plots(self) -> None:
        """Save currently displayed plots (3D trajectory and three-view)."""
        try:
            # Select save directory
            save_dir = QFileDialog.getExistingDirectory(self, "Select Save Directory", "")
            if not save_dir:
                return
            
            # Save 3D trajectory plot
            if self.traj_canvas._ax.has_data():
                traj_path = os.path.join(save_dir, "trajectory_3d.png")
                self.traj_canvas._fig.savefig(traj_path, dpi=300, bbox_inches="tight")
            
            # Save three-view plot
            if self.traj_viz_tab.three_view_canvas._ax_xy.has_data():
                three_view_path = os.path.join(save_dir, "trajectory_three_views.png")
                self.traj_viz_tab.three_view_canvas._fig.savefig(three_view_path, dpi=300, bbox_inches="tight")
            
            # Check if at least one plot was saved
            saved_files = []
            if self.traj_canvas._ax.has_data():
                saved_files.append("trajectory_3d.png")
            if self.traj_viz_tab.three_view_canvas._ax_xy.has_data():
                saved_files.append("trajectory_three_views.png")
            
            if saved_files:
                QtWidgets.QMessageBox.information(
                    self, "Success", 
                    f"Plots saved to:\n{save_dir}\n\nSaved files:\n" + "\n".join(saved_files)
                )
            else:
                QtWidgets.QMessageBox.warning(self, "Info", "No plots to save, please plot trajectory first")
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save plots:\n{exc}")

    def on_open_pcd_align(self) -> None:
        """Open MLSD dialog."""
        dialog = PcdAlignDialog(self)
        dialog.exec()
    
    def _log_map_eval(self, message: str) -> None:
        self.metrics_tab.append_metrics(f"{message}\n")
    
    def _safe_float(self, text: str, default: float) -> float:
        try:
            return float((text or "").strip())
        except ValueError:
            return default

    def _parse_trunc_distances(self, text: str) -> list[float]:
        values: list[float] = []
        for token in text.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                values.append(float(token))
            except ValueError:
                continue
        return values
    
    def _get_icp_method_key(self) -> str:
        icp_method_map = {
            0: "point_to_point",
            1: "point_to_plane",
            2: "generalized",
        }
        return icp_method_map.get(self.pcd_ctrl.combo_icp_method.currentIndex(), "generalized")
    
    def _set_icp_method_from_key(self, key: str) -> None:
        reverse_map = {
            "point_to_point": 0,
            "point_to_plane": 1,
            "generalized": 2,
        }
        self.pcd_ctrl.combo_icp_method.setCurrentIndex(reverse_map.get(key, 2))
    
    def _collect_map_eval_config(self) -> MapEvalConfigModel:
        trunc_distances = self._parse_trunc_distances(self.pcd_ctrl.edit_trunc_distances.text() or "")
        if not trunc_distances:
            trunc_distances = [0.2, 0.1, 0.08, 0.05, 0.01]
        
        return MapEvalConfigModel(
            estimated_map=self.pcd_ctrl.edit_map_est.text().strip(),
            ground_truth_map=self.pcd_ctrl.edit_map_gt.text().strip(),
            downsample_size=self._safe_float(self.pcd_ctrl.edit_downsample.text(), 0.01),
            icp_method=self._get_icp_method_key(),
            icp_max_distance=self._safe_float(self.pcd_ctrl.edit_icp_max_distance.text(), 1.0),
            trunc_distances=trunc_distances,
            evaluate_mme=self.pcd_ctrl.chk_eval_mme.isChecked(),
            mme_radius=self._safe_float(self.pcd_ctrl.edit_mme_radius.text(), 0.3),  # Default 0.3, consistent with Cloud_Map_Evaluation
            evaluate_awd=self.pcd_ctrl.chk_eval_awd.isChecked(),
            awd_voxel_size=self._safe_float(self.pcd_ctrl.edit_awd_voxel.text(), 3.0),
            evaluate_scs=self.pcd_ctrl.chk_eval_scs.isChecked(),
            evaluate_chamfer=self.pcd_ctrl.chk_eval_chamfer.isChecked(),
            enable_debug=self.pcd_ctrl.chk_map_debug.isChecked(),
            evaluate_gt_mme=self.pcd_ctrl.chk_map_eval_gt.isChecked(),
            evaluate_using_initial=self.pcd_ctrl.chk_map_use_initial.isChecked(),
        )
    
    def _apply_map_eval_config_to_ui(self, config: MapEvalConfigModel) -> None:
        self.pcd_ctrl.edit_map_est.setText(config.estimated_map)
        self.pcd_ctrl.edit_map_gt.setText(config.ground_truth_map)
        self.pcd_ctrl.edit_downsample.setText(f"{config.downsample_size}")
        self.pcd_ctrl.edit_icp_max_distance.setText(f"{config.icp_max_distance}")
        self.pcd_ctrl.edit_trunc_distances.setText(",".join(f"{v:.3f}" for v in config.trunc_distances))
        self.pcd_ctrl.edit_mme_radius.setText(f"{config.mme_radius}")
        self.pcd_ctrl.edit_awd_voxel.setText(f"{config.awd_voxel_size}")
        self.pcd_ctrl.chk_eval_mme.setChecked(config.evaluate_mme)
        self.pcd_ctrl.chk_eval_awd.setChecked(config.evaluate_awd)
        self.pcd_ctrl.chk_eval_scs.setChecked(config.evaluate_scs)
        self.pcd_ctrl.chk_eval_chamfer.setChecked(config.evaluate_chamfer)
        self.pcd_ctrl.chk_map_debug.setChecked(config.enable_debug)
        self.pcd_ctrl.chk_map_eval_gt.setChecked(config.evaluate_gt_mme)
        self.pcd_ctrl.chk_map_use_initial.setChecked(config.evaluate_using_initial)
        self._set_icp_method_from_key(config.icp_method)
    
    def _update_map_eval_config_from_ui(self) -> None:
        self.map_eval_config = self._collect_map_eval_config()
    
    def _check_map_eval_dependencies(self, config: MapEvalConfigModel) -> bool:
        missing = []
        for module in ("open3d", "scipy"):
            if importlib.util.find_spec(module) is None:
                missing.append(module)
        if missing:
            QMessageBox.warning(
                self,
                "Missing Dependencies",
                "The following dependencies are not installed, cannot run map evaluation:\n- " + "\n- ".join(missing),
            )
            return False
        return True
    
    def on_load_map_config(self) -> None:
        if yaml is None:
            QMessageBox.warning(self, "Missing Dependencies", "PyYAML not installed, please run pip install pyyaml")
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load map_eval config",
            "",
            "YAML Files (*.yaml *.yml);;All Files (*.*)",
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            self.map_eval_config = MapEvalConfigModel.from_dict(data)
            self._apply_map_eval_config_to_ui(self.map_eval_config)
            self.metrics_tab.append_metrics(f"Config loaded: {path}\n")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Error", f"Failed to load config:\n{exc}")
    
    def on_save_map_config(self) -> None:
        if yaml is None:
            QMessageBox.warning(self, "Missing Dependencies", "PyYAML not installed, please run pip install pyyaml")
            return
        self._update_map_eval_config_from_ui()
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save map_eval config",
            "map_eval_config.yaml",
            "YAML Files (*.yaml *.yml);;All Files (*.*)",
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump(self.map_eval_config.to_dict(), f, allow_unicode=True, sort_keys=False)
            self.metrics_tab.append_metrics(f"Config saved: {path}\n")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Error", f"Failed to save config:\n{exc}")
    
    def on_evaluate_map(self) -> None:
        """Execute map evaluation (using multi-threading)."""
        # If evaluation is already running, stop first
        if self._map_eval_worker is not None and self._map_eval_worker.isRunning():
            QMessageBox.warning(self, "Info", "Map evaluation is in progress, please wait")
            return
        
        config_model = self._collect_map_eval_config()
        self.map_eval_config = config_model
        
        if not config_model.estimated_map:
            QMessageBox.warning(self, "Info", "Please select estimated map file first")
            return
        
        if not self._check_map_eval_dependencies(config_model):
            return
        
        # Check metric config before loading point clouds
        # If no GT map, only compute MME for estimated map
        if not config_model.ground_truth_map:
            if not config_model.evaluate_mme:
                QMessageBox.warning(self, "Info", "No ground truth map provided, but selected metrics require one")
                return
        
        # If GT map provided, check if at least one metric is selected
        if config_model.ground_truth_map:
            if not (config_model.evaluate_mme or config_model.evaluate_awd or config_model.evaluate_chamfer):
                QMessageBox.warning(self, "Info", "Please select at least one map evaluation metric")
                return
            
            # SCS depends on AWD. If SCS selected but AWD not, warn user
            if config_model.evaluate_scs and not config_model.evaluate_awd:
                QMessageBox.warning(
                    self, "Info", 
                    "SCS depends on AWD results.\n"
                    "Please check 'Evaluate AWD' or uncheck 'Evaluate SCS'."
                )
                return
        
        try:
            # Disable evaluation button
            self.pcd_ctrl.btn_eval_map.setEnabled(False)
            self.pcd_ctrl.btn_eval_map.setText("Evaluating...")
            
            # Load point clouds
            self.metrics_tab.append_metrics("Loading point clouds...\n")
            est_points = load_point_cloud(config_model.estimated_map)
            if est_points.shape[1] > 3:
                est_points = est_points[:, :3]
            self.metrics_tab.append_metrics(f"Estimated map points: {len(est_points)}\n")
            
            # Prepare ground truth map (if any)
            gt_points = None
            if config_model.ground_truth_map:
                gt_points = load_point_cloud(config_model.ground_truth_map)
                if gt_points.shape[1] > 3:
                    gt_points = gt_points[:, :3]
                self.metrics_tab.append_metrics(f"Ground truth map points: {len(gt_points)}\n\n")
            
            # Prepare config
            config = {
                "downsample_size": config_model.downsample_size,
                "icp_method": config_model.icp_method,
                "icp_max_distance": config_model.icp_max_distance,
                "trunc_distances": config_model.trunc_distances,
                "evaluate_mme": config_model.evaluate_mme,
                "mme_radius": config_model.mme_radius,
                "evaluate_vmd": config_model.evaluate_awd,  # Internally use vmd for compatibility
                "vmd_voxel_size": config_model.awd_voxel_size,  # Internally use vmd for compatibility
                "evaluate_scs": config_model.evaluate_scs,
                "evaluate_chamfer": config_model.evaluate_chamfer,
                "enable_debug": config_model.enable_debug,
                "evaluate_gt_mme": config_model.evaluate_gt_mme,
                "evaluate_using_initial": config_model.evaluate_using_initial,
            }
            
            if config_model.enable_debug:
                self._log_map_eval("Current Config: " + str(config_model.to_dict()))
            
            # Create and start worker thread
            self._map_eval_worker = MapEvaluationWorker(est_points, gt_points, config, config_model)
            self._map_eval_worker.progress_signal.connect(self._on_map_eval_progress)
            self._map_eval_worker.finished_signal.connect(self._on_map_eval_finished)
            self._map_eval_worker.error_signal.connect(self._on_map_eval_error)
            self._map_eval_worker.start()
            
            # Start progress update timer
            self._progress_timer = QtCore.QTimer()
            self._progress_timer.timeout.connect(self._update_progress_message)
            self._progress_timer.start(2000)  # Update progress every 2 seconds
            self._progress_start_time = QtCore.QDateTime.currentDateTime()
            
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Error", f"Failed to start map evaluation:\n{exc}")
            self.metrics_tab.append_metrics(f"\n[Error] Failed to start map evaluation: {exc}\n\n")
            self.pcd_ctrl.btn_eval_map.setEnabled(True)
            self.pcd_ctrl.btn_eval_map.setText("Start Map Evaluation")
    
    def _on_map_eval_progress(self, message: str) -> None:
        """Handle map evaluation progress messages."""
        self.metrics_tab.append_metrics(f"{message}\n")
    
    def _update_progress_message(self) -> None:
        """Update progress message periodically."""
        if self._map_eval_worker is None or not self._map_eval_worker.isRunning():
            if hasattr(self, '_progress_timer'):
                self._progress_timer.stop()
            return
        
        elapsed = self._progress_start_time.secsTo(QtCore.QDateTime.currentDateTime())
        minutes = elapsed // 60
        seconds = elapsed % 60
        self.metrics_tab.append_metrics(f"Evaluation in progress... Elapsed: {minutes}m {seconds}s\n")
    
    def _on_map_eval_finished(self, results: dict) -> None:
        """Handle map evaluation completion."""
        # Stop progress timer
        if hasattr(self, '_progress_timer'):
            self._progress_timer.stop()
        
        # Re-enable button
        self.pcd_ctrl.btn_eval_map.setEnabled(True)
        self.pcd_ctrl.btn_eval_map.setText("Start Map Evaluation")
        
        # Show results
        if results.get("type") == "single_map":
            # Single map evaluation results
            self.metrics_tab.append_metrics("=" * 60 + "\n")
            self.metrics_tab.append_metrics("[Map Evaluation Results (Single Map)]\n")
            self.metrics_tab.append_metrics("=" * 60 + "\n\n")
            self.metrics_tab.append_metrics("[Mean Map Entropy (MME)]\n")
            self.metrics_tab.append_metrics(f"  Estimated Map MME: {results['mme']:.6f}\n")
            self.metrics_tab.append_metrics("  No ground truth map provided, skipping comparison\n")
            self.metrics_tab.append_metrics("=" * 60 + "\n")
            self.metrics_tab.append_metrics("Map evaluation completed!\n\n")
        else:
            # Full evaluation results
            self.metrics_tab.append_metrics("\n" + "="*60 + "\n")
            self.metrics_tab.append_metrics("[Map Evaluation Results]\n")
            self.metrics_tab.append_metrics("="*60 + "\n\n")
            
            # Show evaluation time
            if "evaluation_time" in results:
                elapsed = results["evaluation_time"]
                minutes = int(elapsed // 60)
                seconds = int(elapsed % 60)
                self.metrics_tab.append_metrics(f"Evaluation Time: {minutes}m {seconds}s\n\n")
            
            # MME Results
            if "mme" in results:
                mme = results["mme"]
                self.metrics_tab.append_metrics("[Mean Map Entropy (MME)]\n")
                self.metrics_tab.append_metrics(f"  Estimated Map MME: {mme['estimated']:.6f}\n")
                self.metrics_tab.append_metrics(f"  Ground Truth Map MME: {mme['ground_truth']:.6f}\n")
                self.metrics_tab.append_metrics("\n")
            
            # Registration Results
            if "registration" in results:
                reg = results["registration"]
                icp_method = self.map_eval_config.icp_method
                self.metrics_tab.append_metrics("[ICP Registration Results]\n")
                self.metrics_tab.append_metrics(f"  Method: {icp_method}\n")
                self.metrics_tab.append_metrics(f"  Fitness: {reg['fitness']:.6f}\n")
                self.metrics_tab.append_metrics(f"  Inlier RMSE: {reg['inlier_rmse']:.6f}\n")
                
                if "metrics" in reg:
                    metrics = reg["metrics"]
                    self.metrics_tab.append_metrics("\n  Accuracy Metrics (at trunc distances):\n")
                    trunc_dists = self.map_eval_config.trunc_distances
                    for i, dist in enumerate(trunc_dists):
                        if i < len(metrics["rmse"]):
                            self.metrics_tab.append_metrics(
                                f"    Trunc Dist {dist:.2f}m: "
                                f"RMSE={metrics['rmse'][i]:.6f}, "
                                f"Mean={metrics['mean'][i]:.6f}, "
                                f"Std={metrics['std'][i]:.6f}, "
                                f"Fitness={metrics['fitness'][i]:.6f}\n"
                            )
                self.metrics_tab.append_metrics("\n")
            
            # Chamfer Distance
            if "chamfer_distance" in results:
                cd = results["chamfer_distance"]
                self.metrics_tab.append_metrics("[Chamfer Distance]\n")
                self.metrics_tab.append_metrics(f"  Chamfer Distance: {cd:.6f}\n")
                self.metrics_tab.append_metrics("\n")
            
            # AWD Results
            if "vmd" in results:
                vmd = results["vmd"]
                self.metrics_tab.append_metrics("[Average Wasserstein Distance (AWD)]\n")
                self.metrics_tab.append_metrics(f"  Mean AWD: {vmd['mean_vmd']:.6f}\n")
                self.metrics_tab.append_metrics(f"  Valid Voxels: {len(vmd['wasserstein_distances'])}\n")
                self.metrics_tab.append_metrics("\n")
            
            # SCS Results
            if "scs" in results:
                scs = results["scs"]
                self.metrics_tab.append_metrics("[Spatial Consistency Score (SCS)]\n")
                self.metrics_tab.append_metrics(f"  SCS: {scs:.6f}\n")
                self.metrics_tab.append_metrics("\n")
            
            self.metrics_tab.append_metrics("="*60 + "\n")
            self.metrics_tab.append_metrics("Map evaluation completed!\n\n")
        
        # Cleanup worker thread
        self._map_eval_worker = None
    
    def _on_map_eval_error(self, error_msg: str) -> None:
        """Handle map evaluation error."""
        # Stop progress timer
        if hasattr(self, '_progress_timer'):
            self._progress_timer.stop()
        
        # Re-enable button
        self.pcd_ctrl.btn_eval_map.setEnabled(True)
        self.pcd_ctrl.btn_eval_map.setText("Start Map Evaluation")
        
        # Show error
        QMessageBox.critical(self, "Error", f"Map evaluation failed:\n{error_msg}")
        self.metrics_tab.append_metrics(f"\n[Error] Map evaluation failed: {error_msg}\n\n")
        
        # Cleanup worker thread
        self._map_eval_worker = None