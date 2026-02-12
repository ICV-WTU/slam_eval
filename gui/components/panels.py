"""Control panel components."""

from __future__ import annotations

from typing import Optional
from PyQt6.QtWidgets import (
    QGroupBox,
    QGridLayout,
    QVBoxLayout,
    QHBoxLayout,
    QComboBox,
    QLineEdit,
    QListWidget,
    QPushButton,
    QCheckBox,
    QLabel,
    QWidget,
)
from PyQt6 import QtCore, QtWidgets


class TrajectoryControlPanel(QGroupBox):
    """Control panel for trajectory operations."""
    
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("Trajectory operations", parent)
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        layout = QVBoxLayout()
        
        # Format selection and load button
        h_layout = QHBoxLayout()
        self.combo_fmt = QComboBox()
        self.combo_fmt.addItems(["TUM", "KITTI", "EuroC"])
        self.btn_est = QPushButton("Select trajectory...")
        h_layout.addWidget(QLabel("Format:"))
        h_layout.addWidget(self.combo_fmt)
        h_layout.addWidget(self.btn_est)
        layout.addLayout(h_layout)
        
        # Trajectory list
        self.list_est = QListWidget()
        self.list_est.setFixedHeight(100)
        self.list_est.setAlternatingRowColors(True)  # Enable alternating row colors
        self.list_est.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.list_est.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        layout.addWidget(self.list_est)
        
        # List operation buttons
        btn_box = QHBoxLayout()
        self.btn_est_remove = QPushButton("Delete selected")
        self.btn_est_up = QPushButton("Move up")
        self.btn_est_down = QPushButton("Move down")
        self.btn_est_clear = QPushButton("Clear")
        for btn in [self.btn_est_remove, self.btn_est_up, self.btn_est_down, self.btn_est_clear]:
            btn.setStyleSheet("padding: 4px 8px; font-size: 12px;")
            btn_box.addWidget(btn)
        layout.addLayout(btn_box)
        
        # Parameter settings
        from PyQt6.QtWidgets import QFormLayout
        form = QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        self.edit_t_max_diff = QLineEdit("0.01")
        self.edit_t_max_diff.setToolTip("Maximum time difference threshold (seconds) for trajectory synchronization.")
        self.edit_t_offset = QLineEdit("0.0")
        self.edit_t_offset.setToolTip("Time offset (seconds) for trajectory synchronization.")
        form.addRow("Time difference threshold (s):", self.edit_t_max_diff)
        form.addRow("Time offset (s):", self.edit_t_offset)
        layout.addLayout(form)
        
        # Function button grid
        grid = QGridLayout()
        self.btn_plot_traj = QPushButton("Plot trajectories")
        self.btn_compute_ape = QPushButton("Compute APE")
        self.btn_compute_rpe = QPushButton("Compute RPE")
        self.btn_compute_stats = QPushButton("Compute trajectory statistics")
        self.btn_save_plots = QPushButton("Save plots")
        self.chk_align = QCheckBox("Align to reference")
        self.chk_align.setChecked(True)
        self.chk_correct_scale = QCheckBox("Scale correction")
        self.chk_correct_scale.setChecked(False)
        
        grid.addWidget(self.btn_plot_traj, 0, 0)
        grid.addWidget(self.chk_align, 0, 1)
        grid.addWidget(self.btn_compute_ape, 1, 0)
        grid.addWidget(self.chk_correct_scale, 1, 1)
        grid.addWidget(self.btn_compute_rpe, 2, 0)
        grid.addWidget(self.btn_compute_stats, 2, 1)
        grid.addWidget(self.btn_save_plots, 3, 0, 1, 2)
        
        self.btn_save_plots.setObjectName("primary_btn")

        layout.addLayout(grid)
        
        self.setLayout(layout)


class PointCloudControlPanel(QGroupBox):
    """Control panel for point cloud operations (including map evaluation)."""
    
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("Point cloud operations", parent)
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        layout = QVBoxLayout()
        
        # ========== Point cloud loading section ==========
        from PyQt6.QtWidgets import QFormLayout
        form = QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        
        # Estimated map
        est_layout = QHBoxLayout()
        self.edit_map_est = QLineEdit()
        self.edit_map_est.setReadOnly(True)
        self.btn_map_est = QPushButton("...")
        self.btn_map_est.setFixedWidth(30)
        est_layout.addWidget(self.edit_map_est)
        est_layout.addWidget(self.btn_map_est)
        form.addRow("Estimated map:", est_layout)
        
        # Ground-truth map
        gt_layout = QHBoxLayout()
        self.edit_map_gt = QLineEdit()
        self.edit_map_gt.setReadOnly(True)
        self.btn_map_gt = QPushButton("...")
        self.btn_map_gt.setFixedWidth(30)
        gt_layout.addWidget(self.edit_map_gt)
        gt_layout.addWidget(self.btn_map_gt)
        form.addRow("Ground truth map:", gt_layout)
        layout.addLayout(form)
        
        # Parameter grid
        grid_params = QGridLayout()
        self.edit_point_size = QLineEdit("3.0")
        self.edit_point_size.setToolTip("Point size for point cloud display (in pixels).")
        self.edit_icp_max_distance = QLineEdit("1.0")
        self.edit_downsample = QLineEdit("0.01")
        grid_params.addWidget(QLabel("Point size:"), 0, 0)
        grid_params.addWidget(self.edit_point_size, 0, 1)
        grid_params.addWidget(QLabel("Downsample size:"), 2, 0)
        grid_params.addWidget(self.edit_downsample, 2, 1)
        layout.addLayout(grid_params)
        
        # Point cloud operation buttons
        btn_pcd_layout = QHBoxLayout()
        self.btn_load_pcd = QPushButton("Load point cloud")
        self.btn_pcd_align = QPushButton("Evaluate MLSD")
        self.btn_pcd_align.setObjectName("primary_btn")
        btn_pcd_layout.addWidget(self.btn_load_pcd)
        btn_pcd_layout.addWidget(self.btn_pcd_align)
        layout.addLayout(btn_pcd_layout)
        
        # Evaluation option checkboxes
        check_grid = QGridLayout()
        self.chk_eval_mme = QCheckBox("Evaluate MME")
        self.chk_eval_mme.setChecked(False)
        self.chk_eval_awd = QCheckBox("Evaluate AWD")
        self.chk_eval_awd.setChecked(False)
        self.chk_eval_scs = QCheckBox("Evaluate SCS")
        self.chk_eval_scs.setChecked(True)
        # Tooltip will be updated dynamically based on AWD state in _update_awd_visibility
        self.chk_eval_chamfer = QCheckBox("Evaluate Chamfer")
        self.chk_eval_chamfer.setChecked(False)
        self.chk_map_debug = QCheckBox("Enable debug logging")
        self.chk_map_use_initial = QCheckBox("Skip ICP registration (use initial transform)")
        self.chk_map_use_initial.setToolTip(
            "Enabled: skip ICP registration and directly use the initial transform (identity) "
            "to evaluate map quality.\n"
            "Disabled: perform ICP registration to align point clouds before evaluation.\n"
            "Use when: the point clouds are already aligned or you want to test metrics without registration."
        )
        self.chk_map_use_initial.setChecked(True)
        self.chk_map_eval_gt = QCheckBox("Compute ground truth MME")
        self.chk_map_eval_gt.setChecked(True)
        
        checks = [
            self.chk_eval_mme, self.chk_eval_awd,
            self.chk_eval_scs, self.chk_eval_chamfer,
            self.chk_map_debug, self.chk_map_use_initial
        ]
        for i, cb in enumerate(checks):
            check_grid.addWidget(cb, i // 2, i % 2)
        layout.addLayout(check_grid)
        
        # ICP method selection (can be hidden)
        self.icp_method_widget = QWidget()
        icp_layout = QHBoxLayout(self.icp_method_widget)
        icp_layout.setContentsMargins(0, 0, 0, 0)
        icp_layout.addWidget(QLabel("ICP method:"))
        self.combo_icp_method = QComboBox()
        self.combo_icp_method.addItems(["Point-to-point", "Point-to-plane", "Generalized ICP"])
        icp_layout.addWidget(self.combo_icp_method)
        icp_layout.addWidget(QLabel("ICP max distance:"))
        icp_layout.addWidget(self.edit_icp_max_distance)
        layout.addWidget(self.icp_method_widget)
        
        # Other parameters (MME radius, AWD voxel size, truncation distances)
        param_layout = QGridLayout()
        
        # MME radius (can be hidden)
        self.mme_radius_label = QLabel("MME radius:")
        self.edit_mme_radius = QLineEdit("0.3")  # Default 0.3, consistent with Cloud_Map_Evaluation
        param_layout.addWidget(self.mme_radius_label, 0, 0)
        param_layout.addWidget(self.edit_mme_radius, 0, 1)
        
        # AWD voxel size (can be hidden)
        self.awd_voxel_label = QLabel("AWD voxel size:")
        self.edit_awd_voxel = QLineEdit("3.0")
        param_layout.addWidget(self.awd_voxel_label, 1, 0)
        param_layout.addWidget(self.edit_awd_voxel, 1, 1)
        
        # Truncation distances (can be hidden)
        self.trunc_distances_label = QLabel("Precision thresholds:")
        self.edit_trunc_distances = QLineEdit("0.2,0.1,0.08,0.05,0.01")
        self.edit_trunc_distances.setToolTip(
            "Precision thresholds as a 5D vector; the first value is used most prominently.\n"
            "If there are very few inliers, try increasing these values, e.g. for outdoor scenes: "
            "[0.5, 0.3, 0.2, 0.1, 0.05]."
        )
        param_layout.addWidget(self.trunc_distances_label, 2, 0)
        param_layout.addWidget(self.edit_trunc_distances, 2, 1)
        
        layout.addLayout(param_layout)
        
        # Connect signals to update visibility
        self._setup_visibility_connections()
        
        # Configuration management buttons
        btn_config_layout = QHBoxLayout()
        self.btn_map_load_cfg = QPushButton("Load configuration")
        self.btn_map_save_cfg = QPushButton("Save configuration")
        btn_config_layout.addWidget(self.btn_map_load_cfg)
        btn_config_layout.addWidget(self.btn_map_save_cfg)
        layout.addLayout(btn_config_layout)
        
        # Evaluation button (main operation button)
        self.btn_eval_map = QPushButton("Start map evaluation (Start)")
        self.btn_eval_map.setObjectName("primary_btn")
        self.btn_eval_map.setMinimumHeight(40)
        self.btn_eval_map.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        layout.addWidget(self.btn_eval_map)
        
        self.setLayout(layout)
    
    def _setup_visibility_connections(self) -> None:
        """Set up signal connections that control widget visibility."""
        # ICP method: hide when "skip ICP registration" is checked
        self.chk_map_use_initial.toggled.connect(self._update_icp_method_visibility)
        # MME radius: hide when "Evaluate MME" is unchecked
        self.chk_eval_mme.toggled.connect(self._update_mme_radius_visibility)
        # AWD-related: hide when "Evaluate AWD" is unchecked
        self.chk_eval_awd.toggled.connect(self._update_awd_visibility)
        
        # Initialize visibility
        self._update_icp_method_visibility(self.chk_map_use_initial.isChecked())
        self._update_mme_radius_visibility(self.chk_eval_mme.isChecked())
        self._update_awd_visibility(self.chk_eval_awd.isChecked())
    
    def _update_icp_method_visibility(self, use_initial: bool) -> None:
        """Update visibility of ICP method selection widgets."""
        # Hide ICP method selection when "skip ICP registration" is checked
        self.icp_method_widget.setVisible(not use_initial)
    
    def _update_mme_radius_visibility(self, eval_mme: bool) -> None:
        """Update visibility of the MME radius input."""
        # Hide MME radius when "Evaluate MME" is unchecked
        self.mme_radius_label.setVisible(eval_mme)
        self.edit_mme_radius.setVisible(eval_mme)
    
    def _update_awd_visibility(self, eval_awd: bool) -> None:
        """Update visibility of AWD-related widgets."""
        # When "Evaluate AWD" is unchecked, hide AWD voxel size and truncation distances
        self.awd_voxel_label.setVisible(eval_awd)
        self.edit_awd_voxel.setVisible(eval_awd)
        self.trunc_distances_label.setVisible(eval_awd)
        self.edit_trunc_distances.setVisible(eval_awd)
        # SCS depends on AWD; when AWD is unchecked, disable SCS
        self.chk_eval_scs.setEnabled(eval_awd)
        # If AWD is unchecked while SCS is still checked, uncheck SCS automatically
        if not eval_awd and self.chk_eval_scs.isChecked():
            self.chk_eval_scs.setChecked(False)
        # Update SCS tooltip according to AWD state
        if eval_awd:
            self.chk_eval_scs.setToolTip(
                "Spatial Consistency Score (SCS)\n"
                "Computed based on AWD results to evaluate consistency of errors between neighboring voxels.\n"
                "Smaller values indicate better spatial consistency."
            )
        else:
            self.chk_eval_scs.setToolTip(
                "Spatial Consistency Score (SCS)\n"
                "Please enable \"Evaluate AWD\" first; SCS is based on AWD results.\n"
                "SCS evaluates spatial consistency of errors between neighboring voxels."
            )

