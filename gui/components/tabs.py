"""Tab components."""

from __future__ import annotations

from typing import Optional
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QGroupBox,
    QTextEdit,
)

from gui.components.canvas import ThreeViewCanvas


class TrajectoryVisualizationTab(QWidget):
    """Trajectory visualization tab."""
    
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        layout = QVBoxLayout()
        
        # Three-view display area
        self.three_view_canvas = ThreeViewCanvas(self)
        self.three_view_canvas.setMinimumHeight(200)
        three_view_group = QGroupBox("Trajectory three-view comparison")
        three_view_layout = QVBoxLayout()
        three_view_layout.addWidget(self.three_view_canvas)
        three_view_group.setLayout(three_view_layout)
        
        layout.addWidget(three_view_group)
        self.setLayout(layout)
    
    def plot_trajectories(self, ref_xyz, est_xyz_list) -> None:
        """Plot three-view trajectories."""
        self.three_view_canvas.plot_trajectories(ref_xyz, est_xyz_list)


class MetricsTab(QWidget):
    """Metrics results tab."""
    
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        layout = QVBoxLayout()
        metrics_group = QGroupBox("Metrics")
        metrics_inner_layout = QVBoxLayout()
        self.text_metrics = QTextEdit()
        self.text_metrics.setReadOnly(True)
        self.text_metrics.setMinimumHeight(250)
        metrics_inner_layout.addWidget(self.text_metrics)
        metrics_group.setLayout(metrics_inner_layout)
        
        layout.addWidget(metrics_group)
        self.setLayout(layout)
    
    def append_metrics(self, text: str) -> None:
        """Append metrics text."""
        self.text_metrics.append(text)

