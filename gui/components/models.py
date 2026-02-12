"""Data model components."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, Any


@dataclass
class MapEvalConfigModel:
    """GUI configuration model aligned with map_eval.yaml."""

    estimated_map: str = ""
    ground_truth_map: str = ""
    downsample_size: float = 0.01
    icp_method: str = "generalized"
    icp_max_distance: float = 1.0
    trunc_distances: list[float] = field(default_factory=lambda: [0.2, 0.1, 0.08, 0.05, 0.01])
    evaluate_mme: bool = True
    mme_radius: float = 0.3  # Keep consistent with Cloud_Map_Evaluation (nn_radius: 0.3)
    evaluate_awd: bool = True
    awd_voxel_size: float = 3.0
    evaluate_scs: bool = True
    evaluate_chamfer: bool = True
    enable_debug: bool = False
    evaluate_gt_mme: bool = True
    evaluate_using_initial: bool = False
    initial_transform: Optional[list[list[float]]] = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        # YAML more readable: convert None to empty
        if data.get("initial_transform") is None:
            data["initial_transform"] = []
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MapEvalConfigModel":
        return cls(
            estimated_map=data.get("estimated_map", ""),
            ground_truth_map=data.get("ground_truth_map", ""),
            downsample_size=float(data.get("downsample_size", 0.01)),
            icp_method=data.get("icp_method", "generalized"),
            icp_max_distance=float(data.get("icp_max_distance", 1.0)),
            trunc_distances=list(data.get("trunc_distances", [0.2, 0.1, 0.08, 0.05, 0.01])),
            evaluate_mme=bool(data.get("evaluate_mme", True)),
            mme_radius=float(data.get("mme_radius", 0.3)),  # Default 0.3, consistent with Cloud_Map_Evaluation
            evaluate_awd=bool(data.get("evaluate_awd", data.get("evaluate_vmd", True))),  # Compatible with old configuration
            awd_voxel_size=float(data.get("awd_voxel_size", data.get("vmd_voxel_size", 3.0))),  # Compatible with old configuration
            evaluate_scs=bool(data.get("evaluate_scs", True)),
            evaluate_chamfer=bool(data.get("evaluate_chamfer", True)),
            enable_debug=bool(data.get("enable_debug", False)),
            evaluate_gt_mme=bool(data.get("evaluate_gt_mme", True)),
            evaluate_using_initial=bool(data.get("evaluate_using_initial", False)),
            initial_transform=data.get("initial_transform") or None,
        )

