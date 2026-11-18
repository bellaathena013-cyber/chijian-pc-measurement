from dataclasses import dataclass
import math


class InspectionError(ValueError):
    def __init__(self, code, message, result="RETAKE"):
        super().__init__(message)
        self.code, self.result = code, result


@dataclass(frozen=True)
class Options:
    design_length_mm: float = 4000
    design_width_mm: float = 1000
    marker_size_mm: float = 100
    tolerance_mm: float = 50
    component_id: str = "PC-W01"
    rectified_px_per_mm: float = 0.5
    min_marker_edge_px: float = 28
    max_reprojection_px: float = 3
    max_holdout_error_mm: float = 25

    def validate(self):
        values = (self.design_length_mm, self.design_width_mm, self.marker_size_mm,
                  self.rectified_px_per_mm, self.min_marker_edge_px,
                  self.max_reprojection_px, self.max_holdout_error_mm)
        if any(not math.isfinite(x) or x <= 0 for x in values):
            raise InspectionError("INVALID_PARAMETERS", "尺寸、标志和质量参数必须是有限正数。", "INVALID_INPUT")
        if not math.isfinite(self.tolerance_mm) or self.tolerance_mm < 0:
            raise InspectionError("INVALID_TOLERANCE", "允许偏差必须是有限非负数。", "INVALID_INPUT")
        if self.design_length_mm < self.design_width_mm:
            raise InspectionError("DIMENSION_ORDER", "length应为长边，width应为短边。", "INVALID_INPUT")


def failure(exc):
    return {"success": False, "result": exc.result, "reason_code": exc.code,
            "message": str(exc), "measured_length_mm": None, "measured_width_mm": None,
            "length_error_mm": None, "width_error_mm": None,
            "next_actions": (["manual_corners", "adjust_boundary", "upload_again"]
                             if exc.result == "MANUAL_REQUIRED"
                             else ["check_parameters_or_layout", "upload_again"])}
