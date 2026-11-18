from dataclasses import asdict
from pathlib import Path
import json
import cv2
import numpy as np
from .quality_check import Options, InspectionError, failure
from .visualization import read_image, save_image, annotate
from .marker_detector import detect_markers, check_markers
from .perspective import layout_points, calibrate, transform
from .component_detector import detect_component, validate_boundary
from .measurement import measure


def inspect(image_path, layout, options=None, output_dir="outputs/latest", manual_corners=None, selected_marker_ids=None):
    options = options or Options()
    out = Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    result = {"schema_version":"1.0", "component_id":options.component_id,
              "design_length_mm":options.design_length_mm,"design_width_mm":options.design_width_mm,
              "marker_size_mm":options.marker_size_mm,"tolerance_mm":options.tolerance_mm,
              "layout_id":layout.get("layout_id") if isinstance(layout,dict) else None,
              "measurement_mode":"manual" if manual_corners is not None else "automatic",
              "warnings":[],"images":{},"markers":[],
              "limitations":["比赛原型，未通过真实现场精度验证。", "需已知标志布局、实测边长及共面条件；未做手机镜头畸变标定。"]}
    found={}; corners=None; image=None; cal=None
    try:
        options.validate()
        world = layout_points(layout,options.marker_size_mm)
        image = read_image(image_path)
        result["image_size"]={"width":image.shape[1],"height":image.shape[0],"coordinates":"EXIF-normalized original pixels"}
        result["images"]["original"] = save_image(out/"original.png",image)
        found,metrics = detect_markers(image,layout.get("dictionary","DICT_4X4_50"))
        result["markers"]=[dict(m,corners_px=found[m["id"]].tolist()) for m in metrics]
        if selected_marker_ids is not None:
            if set(selected_marker_ids)!=set(world):
                raise InspectionError("MARKER_SELECTION_MISMATCH","所选标志必须与四标志布局一致。","INVALID_INPUT")
            found={i:p for i,p in found.items() if i in selected_marker_ids}
        result["warnings"] += check_markers(found,metrics,world,options,image.shape)
        cal = calibrate(found,world,image,options)
        result["calibration"]=cal["metrics"] | {"image_to_mm":cal["image_to_mm"].tolist(),
                                                "image_to_canvas":cal["image_to_canvas"].tolist()}
        result["images"]["corrected"] = save_image(out/"corrected.png",cal["corrected"])
        corners = np.asarray(manual_corners,np.float64) if manual_corners is not None else detect_component(cal)
        mm = validate_boundary(corners,cal,image.shape)
        result["boundary_original_px"]=corners.tolist()
        result["boundary_mm"]=mm.tolist()
        result.update(measure(mm,options,result["warnings"]))
    except InspectionError as exc:
        result.update(failure(exc))
    except (ValueError, TypeError) as exc:
        result.update(failure(InspectionError("INVALID_INPUT","参数或角点格式无效。","INVALID_INPUT")))
    if image is not None:
        valid_corners = corners if corners is not None and corners.shape==(4,2) and np.isfinite(corners).all() else None
        label=result["result"]
        if result["success"]:
            label+=f'  {result["measured_length_mm"]:.1f} x {result["measured_width_mm"]:.1f} mm'
        result["images"]["annotated"]=save_image(out/"annotated.png",annotate(image,found,valid_corners,label))
        if cal is not None:
            marker_canvas={i:transform(p,cal["mm_to_canvas"]) for i,p in cal["world_markers"].items()}
            boundary_canvas=transform(valid_corners,cal["image_to_canvas"]) if valid_corners is not None else None
            result["images"]["corrected_annotated"]=save_image(out/"corrected_annotated.png",annotate(cal["corrected"],marker_canvas,boundary_canvas,label))
    # Invalid non-finite parameters must never make the output non-standard JSON.
    def finite_json(value):
        if isinstance(value,float) and not np.isfinite(value): return None
        if isinstance(value,dict): return {k:finite_json(v) for k,v in value.items()}
        if isinstance(value,list): return [finite_json(v) for v in value]
        return value
    result=finite_json(result)
    (out/"result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    (out/"parameters.json").write_text(json.dumps(finite_json(asdict(options)),ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    return result
