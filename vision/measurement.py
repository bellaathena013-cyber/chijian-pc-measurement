import numpy as np


def measure(points_mm, options, warnings=()):
    edges = np.linalg.norm(points_mm-np.roll(points_mm,-1,axis=0),axis=1)
    length,width = sorted(((edges[0]+edges[2])/2,(edges[1]+edges[3])/2),reverse=True)
    length_error = float(length-options.design_length_mm)
    width_error = float(width-options.design_width_mm)
    within = abs(length_error)<=options.tolerance_mm and abs(width_error)<=options.tolerance_mm
    return {"success":True, "result":"PASS" if within and not warnings else "REVIEW",
            "measured_length_mm":round(float(length),2), "measured_width_mm":round(float(width),2),
            "length_error_mm":round(length_error,2), "width_error_mm":round(width_error,2),
            "within_tolerance":bool(within),
            "unrounded_measurement_mm":{"length":float(length),"width":float(width)},
            "reason_code":"WITHIN_TOLERANCE" if within and not warnings else ("QUALITY_WARNING" if within else "OUT_OF_TOLERANCE"),
            "message":"尺寸质检通过（比赛原型）" if within and not warnings else "建议人工复核尺寸及测量条件。"}
