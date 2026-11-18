import cv2
import numpy as np
from .quality_check import InspectionError


def detect_markers(image, dictionary_name="DICT_4X4_50"):
    if dictionary_name != "DICT_4X4_50":
        raise InspectionError("UNSUPPORTED_DICTIONARY", "首轮仅支持DICT_4X4_50标准字典。", "INVALID_INPUT")
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.minMarkerPerimeterRate = .008
    detector = cv2.aruco.ArucoDetector(dictionary, params)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return {}, []
    flat = ids.ravel().tolist()
    if len(flat) != len(set(flat)):
        raise InspectionError("DUPLICATE_MARKER_ID", "画面中有重复ID，请使用四个不同ID标志。")
    found = {int(i): c.reshape(4, 2).astype(np.float64) for i, c in zip(flat, corners)}
    metrics = []
    for i, pts in found.items():
        edges = np.linalg.norm(pts - np.roll(pts, -1, axis=0), axis=1)
        x, y, w, h = cv2.boundingRect(pts.astype(np.float32))
        crop = gray[max(0, y):y+h, max(0, x):x+w]
        metrics.append({"id": i, "min_edge_px": float(edges.min()),
                        "max_edge_px": float(edges.max()),
                        "edge_ratio": float(edges.max()/edges.min()),
                        "sharpness": float(cv2.Laplacian(crop, cv2.CV_64F).var())})
    return found, metrics


def check_markers(found, metrics, required, options, shape):
    missing = sorted(set(required) - found.keys())
    if missing:
        raise InspectionError("MISSING_MARKERS", f"缺少标志ID {missing}，标定不完整，不输出尺寸。")
    warnings = []
    for m in metrics:
        if m["id"] not in required:
            continue
        pts = found[m["id"]]
        if (pts < 4).any() or (pts[:, 0] > shape[1]-5).any() or (pts[:, 1] > shape[0]-5).any():
            raise InspectionError("CLIPPED_MARKER", "标志太靠近画面边缘，请留出余量。")
        if m["min_edge_px"] < options.min_marker_edge_px:
            raise InspectionError("SMALL_MARKERS", "标志像素太少，请放大标志或调整拍摄距离。")
        if m["edge_ratio"] > 4:
            raise InspectionError("EXTREME_PERSPECTIVE", "标志透视压缩过大，请减小拍摄倾角。")
        if m["sharpness"] < 80:
            raise InspectionError("BLURRED_MARKERS", "标志不够清晰，请对焦后重拍。")
        if m["min_edge_px"] < 45 or m["sharpness"] < 180:
            warnings.append("标志较小或偏模糊，本次结果需人工复核。")
    return sorted(set(warnings))
