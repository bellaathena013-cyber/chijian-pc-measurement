import cv2
import numpy as np
from .quality_check import InspectionError


def transform(points, matrix):
    pts = np.asarray(points, np.float64)
    hp = np.column_stack((pts.reshape(-1, 2), np.ones(len(pts.reshape(-1, 2))))) @ matrix.T
    if not np.isfinite(hp).all() or np.min(np.abs(hp[:, 2])) < 1e-9:
        raise InspectionError("UNSTABLE_TRANSFORM", "透视映射不稳定，请重新拍摄。")
    return (hp[:, :2]/hp[:, 2:]).reshape(pts.shape)


def layout_points(layout, marker_size_mm):
    try:
        markers = layout["markers"]
        if len(markers) != 4:
            raise ValueError()
        world = {}
        for marker in markers:
            i = marker["id"]
            if isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < 50 or i in world:
                raise ValueError()
            c = np.asarray(marker["center_mm"], np.float64)
            a = float(marker.get("rotation_deg", 0)) * np.pi/180
            if c.shape != (2,) or not np.isfinite(c).all() or not np.isfinite(a):
                raise ValueError()
            r = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
            unit = np.array([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]])
            world[i] = (unit * marker_size_mm) @ r.T + c
        centers = np.array([p.mean(axis=0) for p in world.values()], np.float32)
        hull = cv2.convexHull(centers).reshape(-1, 2)
        if len(hull) != 4 or cv2.contourArea(hull) < marker_size_mm**2 * 16:
            raise ValueError()
        for i, a in enumerate(centers):
            for b in centers[i+1:]:
                if np.linalg.norm(a-b) < marker_size_mm*1.6:
                    raise ValueError()
        return world
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise InspectionError("INVALID_LAYOUT", "布局须包含四个不同ID、有效中心坐标及方向，且分布在四周。", "INVALID_INPUT") from exc


def calibrate(found, world, image, options):
    ids = sorted(world)
    src = np.concatenate([world[i] for i in ids])
    dst = np.concatenate([found[i] for i in ids])
    h, mask = cv2.findHomography(src, dst, cv2.RANSAC, 2.0)
    if h is None or not np.isfinite(h).all() or mask.sum() < 14:
        raise InspectionError("LAYOUT_MISMATCH", "标志与布局不一致或角点不可靠，请核对ID、方向和实测间距。")
    try:
        inverse = np.linalg.inv(h)
    except np.linalg.LinAlgError as exc:
        raise InspectionError("DEGENERATE_LAYOUT", "标志布局退化，请重新布置并拍摄。") from exc
    residual = np.linalg.norm(transform(src, h) - dst, axis=1)
    holdout = []
    for j in range(4):
        keep = np.ones(16, bool); keep[j*4:(j+1)*4] = False
        h_leave, _ = cv2.findHomography(dst[keep], src[keep], 0)
        if h_leave is None:
            raise InspectionError("UNSTABLE_LAYOUT", "布局交叉检验失败，请重拍。")
        holdout.append(float(np.linalg.norm(transform(dst[~keep], h_leave)-src[~keep], axis=1).max()))
    metric_error = np.linalg.norm(transform(dst, inverse)-src, axis=1)
    if residual.max() > options.max_reprojection_px or max(holdout) > options.max_holdout_error_mm or metric_error.max() > 15:
        raise InspectionError("CALIBRATION_INCONSISTENT", "标志几何不一致：核对共面、打印尺寸、布局实测值和清晰度。")
    sizes = [np.linalg.norm(found[i]-np.roll(found[i], -1, axis=0), axis=1).mean() for i in ids]
    if max(sizes)/min(sizes) > 3:
        raise InspectionError("EXTREME_PERSPECTIVE", "近远端比例变化过大，请调整拍摄角度。")
    centers = np.asarray([world[i].mean(axis=0) for i in ids], np.float32)
    hull = cv2.convexHull(centers).reshape(-1, 2)
    projected_hull = transform(hull, h)
    if cv2.contourArea(projected_hull.astype(np.float32)) < image.shape[0]*image.shape[1]*.05:
        raise InspectionError("POOR_COVERAGE", "有效测量区域占画面太小，请调整拍摄。")
    margin = options.marker_size_mm
    low, high = src.min(axis=0)-margin, src.max(axis=0)+margin
    ppm = options.rectified_px_per_mm
    size = np.ceil((high-low)*ppm).astype(int)
    if min(size) < 100 or max(size) > 8000 or int(size[0])*int(size[1]) > 16_000_000:
        raise InspectionError("LAYOUT_SIZE_LIMIT", "布局或校正分辨率超出PoC范围。", "INVALID_INPUT")
    canvas = np.array([[ppm, 0, -low[0]*ppm], [0, ppm, -low[1]*ppm], [0, 0, 1]])
    image_to_canvas = canvas @ inverse
    corrected = cv2.warpPerspective(image, image_to_canvas, tuple(size), borderValue=(35,35,35))
    valid = cv2.warpPerspective(np.full(image.shape[:2], 255, np.uint8), image_to_canvas,
                                tuple(size), flags=cv2.INTER_NEAREST)
    return {"image_to_mm": inverse, "mm_to_image": h, "mm_to_canvas": canvas,
            "image_to_canvas": image_to_canvas, "corrected": corrected, "valid_mask": valid,
            "hull_mm": hull, "world_markers": world,
            "metrics": {"max_reprojection_px": float(residual.max()),
                        "max_metric_residual_mm": float(metric_error.max()),
                        "holdout_errors_mm": holdout,
                        "inlier_count": int(mask.sum()), "marker_count": 4}}
