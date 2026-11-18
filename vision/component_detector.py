import cv2
import numpy as np
from .perspective import transform
from .quality_check import InspectionError


def ordered_quad(points):
    p = np.asarray(points, np.float64)
    if p.shape != (4, 2) or not np.isfinite(p).all():
        raise InspectionError("INVALID_CORNERS", "人工角点须为四组有效原图像素坐标。", "INVALID_INPUT")
    # Input is a boundary polygon: reject crossing edges rather than silently reordering mistakes.
    if not cv2.isContourConvex(p.astype(np.float32)) or abs(cv2.contourArea(p.astype(np.float32))) < 100:
        raise InspectionError("INVALID_CORNERS", "请沿边界顺序选择四角，不能交叉或重复。", "INVALID_INPUT")
    return p


def validate_boundary(points_image, cal, image_shape):
    p = ordered_quad(points_image)
    if (p < 5).any() or (p[:,0] > image_shape[1]-6).any() or (p[:,1] > image_shape[0]-6).any():
        raise InspectionError("COMPONENT_CLIPPED", "构件边界靠近画面裁切处，请完整拍摄。")
    mm = transform(p, cal["image_to_mm"])
    if not cv2.isContourConvex(mm.astype(np.float32)):
        raise InspectionError("INVALID_GEOMETRY", "校正后的边界无效。", "MANUAL_REQUIRED")
    for point in mm:
        if cv2.pointPolygonTest(cal["hull_mm"].astype(np.float32), tuple(point), True) < 5:
            raise InspectionError("OUTSIDE_MARKERS", "边界超出标志包围范围或太贴近标志，请重新布置。")
    v = np.roll(mm, -1, axis=0)-mm
    lengths = np.linalg.norm(v, axis=1)
    if lengths.min() < 100:
        raise InspectionError("BOUNDARY_TOO_SMALL", "边界过小，请核对构件四角。", "MANUAL_REQUIRED")
    cosine = np.abs(np.sum(v*np.roll(v,-1,axis=0),axis=1)/(lengths*np.roll(lengths,-1)))
    if cosine.max() > .12 or max(abs(lengths[0]-lengths[2]), abs(lengths[1]-lengths[3])) > max(25, lengths.min()*.03):
        raise InspectionError("NOT_RECTANGULAR", "校正后对边或直角不一致，请调整边界、核对共面或重新拍摄。", "MANUAL_REQUIRED")
    return mm


def detect_component(cal):
    image = cal["corrected"]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    hull = transform(cal["hull_mm"], cal["mm_to_canvas"]).astype(np.int32)
    roi = np.zeros(gray.shape, np.uint8)
    cv2.fillConvexPoly(roi, hull, 255)
    roi = cv2.erode(roi, np.ones((9,9), np.uint8))
    roi[cal["valid_mask"] == 0] = 0
    for pts in cal["world_markers"].values():
        p = transform(pts, cal["mm_to_canvas"])
        p = (p-p.mean(axis=0))*1.5+p.mean(axis=0)
        cv2.fillConvexPoly(roi,p.astype(np.int32),0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    edges = cv2.Canny(gray, 35, 100)
    masks = [binary, 255-binary, cv2.morphologyEx(edges,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))]
    hull_area = cv2.contourArea(hull)
    candidates = []
    for mask in masks:
        mask = cv2.bitwise_and(mask, roi)
        contours,_ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = abs(cv2.contourArea(contour))
            if not hull_area*.1 < area < hull_area*.92:
                continue
            p = cv2.approxPolyDP(contour, .008*cv2.arcLength(contour,True),True).reshape(-1,2)
            if len(p)!=4 or not cv2.isContourConvex(p):
                continue
            # Reject masks produced by cutting background at the measurement ROI.
            if any(cv2.pointPolygonTest(hull, tuple(map(float,pt)),True)<10 for pt in p):
                continue
            rect = cv2.minAreaRect(p.astype(np.float32))
            if area/max(1,rect[1][0]*rect[1][1]) < .94:
                continue
            a = abs(cv2.contourArea(p))
            if any(np.linalg.norm(p.mean(axis=0)-q.mean(axis=0))<12 and abs(a-b)<.04*a for b,q in candidates):
                continue
            candidates.append((a,p.astype(np.float64)))
    if len(candidates)!=1:
        raise InspectionError("BOUNDARY_AMBIGUOUS" if candidates else "BOUNDARY_NOT_FOUND",
                              "没有找到唯一可靠矩形，请人工沿边界选择四角。", "MANUAL_REQUIRED")
    return transform(candidates[0][1], np.linalg.inv(cal["image_to_canvas"]))
