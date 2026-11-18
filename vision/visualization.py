from pathlib import Path
import warnings
import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from .quality_check import InspectionError


def read_image(path):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as im:
                if im.width * im.height > 32_000_000:
                    raise InspectionError("IMAGE_TOO_LARGE", "图片超过3200万像素，请缩小原图。", "INVALID_INPUT")
                im = ImageOps.exif_transpose(im).convert("RGB")
                if min(im.size) < 240:
                    raise InspectionError("LOW_RESOLUTION", "分辨率过低，请上传清晰原图。")
                return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombWarning, Image.DecompressionBombError) as exc:
        raise InspectionError("IMAGE_READ_FAILED", "无法读取图片，请使用完整JPG或PNG文件。", "INVALID_INPUT") from exc


def save_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv2.imencode(path.suffix, image)
    if not ok:
        raise OSError("Unable to encode image")
    data.tofile(str(path))
    return str(path.resolve())


def annotate(image, markers, corners=None, label=""):
    out = image.copy()
    for marker_id, pts in markers.items():
        pts = np.asarray(pts).round().astype(np.int32)
        cv2.polylines(out, [pts], True, (50, 190, 40), 3)
        cv2.putText(out, f"ID {marker_id}", tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, .65, (20, 110, 20), 2)
    if corners is not None:
        pts = np.asarray(corners).round().astype(np.int32)
        cv2.polylines(out, [pts], True, (0, 140, 255), 3)
        for i, p in enumerate(pts):
            cv2.circle(out, tuple(p), 6, (0, 80, 255), -1)
            cv2.putText(out, str(i+1), tuple(p+[8, -8]), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 80, 255), 2)
    if label:
        cv2.rectangle(out, (0, 0), (min(out.shape[1], 1050), 48), (255, 255, 255), -1)
        cv2.putText(out, label, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, .7, (30, 30, 30), 2)
    return out
