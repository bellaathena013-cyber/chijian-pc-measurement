"""Bridge the isolated local model runtime to the existing web workbench."""
import json
import math
from pathlib import Path

from vision.visualization import read_image, save_image
from api.model_runtime import runtime

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / '.venv-train/Scripts/python.exe'
WEIGHTS = ROOT / 'outputs/training/component-v1-cpu/weights/best.pt'

REASONS = {
    'MARKER_GEOMETRY_INCONSISTENT': '标志角点与方形约束不一致，请核对黑色外框是否清晰、平整。',
    'CALIBRATION_SENSITIVE_OR_UNCHECKED': '标定稳定性不足，或标志数量不足以交叉检查。',
    'BOUNDARY_GEOMETRY_INCONSISTENT': '构件边界校正后不够规则，请核对上表面边缘。',
    'MARKER_REFINEMENT_REJECTED': '角点调整未通过稳定性检查，已保留原始角点。',
    'RECOVERED_MARKERS_REQUIRE_REVIEW': '使用了补检标志，需人工复核其外框角点。',
    'INSUFFICIENT_CUSTOM_MARKERS': '可靠标志不足三个，请让至少三个分散的标志清晰出现在构件表面。',
    'NO_STABLE_QUADRILATERAL': '无法稳定提取构件四边形，请调整拍摄角度并拍全上表面。',
    'INSUFFICIENT_STRAIGHT_EDGE': '构件直边支持不足，请避免遮挡并拍清边缘。',
    'AMBIGUOUS_IMAGE_EDGE': '存在多条相近边缘，请核对是否把侧面或下层板材当作边界。',
    'EMPTY_MASK': '没有识别出目标构件，请让构件清晰占据画面主体。',
    'TARGET_TOO_SMALL': '构件在照片中太小，请靠近后重新拍摄。',
    'MULTIPLE_SIGNIFICANT_REGIONS': '识别区域分散，无法确定完整目标，请减少周围构件干扰。',
    'TARGET_CLIPPED': '构件边缘被画面裁切，请拍全上表面。',
    'NON_RECTANGULAR_OR_CONTAMINATED': '轮廓可能为异形或混入其他物体，需要人工复核。',
    'ILL_CONDITIONED_CORNERS': '边线交点不稳定，请调整拍摄角度。',
    'WEAK_IMAGE_EDGE': '构件边缘不清晰，请改善光线并对焦。',
    'EXCESSIVE_BOUNDARY_CHANGE': '边线修正幅度过大，无法可靠确定上表面。',
    'QUAD_DISAGREES_WITH_MASK': '拟合边界与识别轮廓差异过大，需要人工复核。',
}


def available():
    return PYTHON.is_file() and WEIGHTS.is_file()


def inspect_photo(source, out, marker_size_mm, component_id):
    # Normalize EXIF orientation and enforce the same image limits as the old API.
    image = read_image(source)
    save_image(out/'original.png', image)
    work = out/'model'
    runtime.predict(PYTHON, ROOT, out/'original.png', WEIGHTS, work, marker_size_mm, out/'model.log')
    prediction = json.loads((work/'prediction.json').read_text(encoding='utf-8'))
    measurement = prediction.get('measurement', {})
    status = measurement.get('status', 'MANUAL_REQUIRED')
    if status not in {'REVIEW', 'EXPERIMENTAL_MEASUREMENT', 'RETAKE', 'MANUAL_REQUIRED', 'INVALID_INPUT'}:
        raise RuntimeError('模型返回了无法识别的状态。')
    length, width = (measurement.get(k) for k in ('measured_length_mm', 'measured_width_mm'))
    success = status in {'REVIEW', 'EXPERIMENTAL_MEASUREMENT'} and all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in (length, width))
    if status in {'REVIEW', 'EXPERIMENTAL_MEASUREMENT'} and not success:
        raise RuntimeError('模型尺寸缺失或无效。')
    reasons = measurement.get('quality_reasons', measurement.get('reasons', []))
    if measurement.get('reason_code'):
        reasons = [*reasons, measurement['reason_code']]
    messages = [REASONS.get(code, '图像质量或几何检查未通过（'+code+'），请核对照片。') for code in dict.fromkeys(reasons)]
    import cv2
    import numpy as np
    annotated = read_image(work/'prediction.jpg')
    scale = np.array([annotated.shape[1]/image.shape[1], annotated.shape[0]/image.shape[0]])
    for index, marker in enumerate(prediction.get('markers', [])):
        points = np.round(np.asarray(marker['corners_px'])*scale).astype(np.int32)
        cv2.polylines(annotated, [points], True, (255, 180, 0), 2)
        cv2.putText(annotated, f'C{index+1}', tuple(points[0]), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 180, 0), 1)
    save_image(out/'annotated.png', annotated)
    from vision.marker_diagnostics import marker_diagnostics, render_marker_details, measurement_readiness
    readiness=measurement_readiness(prediction)
    messages.extend(readiness['next_actions'])
    markers = prediction.get('markers', [])
    diagnostics = marker_diagnostics(markers, measurement)
    details = render_marker_details(image, markers, diagnostics, out/'marker-details.png')
    for item in diagnostics:
        if item['requires_review']:
            messages.append(item['label']+'：'+' '.join(item['reasons']))
    (out/'prediction.json').write_text(json.dumps(prediction, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    return {
        'result': status, 'success': success, 'component_id': component_id,
        'measurement_mode': 'photo_model', 'source_type': 'uploaded',
        'message': '已计算试验尺寸，请结合复核提示核对；不代表现场验收通过。' if success else '本次未得到可靠尺寸，请按提示调整拍摄或人工复核。',
        'measured_length_mm': round(length, 1) if success else None,
        'measured_width_mm': round(width, 1) if success else None,
        'warnings': messages, 'quality_reasons': reasons,
        'markers': markers, 'marker_diagnostics': diagnostics,
        'measurement_readiness':readiness,
        'known_marker_size_mm': marker_size_mm,
        'model_weight_sha256': prediction['weight_sha256'],
        'measurement_details': measurement,
        'boundary_refinement': prediction.get('boundary_refinement'),
        # Original-image coordinates; replay shows actual outputs, not live inference.
        'process_visualization': {
            'kind': 'result_replay',
            'image_width': int(image.shape[1]), 'image_height': int(image.shape[0]),
            'component_polygon_px': (prediction.get('candidates') or [{}])[0].get('polygon_original_px'),
            'boundary_px': (prediction.get('boundary_refinement') or {}).get('quad_original_px'),
        },
        'images': {'original': str(out/'original.png'), 'annotated': str(out/'annotated.png'), **({'marker_details': details} if details else {})},
        'calibration': None,
    }
