"""Explain marker fit problems and render the actual calibration corners."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def measurement_readiness(prediction):
    measurement=prediction.get('measurement', {})
    boundary=prediction.get('boundary_refinement', {}).get('quad_original_px') is not None
    count=prediction.get('detected_marker_count')
    target=prediction.get('markers_on_target_count')
    used=len(measurement.get('calibration_corners_px', []))
    actions=[]
    if not boundary:
        actions.append('请拍全构件上表面，适当提高拍摄角度，避免钢筋、侧面或下层板材干扰边界。')
    if count is not None and count < 3:
        actions.append('请让至少三个分散的标志完整入镜，靠近并对焦，避免反光或遮挡。')
    elif boundary and target is not None and target < 3:
        actions.append('图中有标志，但目标构件内的候选不足三个；请检查标志是否放在同一块构件上。')
    elif not boundary and count is not None:
        actions.append('已找到标志，但尚不能确认它们属于同一构件；请先拍清目标边界。')
    return {'boundary_available':boundary,'detected_marker_count':count,
            'markers_on_target_count':target,'calibration_marker_count':used,'next_actions':actions}


def marker_diagnostics(markers, measurement):
    calibration = measurement.get('calibration', {})
    errors = calibration.get('per_marker_max_error_mm', [])
    changes = calibration.get('leave_one_out_dimension_change_mm', [])
    rows = []
    for index, marker in enumerate(markers):
        error = errors[index] if index < len(errors) else None
        change = changes[index] if index < len(changes) else None
        reasons = []
        if error is not None and error > 7.5:
            reasons.append('外框与方形约束不一致，请检查模糊、定位偏移或纸面翘曲。')
        if change is not None and change > 50:
            reasons.append('移除此标志后尺寸变化较大，请核对其外框和位置。')
        if marker.get('detection_pass') == 'irregular_contour_recovery':
            reasons.append('这是补检标志，请人工核对黑色外框。')
        rows.append({'label': f'C{index+1}', 'candidate_index': marker.get('candidate_index'),
                     'max_square_fit_error_mm': error, 'leave_one_out_change_mm': change,
                     'requires_review': bool(reasons), 'reasons': reasons,
                     'corners_source': marker.get('corners_source', 'unverified_candidate')})
    return rows


def render_marker_details(image, markers, diagnostics, path):
    """Diagnostic crops only; no cropping or annotation feeds measurement."""
    if not markers:
        return None
    cols = min(2, len(markers))
    sheet = Image.new('RGB', (cols*380, ((len(markers)+cols-1)//cols)*320+60), 'white')
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 18)
    small = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 15)
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 10), '标志细节：蓝框为当前候选角点', font=font, fill='#203126')
    draw.text((12, 35), 'C 编号是候选序号；局部检查不代表尺寸准确。', font=small, fill='#465546')
    for index, (marker, diagnostic) in enumerate(zip(markers, diagnostics)):
        points = np.asarray(marker['corners_px'], float)
        lo = np.maximum(np.floor(points.min(axis=0)-20).astype(int), 0)
        hi = np.minimum(np.ceil(points.max(axis=0)+20).astype(int), [image.shape[1], image.shape[0]])
        crop = image[lo[1]:hi[1], lo[0]:hi[0]].copy()
        ratio = min(350/crop.shape[1], 232/crop.shape[0])
        crop = cv2.resize(crop, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_LINEAR)
        cv2.polylines(crop, [np.round((points-lo)*ratio).astype(np.int32)], True, (255, 130, 0), 2)
        x, y = (index % cols)*380, (index//cols)*320+60
        status = '需要复核' if diagnostic['requires_review'] else ('已参与标定' if diagnostic['corners_source']=='actual_calibration' else '未参与有效标定')
        draw.text((x+12,y+8), f'{diagnostic["label"]} · {status}', font=font, fill='#704714' if diagnostic['requires_review'] else '#203126')
        sheet.paste(Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)),(x+(380-crop.shape[1])//2,y+40))
        note = '检查黑色外框，避免把白色内框当作边界。'
        draw.text((x+12,y+280),note,font=small,fill='#465546')
    sheet.save(Path(path))
    return str(Path(path))
