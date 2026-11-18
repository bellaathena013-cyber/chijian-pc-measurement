"""Evaluate manually annotated pilot boundaries without fitting to tape values."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from vision.free_square_calibration import measure_with_free_squares
from vision.visualization import read_image, save_image
from vision.quality_check import InspectionError

root=Path('samples/real/20260930')
out=Path('outputs/custom-marker-pilot')
manifest=json.loads((root/'reference_measurements.json').read_text(encoding='utf-8'))
records={r['image_id']:r for r in manifest['records']}
detections={r['image_id']:r for r in json.loads((out/'detections.json').read_text(encoding='utf-8'))}
annotations=json.loads((root/'pilot_boundaries.json').read_text(encoding='utf-8'))
reports=[]
for annotation in annotations['records']:
    image_id=annotation['image_id'];record=records[image_id]
    boundary=np.array(annotation['boundary_original_px'],float)
    detected=detections[image_id]['markers']
    markers=[m for m in detected if cv2.pointPolygonTest(boundary.astype(np.float32),tuple(np.mean(m['corners_px'],axis=0)),False)>=0]
    row={'image_id':image_id,'selection':'six_development_photos_with_complete_smooth_top_surface',
         'marker_detection':'automatic_pattern_candidates','boundary_source':'assistant_manual_image_annotation_unreviewed',
         'marker_count':len(markers),'marker_candidates':markers}
    try:
        row.update(measure_with_free_squares([m['corners_px'] for m in markers],boundary,record['marker_outer_side_mm']))
    except InspectionError as exc:
        row.update({'status':'RETAKE','reason':str(exc),'reason_code':exc.code})
    row['boundary_source'] = annotations['annotation_source']
    row['boundary_review_status'] = annotations['status']
    # Evaluation is deliberately after inference; truth is never passed above.
    row['reference_length_mm']=record['reference_length_mm'];row['reference_width_mm']=record['reference_width_mm']
    if 'measured_length_mm' in row:
        row['length_error_mm']=row['measured_length_mm']-record['reference_length_mm']
        row['width_error_mm']=row['measured_width_mm']-record['reference_width_mm']
        row['within_50mm_of_tape']=max(abs(row['length_error_mm']),abs(row['width_error_mm']))<=50
    reports.append(row)
    image=read_image(root/record['image'])
    for m in markers:cv2.polylines(image,[np.round(m['corners_px']).astype(np.int32)],True,(0,170,255),5)
    cv2.polylines(image,[boundary.round().astype(np.int32)],True,(80,200,20),5)
    preview=Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB));preview.thumbnail((960,1280))
    page=Image.new('RGB',(1000,preview.height+180),'white');page.paste(preview,((1000-preview.width)//2,160));d=ImageDraw.Draw(page);f=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',23)
    d.text((24,12),f'{image_id} · 开发试验，人工构件边界',font=f,fill='black')
    d.text((24,48),f"卷尺：{row['reference_length_mm']} × {row['reference_width_mm']} mm",font=f,fill='black')
    if 'measured_length_mm' in row:
        d.text((24,84),f"计算：{row['measured_length_mm']:.0f} × {row['measured_width_mm']:.0f} mm；偏差 {row['length_error_mm']:+.0f} / {row['width_error_mm']:+.0f} mm",font=f,fill='black')
    d.text((24,120),f"状态：{row['status']}；橙框为标志候选，绿框为人工边界",font=f,fill='black')
    page.save(out/f'{image_id}-measurement.jpg',quality=92)
    print(json.dumps({k:v for k,v in row.items() if k not in ('marker_candidates','image_to_mm','boundary_mm')},ensure_ascii=False),flush=True)
(out/'pilot_results.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2),encoding='utf-8')
