"""Detect custom cards in the confirmed real-photo intake; no ground-truth fitting."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from vision.visualization import read_image, save_image
from vision.custom_markers import detect_custom_markers

root = Path('samples/real/20260930')
out = Path('outputs/custom-marker-pilot'); out.mkdir(parents=True, exist_ok=True)
records = json.loads((root/'reference_measurements.json').read_text(encoding='utf-8'))['records']
rows = []; thumbs = []
font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 15)
for index, record in enumerate(records):
    image = read_image(root/record['image'])
    markers = detect_custom_markers(image)
    rows.append({'image_id': record['image_id'], 'image': record['image'], 'markers': markers})
    annotated = image.copy()
    for item in markers:
        p = np.round(item['corners_px']).astype(np.int32)
        cv2.polylines(annotated, [p], True, (0, 150, 255), 5)
        cv2.putText(annotated, f"C{item['candidate_index']}", tuple(p[0]), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)
    save_image(out/f"{record['image_id']}-markers.jpg", annotated)
    thumb = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)); thumb.thumbnail((250, 330))
    tile = Image.new('RGB', (270, 370), 'white'); tile.paste(thumb, ((270-thumb.width)//2, 0))
    ImageDraw.Draw(tile).text((4, 336), f"{index+1:02d} {record['image_id']}  候选{len(markers)}", font=font, fill='black')
    thumbs.append(tile)
    print(record['image_id'], len(markers), flush=True)
(out/'detections.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
for start in range(0, len(thumbs), 15):
    sheet = Image.new('RGB', (1350, 1110), '#ddd')
    for k, tile in enumerate(thumbs[start:start+15]):
        sheet.paste(tile, ((k%5)*270, (k//5)*370))
    sheet.save(out/f'contact-{start//15+1}.jpg', quality=90)
print('count distribution:', {n: sum(len(r['markers']) == n for r in rows) for n in sorted(set(len(r['markers']) for r in rows))})
