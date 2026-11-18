"""Compare contour recovery on selected splits without fitting to tape values."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--splits', nargs='+', choices=['train', 'val', 'test'], default=['train', 'val'])
    args = parser.parse_args()
    import cv2
    import numpy as np
    from vision.custom_markers import detect_custom_markers_robust
    from vision.visualization import read_image, save_image
    source = ROOT/'samples/real/20260930'
    snapshot = json.loads((source/'segmentation-v2/dataset_snapshot.json').read_text(encoding='utf-8'))
    # Manifest paths only; tape dimensions are not passed to the detector.
    paths = {r['image_id']: r['image'] for r in json.loads((source/'reference_measurements.json').read_text(encoding='utf-8'))['records']}
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for row in snapshot['records']:
        if row['split'] not in args.splits:
            continue
        image_id = row['image_id']
        image = read_image(source/paths[image_id])
        markers = detect_custom_markers_robust(image)
        added = [m for m in markers if m['detection_pass'] != 'baseline']
        rows.append({'image_id': image_id, 'image': paths[image_id], 'split': row['split'], 'markers': markers})
        for index, marker in enumerate(added):
            points = np.array(marker['corners_px'])
            lo = np.maximum(np.floor(points.min(axis=0)-25).astype(int), 0)
            hi = np.minimum(np.ceil(points.max(axis=0)+25).astype(int), [image.shape[1], image.shape[0]])
            crop = image[lo[1]:hi[1], lo[0]:hi[0]].copy()
            crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
            cv2.polylines(crop, [np.round((points-lo)*3).astype(np.int32)], True, (0, 165, 255), 2)
            save_image(args.output/f'{image_id}-added-{index}.jpg', crop)
        print(image_id, row['split'], 'baseline', len(markers)-len(added), 'recovered', len(added), flush=True)
    (args.output/'detections.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    (args.output/'provenance.json').write_text(json.dumps({'splits': args.splits, 'algorithm_sha256': hashlib.sha256((ROOT/'vision/custom_markers.py').read_bytes()).hexdigest(), 'limitations': 'Pattern candidates, not manually established detection recall.'}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
