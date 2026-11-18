"""Export audited draft top-surface polygons, preserving group split and provenance."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'samples/real/20260930/segmentation-v2')
    args = parser.parse_args()
    root = ROOT/'samples/real/20260930'
    manifest = json.loads((root/'reference_measurements.json').read_text(encoding='utf-8'))
    draft_path = root/'segmentation_draft_preview.json'
    draft = json.loads(draft_path.read_text(encoding='utf-8'))
    groups_path = ROOT/'config/segmentation-groups.v1.json'
    groups = json.loads(groups_path.read_text(encoding='utf-8'))
    assignments = {}
    for group in groups['groups']:
        for index in group['indices']:
            if index in assignments:
                raise ValueError(f'Duplicate split assignment: {index}')
            assignments[index] = group
    expected = set(range(1, len(manifest['records'])+1))
    if set(assignments) != expected or set(map(int, draft['polygons'])) != expected:
        raise ValueError('All source images must have exactly one annotation and split')
    if args.output.exists():
        raise FileExistsError('Use a new version directory; do not overwrite a frozen dataset')
    # Validate everything before writing the frozen dataset.
    records = []
    seen_hashes = set()
    dims = np.array([draft['coordinate_system']['width'], draft['coordinate_system']['height']])
    for index, source in enumerate(manifest['records'], 1):
        original = root/source['image']
        digest = hashlib.sha256(original.read_bytes()).hexdigest()
        if digest != source['sha256'] or digest in seen_hashes:
            raise ValueError('Source modified or duplicate image found')
        seen_hashes.add(digest)
        polygon = np.asarray(draft['polygons'][str(index)], dtype=float) / dims
        if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon)<3 or not np.isfinite(polygon).all():
            raise ValueError(f'Invalid polygon for {index}')
        if (polygon<0).any() or (polygon>1).any() or abs(cv2.contourArea(polygon.astype(np.float32)))<.01:
            raise ValueError(f'Out-of-range or tiny polygon for {index}')
        group = assignments[index]
        records.append({'image_id':source['image_id'], 'source_index':index, 'source_image':source['image'],
                        'source_sha256':digest, 'group_id':group['group_id'], 'split':group['split'],
                        'polygon_normalized':polygon.tolist(), 'annotation_status':draft['review_status']})
    args.output.mkdir(parents=True)
    gallery = ROOT/'outputs/segmentation-v1/annotations'
    gallery.mkdir(parents=True, exist_ok=True)
    tiles = []
    for row in records:
        split = row['split']; image_id = row['image_id']
        images = args.output/'images'/split; images.mkdir(parents=True, exist_ok=True)
        labels = args.output/'labels'/split; labels.mkdir(parents=True, exist_ok=True)
        im = Image.open(root/row['source_image']).convert('RGB')
        im.thumbnail((1280,1280), Image.Resampling.LANCZOS)
        im.save(images/f'{image_id}.jpg', quality=95)
        polygon = np.asarray(row['polygon_normalized'])
        label = '0 '+' '.join(f'{n:.8f}' for n in polygon.ravel())+'\n'
        (labels/f'{image_id}.txt').write_text(label, encoding='utf-8', newline='\n')
        row['export_image_sha256'] = hashlib.sha256((images/f'{image_id}.jpg').read_bytes()).hexdigest()
        row['label_sha256'] = hashlib.sha256((labels/f'{image_id}.txt').read_bytes()).hexdigest()
        preview = im.resize((768,1024))
        draw = ImageDraw.Draw(preview)
        pts = [tuple(p) for p in polygon*[768,1024]]
        draw.line(pts+[pts[0]], fill='#00ff70', width=3)
        draw.rectangle((0,0,768,28), fill='white')
        draw.text((8,6), f"{row['source_index']:02d} {image_id} {split} / {row['group_id']} / DRAFT", fill='black')
        preview.save(gallery/f'{image_id}.jpg', quality=92)
        tiles.append(preview.resize((240,320)))
    for start in range(0,len(tiles),15):
        sheet = Image.new('RGB',(1200,960),'#ddd')
        for k,tile in enumerate(tiles[start:start+15]):sheet.paste(tile,((k%5)*240,(k//5)*320))
        sheet.save(gallery/f'contact-{start//15+1}.jpg', quality=92)
    yaml = f'path: {json.dumps(args.output.resolve().as_posix(), ensure_ascii=False)}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: marked_component_top\n'
    (args.output/'data.yaml').write_text(yaml, encoding='utf-8')
    snapshot = {'status':'draft_training_baseline_only', 'class_name':draft['class_name'],
                'annotation_policy':draft['policy'], 'annotation_source':draft['annotation_source'],
                'source_annotation_sha256':hashlib.sha256(draft_path.read_bytes()).hexdigest(),
                'split_config_sha256':hashlib.sha256(groups_path.read_bytes()).hexdigest(),
                'grouping_status':groups['status'], 'counts':dict(Counter(r['split'] for r in records)),
                'reference_dimensions_excluded_from_training':True, 'records':records}
    (args.output/'dataset_snapshot.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(snapshot['counts']))
    print(args.output/'data.yaml')


if __name__ == '__main__':
    main()
