"""Report fixed-threshold predictions on model-held-out photos, including failures."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'outputs/ultralytics-local-config'))
Path(os.environ['YOLO_CONFIG_DIR']).mkdir(parents=True, exist_ok=True)
os.environ.pop('ULTRALYTICS_API_KEY', None)
os.environ.setdefault('OMP_NUM_THREADS','4')
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont


def rectangle_candidate(polygon):
    contour=np.asarray(polygon,np.float32).reshape(-1,1,2)
    if len(contour)<4:return None
    hull=cv2.convexHull(contour)
    if cv2.contourArea(hull)<=0:return None
    # Do not invent a rectangle if the model predicts an irregular boundary.
    if abs(cv2.contourArea(contour))/cv2.contourArea(hull)<.94:return None
    for epsilon in (.005,.01,.015,.02):
        quad=cv2.approxPolyDP(hull,epsilon*cv2.arcLength(hull,True),True)
        if len(quad)==4 and cv2.isContourConvex(quad):
            if abs(cv2.contourArea(quad)-cv2.contourArea(hull))/cv2.contourArea(hull)<.04:
                return quad.reshape(4,2)
    return None


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--weights',type=Path,default=ROOT/'models/component-v1/best.pt')
    parser.add_argument('--dataset',type=Path,default=ROOT/'samples/real/20260930/segmentation-v2')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/segmentation-v1/evaluation')
    parser.add_argument('--imgsz',type=int,default=512)
    parser.add_argument('--conf',type=float,default=.25)
    parser.add_argument('--splits',nargs='+',choices=['train','val','test'],default=['val','test'])
    parser.add_argument('--refine-boundary',action='store_true')
    parser.add_argument('--mask-only-refinement',action='store_true')
    parser.add_argument('--refine-markers',action='store_true')
    parser.add_argument('--selective-markers',action='store_true',help='Experimental partial updates after joint rejection')
    parser.add_argument('--marker-detections',type=Path,default=ROOT/'outputs/custom-marker-pilot/detections.json')
    args=parser.parse_args()
    if args.selective_markers and not args.refine_markers:
        parser.error('--selective-markers requires --refine-markers')
    import torch
    from ultralytics import YOLO,settings
    settings.update({k:False for k in ('sync','hub','wandb','mlflow','comet','tensorboard') if k in settings})
    settings.update({'api_key':''})
    torch.set_num_threads(4)
    from vision.free_square_calibration import measure_with_free_squares,measure_with_guarded_marker_update
    from vision.free_square_calibration import measure_with_selective_marker_update
    from vision.quality_check import InspectionError
    from vision.custom_markers import order_quad
    from vision.segmentation_refinement import refine_component_mask
    from vision.custom_marker_refinement import refine_marker_candidates
    snapshot=json.loads((args.dataset/'dataset_snapshot.json').read_text(encoding='utf-8'))
    original_root=ROOT/'samples/real/20260930'
    refs={r['image_id']:r for r in json.loads((original_root/'reference_measurements.json').read_text(encoding='utf-8'))['records']}
    detections={r['image_id']:r for r in json.loads(args.marker_detections.read_text(encoding='utf-8'))}
    model=YOLO(str(args.weights))
    args.output.mkdir(parents=True,exist_ok=False)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',19)
    results=[];tiles=[]
    for row in snapshot['records']:
        if row['split'] not in args.splits:continue
        image_id=row['image_id'];path=args.dataset/'images'/row['split']/f'{image_id}.jpg'
        im=Image.open(path).convert('RGB');w,h=im.size
        # Model sees pixels only. Annotations and tape values are used after prediction.
        predicted=model.predict(source=np.array(im)[:,:,::-1].copy(),imgsz=args.imgsz,conf=args.conf,
                                device='cpu',retina_masks=True,verbose=False)[0]
        polygons=predicted.masks.xy if predicted.masks is not None else []
        confidence=predicted.boxes.conf.cpu().numpy() if len(polygons) else []
        index=int(np.argmax(confidence)) if len(polygons) else None
        polygon=polygons[index] if index is not None else None
        record={'image_id':image_id,'split':row['split'],'group_id':row['group_id'],
                'detected_instances':len(polygons),'selection_policy':'highest_confidence_no_ground_truth_selection',
                'confidence':float(confidence[index]) if index is not None else None,
                'polygon_normalized':(polygon/[w,h]).tolist() if polygon is not None else None}
        truth=np.asarray(row['polygon_normalized'])*[w,h]
        gt=np.zeros((h,w),np.uint8);cv2.fillPoly(gt,[np.round(truth).astype(np.int32)],1)
        pm=np.zeros_like(gt)
        if index is not None:
            # Evaluate the actual binary mask. masks.xy may join disjoint islands
            # with zero-width bridges, which are not real predicted edges.
            raw_mask=predicted.masks.data[index].cpu().numpy()
            pm=cv2.resize(raw_mask.astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)>0
        contours,_=cv2.findContours(pm.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        record['external_contours_normalized']=[(c.reshape(-1,2)/[w,h]).tolist() for c in contours]
        record['mask_iou_against_draft']=float(np.logical_and(gt,pm).sum()/np.logical_or(gt,pm).sum())
        record['measurement']={'status':'MANUAL_REQUIRED','reason':'No reliable four-corner model boundary'}
        quad=rectangle_candidate(polygon) if polygon is not None else None
        if args.refine_boundary:
            refined=refine_component_mask(pm,np.array(im)[:,:,::-1].copy(),use_image_edges=not args.mask_only_refinement)
            cleaned=refined.pop('mask')
            record['boundary_refinement']=refined
            record['cleaned_mask_iou_against_draft']=float(np.logical_and(gt,cleaned).sum()/np.logical_or(gt,cleaned).sum())
            contours,_=cv2.findContours(cleaned,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            quad=np.asarray(refined['quad_px']) if refined['quad_px'] is not None else None
            if quad is None:record['measurement']['reason']='; '.join(refined['reasons'])
        ref=refs[image_id]
        image_markers=detections[image_id]['markers']
        if args.refine_markers:
            original=cv2.imdecode(np.frombuffer((original_root/ref['image']).read_bytes(),np.uint8),cv2.IMREAD_COLOR)
            image_markers=refine_marker_candidates(original,image_markers)
            record['marker_refinement']=[m['outer_edge_refinement'] for m in image_markers]
        if quad is not None:
            original_quad=order_quad(quad/[w,h]*[ref['width_px'],ref['height_px']])
            selected_indices=[i for i,m in enumerate(detections[image_id]['markers']) if cv2.pointPolygonTest(original_quad,tuple(np.mean(m['corners_px'],axis=0)),False)>=0]
            markers=[image_markers[i] for i in selected_indices]
            try:
                if args.refine_markers:
                    baseline=[detections[image_id]['markers'][i]['corners_px'] for i in selected_indices]
                    measure=measure_with_selective_marker_update if args.selective_markers else measure_with_guarded_marker_update
                    measurement=measure(baseline,[m['corners_px'] for m in markers],original_quad,150)
                else:
                    measurement=measure_with_free_squares([m['corners_px'] for m in markers],original_quad,150)
                measurement['boundary_source']='trained_segmentation_polygon_approximation'
                if args.refine_boundary:
                    measurement['boundary_source']='cleaned_mask_robust_line_intersections'
                    if refined['reasons']:
                        measurement['status']='REVIEW'
                        measurement['quality_reasons'].extend(refined['reasons'])
                measurement['limitation']='Draft model baseline with experimental marker candidates; not field acceptance.'
                if any(m.get('detection_pass') == 'irregular_contour_recovery' for m in markers):
                    measurement['status']='REVIEW'
                    measurement['quality_reasons'].append('RECOVERED_MARKERS_REQUIRE_REVIEW')
                measurement['length_error_mm']=measurement['measured_length_mm']-ref['reference_length_mm']
                measurement['width_error_mm']=measurement['measured_width_mm']-ref['reference_width_mm']
                measurement['within_50mm_of_tape']=max(abs(measurement['length_error_mm']),abs(measurement['width_error_mm']))<=50
                record['measurement']=measurement
            except InspectionError as exc:
                record['measurement']={'status':'RETAKE','reason_code':exc.code,'reason':str(exc)}
        results.append(record)
        preview=im.resize((600,800));draw=ImageDraw.Draw(preview)
        target=np.asarray(row['polygon_normalized'])*[600,800]
        pts=[tuple(p) for p in target];draw.line(pts+[pts[0]],fill='#00e890',width=3)
        for contour in contours:
            pts=[tuple(p) for p in contour.reshape(-1,2)/[w,h]*[600,800]]
            if len(pts)>1:draw.line(pts+[pts[0]],fill='#ff4055',width=3)
        if args.refine_boundary and quad is not None:
            pts=[tuple(p) for p in quad/[w,h]*[600,800]]
            draw.line(pts+[pts[0]],fill='#ff9900',width=2)
        page=Image.new('RGB',(600,910),'white');page.paste(preview,(0,110));d=ImageDraw.Draw(page)
        d.text((10,8),f'{image_id} · {row["split"]} · '+('轮廓优化' if args.refine_boundary else '初版模型预测'),font=font,fill='black')
        d.text((10,36),'红：模型轮廓；绿：标注草稿；橙：拟合边界' if args.refine_boundary else '红线：模型轮廓；绿线：人工标注草稿',font=font,fill='black')
        d.text((10,64),f'轮廓 IoU {record["mask_iou_against_draft"]:.3f}；测量 {record["measurement"]["status"]}',font=font,fill='black')
        page.save(args.output/f'{image_id}-prediction.jpg',quality=94)
        tiles.append(page.resize((300,455)))
        print(json.dumps({k:v for k,v in record.items() if k not in ('polygon_normalized','external_contours_normalized','measurement')},ensure_ascii=True),flush=True)
    summary={}
    for split in args.splits:
        rows=[r for r in results if r['split']==split]
        measured=[r for r in rows if 'measured_length_mm' in r['measurement']]
        summary[split]={'images':len(rows),'groups':len(set(r['group_id'] for r in rows)),
                        'detected_images':sum(r['detected_instances']>0 for r in rows),
                        'mean_mask_iou_against_draft':float(np.mean([r['mask_iou_against_draft'] for r in rows])),
                        'minimum_mask_iou_against_draft':min(r['mask_iou_against_draft'] for r in rows),
                        'computed_measurements':len(measured),
                        'within_50mm_among_computed':sum(r['measurement']['within_50mm_of_tape'] for r in measured)}
        if args.refine_boundary:
            summary[split]['mean_cleaned_mask_iou_against_draft']=float(np.mean([r['cleaned_mask_iou_against_draft'] for r in rows]))
    report={'weights':str(args.weights),'weight_sha256':hashlib.sha256(args.weights.read_bytes()).hexdigest(),
            'algorithm_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ('vision/segmentation_refinement.py','vision/custom_marker_refinement.py','vision/free_square_calibration.py','vision/custom_markers.py')},
            'test_interpretation':'historical_regression_after_prior_inspection' if 'test' in args.splits and args.refine_boundary else 'original_split_names_preserved',
            'iou_method':'actual_binary_mask_vs_draft_envelope',
            'boundary_refinement':args.refine_boundary,'image_edge_refinement':args.refine_boundary and not args.mask_only_refinement,
            'marker_edge_refinement':args.refine_markers,
            'selective_marker_updates':args.selective_markers,
            'marker_detections':str(args.marker_detections),
            'marker_detections_sha256':hashlib.sha256(args.marker_detections.read_bytes()).hexdigest(),
            'confidence_threshold':args.conf,'image_size':args.imgsz,'summary':summary,'records':results,
            'limitations':['39 original photos, one collection site','Assistant draft labels, not domain-reviewed ground truth',
                           'Conservative provisional groups, physical identities unconfirmed','Test is held out from training, not blind to earlier project development',
                           'IoU is segmentation overlap, not dimension accuracy','Markers are classical pattern candidates, not trained segmentation or decoded IDs']}
    (args.output/'evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    for start in range(0,len(tiles),6):
        sheet=Image.new('RGB',(900,910),'#ddd')
        for k,tile in enumerate(tiles[start:start+6]):sheet.paste(tile,((k%3)*300,(k//3)*455))
        sheet.save(args.output/f'contact-{start//6+1}.jpg',quality=94)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
