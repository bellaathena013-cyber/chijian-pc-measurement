"""Run the local draft model on a new photo without any reference dimensions."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'outputs/ultralytics-local-config'))
Path(os.environ['YOLO_CONFIG_DIR']).mkdir(parents=True,exist_ok=True)
os.environ.pop('ULTRALYTICS_API_KEY',None)


def predict(args, model=None):
    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLO,settings
    settings.update({'sync':False,'api_key':''})
    torch.set_num_threads(4)
    if not args.image.is_file() or not args.weights.is_file():
        raise FileNotFoundError('Photo or trained weights not found')
    image=cv2.imdecode(np.frombuffer(args.image.read_bytes(),np.uint8),cv2.IMREAD_COLOR)
    if image is None:raise ValueError('Unable to decode photo')
    if not np.isfinite(args.marker_size_mm) or args.marker_size_mm<=0:raise ValueError('Invalid marker size')
    if args.output.exists():raise FileExistsError('Choose a new output folder to preserve prior results')
    if model is None:
        model=YOLO(str(args.weights))
    ratio=min(1.,1280/max(image.shape[:2]))
    working=cv2.resize(image,None,fx=ratio,fy=ratio,interpolation=cv2.INTER_AREA)
    coordinate_scale=np.array([image.shape[1]/working.shape[1],image.shape[0]/working.shape[0]])
    result=model.predict(working,imgsz=512,conf=.25,retina_masks=True,device='cpu',verbose=False)[0]
    candidates=[]
    if result.masks is not None:
        for index,(polygon,confidence) in enumerate(zip(result.masks.xy,result.boxes.conf.cpu().numpy())):
            candidates.append({'candidate_index':index,'confidence_score':float(confidence),'polygon_original_px':(polygon*coordinate_scale).tolist()})
    candidates.sort(key=lambda c:c['confidence_score'],reverse=True)
    output={'status':'DRAFT_PREDICTION' if candidates else 'MANUAL_REQUIRED',
            'weight_sha256':hashlib.sha256(args.weights.read_bytes()).hexdigest(),
            'image_width':image.shape[1],'image_height':image.shape[0],
            'working_width':working.shape[1],'working_height':working.shape[0],
            'selection_policy':'highest_confidence','candidates':candidates,
            'limitation':'Experimental contour only; not a dimension or quality acceptance result.'}
    plotted=result.plot()
    if args.refine_boundary or args.measure:
        from vision.segmentation_refinement import refine_component_mask
        index=candidates[0]['candidate_index'] if candidates else None
        mask=result.masks.data[index].cpu().numpy() if index is not None else np.zeros(working.shape[:2],np.uint8)
        mask=cv2.resize(mask.astype(np.uint8),(working.shape[1],working.shape[0]),interpolation=cv2.INTER_NEAREST)
        refined=refine_component_mask(mask,working)
        refined.pop('mask')
        quad_points=refined.pop('quad_px')
        quad=np.asarray(quad_points) if quad_points is not None else None
        refined['quad_original_px']=(quad*coordinate_scale).tolist() if quad is not None else None
        output['boundary_refinement']=refined
        if quad is not None:cv2.polylines(plotted,[np.round(quad).astype(np.int32)],True,(0,165,255),3)
        if args.measure:
            from vision.custom_markers import detect_custom_markers_robust,order_quad
            from vision.custom_marker_refinement import refine_marker_candidates,calibrated_marker_candidates
            from vision.free_square_calibration import measure_with_guarded_marker_update
            from vision.quality_check import InspectionError
            output['measurement']={'status':'MANUAL_REQUIRED','reasons':refined['reasons']}
            detected=detect_custom_markers_robust(image)
            output['detected_marker_count']=len(detected)
            output['markers']=detected
            output['markers_on_target_count']=None
            if quad is not None:
                original_quad=order_quad(quad*coordinate_scale)
                markers=[m for m in detected if cv2.pointPolygonTest(original_quad,tuple(np.mean(m['corners_px'],axis=0)),False)>=0]
                output['markers_on_target_count']=len(markers)
                adjusted=refine_marker_candidates(image,markers)
                output['markers']=adjusted
                try:
                    measurement=measure_with_guarded_marker_update([m['corners_px'] for m in markers],[m['corners_px'] for m in adjusted],original_quad,args.marker_size_mm)
                    measurement['boundary_source']='trained_mask_with_image_edge_refinement'
                    measurement['limitation']='Experimental photo measurement; not field acceptance.'
                    if refined['reasons']:
                        measurement['status']='REVIEW';measurement['quality_reasons'].extend(refined['reasons'])
                    if any(m.get('detection_pass') == 'irregular_contour_recovery' for m in markers):
                        measurement['status']='REVIEW'
                        measurement['quality_reasons'].append('RECOVERED_MARKERS_REQUIRE_REVIEW')
                    output['measurement']=measurement
                    output['markers']=calibrated_marker_candidates(adjusted,measurement)
                except InspectionError as exc:
                    output['measurement']={'status':exc.result,'reason_code':exc.code,'reason':str(exc)}
            output['status']=output['measurement']['status']
    args.output.mkdir(parents=True)
    (args.output/'prediction.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    ok,encoded=cv2.imencode('.jpg',plotted)
    if not ok:raise ValueError('Failed to encode prediction image')
    (args.output/'prediction.jpg').write_bytes(encoded.tobytes())
    print(json.dumps({'status':output['status'],'candidates':len(candidates),'output':str(args.output)},ensure_ascii=True))


def main():
    parser=argparse.ArgumentParser(description='Local draft component segmentation; not dimension acceptance')
    parser.add_argument('image',type=Path)
    parser.add_argument('--weights',type=Path,default=ROOT/'outputs/training/component-v1-cpu/weights/best.pt')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--refine-boundary',action='store_true')
    parser.add_argument('--measure',action='store_true',help='Experimental measurement using detected 150 mm cards')
    parser.add_argument('--marker-size-mm',type=float,default=150)
    args=parser.parse_args()
    predict(args)


if __name__=='__main__':main()
