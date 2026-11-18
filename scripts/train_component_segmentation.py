"""Local, reproducible draft segmentation baseline. No site integration or uploads."""
import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT/'outputs/ultralytics-local-config'))
Path(os.environ['YOLO_CONFIG_DIR']).mkdir(parents=True, exist_ok=True)
os.environ.pop('ULTRALYTICS_API_KEY', None)
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=ROOT/'samples/real/20260930/segmentation-v2/data.yaml')
    parser.add_argument('--model', default='yolo11n-seg.pt')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--imgsz', type=int, default=512)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--name', default='component-v1-cpu')
    args = parser.parse_args()
    import torch
    import ultralytics
    from ultralytics import YOLO, settings
    settings.update({k:False for k in ('sync','hub','wandb','mlflow','clearml','comet','dvc','neptune','raytune','tensorboard') if k in settings})
    settings.update({'api_key':''})
    torch.set_num_threads(args.threads)
    snapshot_path = args.data.parent/'dataset_snapshot.json'
    snapshot = json.loads(snapshot_path.read_text(encoding='utf-8'))
    # Catch modified exports or accidental cross-group splitting before training.
    groups = {}
    for record in snapshot['records']:
        group = record['group_id']; split = record['split']
        if group in groups and groups[group] != split:
            raise ValueError('Group leakage between dataset splits')
        groups[group] = split
        for folder, suffix, key in [('images','.jpg','export_image_sha256'),('labels','.txt','label_sha256')]:
            path = args.data.parent/folder/split/(record['image_id']+suffix)
            if hashlib.sha256(path.read_bytes()).hexdigest() != record[key]:
                raise ValueError(f'Frozen dataset modified: {path}')
    project = ROOT/'outputs/training'
    run_dir = project/args.name
    if run_dir.exists():
        raise FileExistsError('Choose a new --name to preserve previous training evidence')
    status_dir = ROOT/'outputs/segmentation-v1';status_dir.mkdir(parents=True,exist_ok=True)
    status_path = status_dir/'training_status.json'
    started = time.time()
    status = {'state':'initializing', 'model':args.model, 'epochs_requested':args.epochs,
              'dataset_snapshot_sha256':hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
              'counts':snapshot['counts'], 'versions':{'torch':torch.__version__,'ultralytics':ultralytics.__version__},
              'device':'cpu','threads':args.threads,'run_directory':str(run_dir),'label_status':'draft_needs_domain_review'}
    def save_status(**values):
        status.update(values);status['elapsed_seconds']=round(time.time()-started,1)
        status_path.write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
    def epoch_end(trainer):
        save_status(state='training',completed_epochs=trainer.epoch+1,
                    metrics={k:float(v) for k,v in trainer.metrics.items()})
        print('EPOCH_STATUS '+json.dumps(status,ensure_ascii=True),flush=True)
    save_status()
    try:
        model_dir=ROOT/'models/pretrained';model_dir.mkdir(parents=True,exist_ok=True)
        model_path=Path(args.model)
        if not model_path.is_absolute() and model_path.parent == Path('.'):
            model_path=model_dir/model_path
        model=YOLO(str(model_path))
        status['pretrained_weight_sha256']=hashlib.sha256(model_path.read_bytes()).hexdigest()
        model.add_callback('on_fit_epoch_end',epoch_end)
        model.train(data=str(args.data.resolve()),epochs=args.epochs,imgsz=args.imgsz,batch=args.batch,
                    device='cpu',workers=0,seed=20260930,deterministic=True,
                    project=str(project),name=args.name,exist_ok=False,
                    optimizer='AdamW',lr0=.001,lrf=.05,weight_decay=.0005,
                    freeze=10,warmup_epochs=3,patience=15,amp=False,
                    mosaic=0.0,mixup=0.0,copy_paste=0.0,degrees=8,translate=.06,scale=.2,
                    fliplr=.5,flipud=0.0,perspective=0.0,
                    hsv_h=.01,hsv_s=.15,hsv_v=.2,mask_ratio=2,
                    cache=False,plots=True,save=True,verbose=False)
        best=run_dir/'weights/best.pt'
        # Final validation also triggers on_fit_epoch_end after trainer.epoch has
        # advanced; count the actual training CSV rows instead of that callback.
        with (run_dir/'results.csv').open(encoding='utf-8', newline='') as csv_file:
            completed_epochs=len(list(csv.DictReader(csv_file)))
        save_status(state='completed',completed_epochs=completed_epochs,best_weights=str(best),best_weight_sha256=hashlib.sha256(best.read_bytes()).hexdigest())
        (run_dir/'run_provenance.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
    except Exception as exc:
        save_status(state='failed',error=str(exc))
        raise


if __name__=='__main__':
    main()
