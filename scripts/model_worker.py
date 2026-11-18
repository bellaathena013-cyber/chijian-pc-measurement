"""Local pipe worker: retain model weights between serialized requests."""
import contextlib
import hashlib
import json
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

from predict_component import predict


def main():
    protocol = sys.stdout
    model = None
    loaded_key = None
    for line in sys.stdin:
        request = json.loads(line)
        try:
            with Path(request['log']).open('w', encoding='utf-8') as log:
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    from ultralytics import YOLO, settings
                    settings.update({'sync': False, 'api_key': ''})
                    weights = Path(request['weights'])
                    key = (str(weights.resolve()), hashlib.sha256(weights.read_bytes()).hexdigest())
                    if key != loaded_key:
                        model = YOLO(str(weights))
                        loaded_key = key
                    if request.get('operation') == 'warmup':
                        import numpy as np
                        import torch
                        torch.set_num_threads(4)
                        model.predict(np.zeros((512, 512, 3), dtype=np.uint8), imgsz=512,
                                      conf=.25, retina_masks=True, device='cpu', verbose=False)
                    else:
                        args = SimpleNamespace(image=Path(request['image']), output=Path(request['output']),
                                               weights=weights, marker_size_mm=request['marker_size_mm'],
                                               refine_boundary=False, measure=True)
                        predict(args, model=model)
            response = {'id': request['id'], 'ok': True}
        except Exception:
            with Path(request['log']).open('a', encoding='utf-8') as log:
                traceback.print_exc(file=log)
            # Discard partially initialized or failed predictor state.
            model = None
            loaded_key = None
            response = {'id': request['id'], 'ok': False}
        protocol.write(json.dumps(response)+'\n')
        protocol.flush()


if __name__ == '__main__':
    main()
