from pathlib import Path
import csv
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tests.synthetic import scene
from vision.pipeline import inspect
from vision.quality_check import Options

layout=json.loads(Path('config/layout.example.json').read_text())
root=Path('outputs/simulation');root.mkdir(parents=True,exist_ok=True)
cases=[
 ('front',{}),('perspective',{'perspective':.22}),('portrait',{'rotate':90}),
 ('upside_down',{'rotate':180}),('shadow',{'shadow':True}),
 ('low_res',{'resolution':(1800,900),'perspective':.05}),
 ('high_res',{'resolution':(3600,1800),'perspective':.2}),
 ('rotation_17',{'rotate':17,'resolution':(3200,1600)}),
 ('partial_shadow',{'rotate':17,'resolution':(3200,1600),'shadow':'partial'}),
 ('marker_150mm',{'marker_size':150}),
 ('oversize',{'length':4080,'width':1060}),('different_shape',{'length':3800,'width':1200}),
 ('exif',{'exif':True}),('missing_marker',{'missing':(0,)}),
 ('wrong_layout',{'shifted_marker':1}),('blur',{'blur':8}),
 ('manual_recovery',{'contrast':0})]
rows=[]
for name,kwargs in cases:
    directory=root/name;directory.mkdir(exist_ok=True)
    image=directory/('input.jpg' if kwargs.get('exif') else 'input.png')
    corners=scene(image,**kwargs)
    (directory/'truth_corners.json').write_text(json.dumps(corners.tolist()),encoding='utf-8')
    r=inspect(image,layout,Options(marker_size_mm=kwargs.get('marker_size',100)),directory/'result',corners if name=='manual_recovery' else None)
    true_l,true_w=kwargs.get('length',4000),kwargs.get('width',1000)
    le=r['measured_length_mm']-true_l if r['success'] else None
    we=r['measured_width_mm']-true_w if r['success'] else None
    row={'image_id':name,'true_length_mm':true_l,'true_width_mm':true_w,
         'measured_length_mm':r['measured_length_mm'],'measured_width_mm':r['measured_width_mm'],
         'length_error_mm':round(le,2) if le is not None else None,
         'width_error_mm':round(we,2) if we is not None else None,
         'length_error_percent':round(100*le/true_l,3) if le is not None else None,
         'width_error_percent':round(100*we/true_w,3) if we is not None else None,
         'within_50mm':bool(abs(le)<=50 and abs(we)<=50) if le is not None else False,
         'manual':name=='manual_recovery','result':r['result'],
         'failure_reason':r['reason_code'] if not r['success'] else ''}
    rows.append(row)
    print(name,r['result'],row['length_error_mm'],row['width_error_mm'],flush=True)
with (root/'measurements.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
(root/'measurements.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
