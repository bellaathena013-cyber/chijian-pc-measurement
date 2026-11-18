from pathlib import Path
import copy
import json
import numpy as np
import pytest
from vision.pipeline import inspect
from vision.quality_check import Options
from vision.measurement import measure
from tests.synthetic import scene

LAYOUT=json.loads(Path('config/layout.example.json').read_text())


@pytest.mark.parametrize('resolution,perspective,rotate,shadow',[
    ((2800,1400),0,0,False),((2800,1400),.12,0,False),((2800,1400),.22,0,False),
    ((2400,1200),.08,90,False),((3200,1600),.18,180,False),((3600,1800),.2,270,True),
    ((2800,1400),.12,0,True),((1800,900),.05,0,False),
    ((3200,1600),.12,17,False),((3200,1600),.12,-12,False)])
def test_auto_measurement(tmp_path,resolution,perspective,rotate,shadow):
    image=tmp_path/'input.png'
    scene(image,resolution=resolution,perspective=perspective,rotate=rotate,shadow=shadow)
    result=inspect(image,LAYOUT,output_dir=tmp_path/'out')
    assert result['success'],result
    assert abs(result['measured_length_mm']-4000)<=50,result
    assert abs(result['measured_width_mm']-1000)<=50,result
    assert result['result'] in ['PASS','REVIEW']


@pytest.mark.parametrize('length,width',[(4080,1060),(3800,1200),(3972,1015)])
def test_design_is_not_truth(tmp_path,length,width):
    image=tmp_path/'input.png';scene(image,length=length,width=width)
    r=inspect(image,LAYOUT,output_dir=tmp_path/'out')
    assert r['success'],r
    assert abs(r['measured_length_mm']-length)<30
    assert abs(r['measured_width_mm']-width)<30
    if length==4080: assert r['result']=='REVIEW' and not r['within_tolerance']


@pytest.mark.parametrize('size',[150,200])
def test_marker_size_configurable(tmp_path,size):
    image=tmp_path/'input.png';scene(image,marker_size=size)
    r=inspect(image,LAYOUT,Options(marker_size_mm=size),tmp_path/'out')
    assert r['success'],r
    assert abs(r['measured_length_mm']-4000)<50


def test_manual_and_exif(tmp_path):
    image=tmp_path/'input.jpg';truth=scene(image,exif=True,rotate=90)
    r=inspect(image,LAYOUT,output_dir=tmp_path/'out',manual_corners=truth)
    assert r['success'],r
    assert abs(r['measured_length_mm']-4000)<30
    assert abs(r['measured_width_mm']-1000)<30
    assert r['measurement_mode']=='manual'


@pytest.mark.parametrize('kwargs',[
    {'missing':(0,)},{'missing':(0,1,2,3)}, {'shifted_marker':1}, {'blur':8},
    {'resolution':(800,400)},{'component':False},{'perspective':.70},
    {'length':4700,'width':1000}])
def test_unreliable_input_never_passes(tmp_path,kwargs):
    image=tmp_path/'input.png';scene(image,**kwargs)
    r=inspect(image,LAYOUT,output_dir=tmp_path/'out')
    assert not r['success'],r
    assert r['result'] in ['RETAKE','MANUAL_REQUIRED']
    assert r['measured_length_mm'] is None


def test_manual_can_recover_boundary_failure(tmp_path):
    image=tmp_path/'input.png'
    truth=scene(image,contrast=0)
    auto=inspect(image,LAYOUT,output_dir=tmp_path/'auto')
    assert auto['result']=='MANUAL_REQUIRED',auto
    manual=inspect(image,LAYOUT,output_dir=tmp_path/'manual',manual_corners=truth)
    assert manual['success'],manual
    assert abs(manual['measured_length_mm']-4000)<30


def test_partial_shadow_routes_to_manual(tmp_path):
    image=tmp_path/'input.png'
    truth=scene(image,resolution=(3200,1600),rotate=17,shadow='partial')
    auto=inspect(image,LAYOUT,output_dir=tmp_path/'auto')
    # Rectangular hard shadow is a competing contour: do not guess which object is intended.
    assert auto['result']=='MANUAL_REQUIRED' and auto['reason_code']=='BOUNDARY_AMBIGUOUS',auto
    manual=inspect(image,LAYOUT,output_dir=tmp_path/'manual',manual_corners=truth)
    assert manual['success'],manual
    assert abs(manual['measured_length_mm']-4000)<30
    assert abs(manual['measured_width_mm']-1000)<30


def test_manual_cannot_bypass_calibration(tmp_path):
    image=tmp_path/'input.png';truth=scene(image,missing=(0,))
    r=inspect(image,LAYOUT,output_dir=tmp_path/'out',manual_corners=truth)
    assert not r['success'] and r['reason_code']=='MISSING_MARKERS'


def test_crossed_corners_rejected(tmp_path):
    image=tmp_path/'input.png';truth=scene(image)
    r=inspect(image,LAYOUT,output_dir=tmp_path/'out',manual_corners=truth[[0,2,1,3]])
    assert r['result']=='INVALID_INPUT'


def test_layout_and_marker_size_mismatch(tmp_path):
    image=tmp_path/'input.png';scene(image)
    wrong=copy.deepcopy(LAYOUT);wrong['markers'][1]['center_mm'][0]+=200
    for layout,options in [(wrong,Options()),(LAYOUT,Options(marker_size_mm=150))]:
        r=inspect(image,layout,options,tmp_path/'out')
        assert not r['success'],r


def test_tolerance_uses_unrounded_values():
    for delta,expected in [(50,'PASS'),(50.001,'REVIEW'),(-50,'PASS')]:
        p=np.array([[0,0],[4000+delta,0],[4000+delta,1000],[0,1000]])
        assert measure(p,Options())['result']==expected


def test_invalid_image_parameters_and_layout(tmp_path):
    image=tmp_path/'bad.png';image.write_bytes(b'not an image')
    for opts in [Options(),Options(marker_size_mm=0),Options(tolerance_mm=-1),Options(tolerance_mm=float('nan'))]:
        r=inspect(image,LAYOUT,opts,tmp_path/'out')
        assert r['result']=='INVALID_INPUT'
        assert 'NaN' not in (tmp_path/'out/result.json').read_text(encoding='utf-8')
    r=inspect(image,{'markers':[]},output_dir=tmp_path/'out')
    assert r['reason_code']=='INVALID_LAYOUT'
