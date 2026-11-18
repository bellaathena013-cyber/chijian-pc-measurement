import cv2
import numpy as np
import pytest
pytest.importorskip('scipy')
from vision.free_square_calibration import measure_with_free_squares, fit_free_squares, measure_with_guarded_marker_update
from vision.free_square_calibration import measure_with_selective_marker_update
from vision.quality_check import InspectionError
from vision.perspective import transform

def scene():
    centers=np.array([[180,170],[3800,260],[3700,900],[200,950]],float)
    q=[]
    for center,angle in zip(centers,[.15,.9,1.7,-.3]):
        rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        q.append(np.array([[-75,-75],[75,-75],[75,75],[-75,75]])@rotation.T+center)
    boundary=np.array([[0,0],[4000,0],[4000,1100],[0,1100]],float)
    h=np.array([[.65,.23,400],[-.02,.8,300],[.00003,.0003,1.]])
    return transform(np.array(q),h),transform(boundary,h)

def test_unknown_positions_and_rotations():
    markers,boundary=scene()
    r=measure_with_free_squares(markers,boundary)
    assert abs(r['measured_length_mm']-4000)<.01
    assert abs(r['measured_width_mm']-1100)<.01
    assert r['calibration']['uses_known_marker_layout'] is False
    assert r['status']=='EXPERIMENTAL_MEASUREMENT'


def test_selective_update_keeps_good_correction_and_rejects_bad_one():
    truth,boundary=scene()
    original=truth.copy();original[0,1]+=[1,-1]
    proposed=truth.copy()
    center=proposed[1].mean(axis=0)
    proposed[1]=center+(proposed[1]-center)*1.2
    result=measure_with_selective_marker_update(original,proposed,boundary)
    assert result['selective_marker_update']['accepted_indices']==[0]
    assert result['status']=='REVIEW'
    assert abs(result['measured_length_mm']-4000)<.01
    assert np.allclose(result['calibration_corners_px'],truth)

def test_marker_scale_is_only_scale_source():
    markers,boundary=scene()
    r=measure_with_free_squares(markers,boundary,300)
    assert abs(r['measured_length_mm']-8000)<.01
    assert abs(r['measured_width_mm']-2200)<.01

def test_insufficient_and_crossed():
    markers,boundary=scene()
    with pytest.raises(InspectionError):fit_free_squares(markers[:2])
    with pytest.raises(InspectionError):measure_with_free_squares(markers,boundary[[0,2,1,3]])

def test_inconsistent_square_is_not_accepted():
    markers,boundary=scene();markers[0,1]+=[25,-20]
    r=measure_with_free_squares(markers,boundary)
    assert r['status']=='REVIEW'


def test_refined_corners_cannot_silently_change_scale():
    markers,boundary=scene()
    centers=markers.mean(axis=1,keepdims=True)
    inflated=centers+(markers-centers)*1.1
    r=measure_with_guarded_marker_update(markers,inflated,boundary)
    assert r['marker_update_guard']['accepted'] is False
    assert r['status']=='REVIEW'
    assert abs(r['measured_length_mm']-4000)<.01


def test_identical_marker_update_preserves_measurement():
    markers,boundary=scene()
    r=measure_with_guarded_marker_update(markers,markers.copy(),boundary)
    assert r['marker_update_guard']['accepted'] is True
    assert abs(r['measured_length_mm']-4000)<.01
