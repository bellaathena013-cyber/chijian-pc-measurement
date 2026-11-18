import cv2
import numpy as np
import pytest
from vision.segmentation_refinement import refine_component_mask


def scene():
    quad=np.int32([[100,80],[480,100],[530,580],[50,570]])
    mask=np.zeros((650,600),np.uint8)
    cv2.fillPoly(mask,[quad],1)
    return mask,quad


def test_remove_small_island_without_hull_bridge():
    mask,quad=scene();mask[20:25,560:565]=1
    result=refine_component_mask(mask)
    assert result['quad_px'] is not None
    assert result['mask'][22,562]==0
    assert np.max(np.linalg.norm(np.array(result['quad_px'])-quad,axis=1))<2


def test_competing_regions_are_rejected():
    mask=np.zeros((600,800),np.uint8)
    mask[100:500,50:350]=1;mask[100:500,450:750]=1
    r=refine_component_mask(mask)
    assert r['quad_px'] is None
    assert 'MULTIPLE_SIGNIFICANT_REGIONS' in r['reasons']


def test_large_notch_preserved_and_not_measured_as_rectangle():
    mask,_=scene();mask[350:610,400:550]=0
    r=refine_component_mask(mask)
    assert r['quad_px'] is None
    assert not r['mask'][450,450]


def test_pixels_correct_small_mask_offset():
    mask,quad=scene()
    image=np.where(mask,220,30).astype(np.uint8)
    offset=cv2.dilate(mask,np.ones((9,9),np.uint8))
    r=refine_component_mask(offset,image)
    assert r['quad_px'] is not None
    assert np.max(np.linalg.norm(np.array(r['quad_px'])-quad,axis=1))<2.5
    assert all(e['status']=='refined' for e in r['metrics']['image_edges'])


def test_empty_clipped_and_invalid_inputs():
    assert refine_component_mask(np.zeros((80,80),np.uint8))['reasons']==['EMPTY_MASK']
    mask=np.zeros((100,100),np.uint8);mask[10:80,0:80]=1
    assert refine_component_mask(mask)['quad_px'] is None
    with pytest.raises(ValueError):refine_component_mask(np.ones((8,8,3)))
    with pytest.raises(ValueError):refine_component_mask(np.full((8,8),np.nan))
    with pytest.raises(ValueError):refine_component_mask(np.ones((8,8)),np.ones((9,8)))


def test_rounded_mask_corners_are_recovered_by_straight_edges():
    mask=np.zeros((600,700),np.uint8);mask[60:540,60:640]=1
    rounded=cv2.morphologyEx(mask,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(55,55)))
    result=refine_component_mask(rounded)
    expected=np.array([[60,60],[639,60],[639,539],[60,539]])
    assert result['quad_px'] is not None
    assert np.max(np.linalg.norm(np.array(result['quad_px'])-expected,axis=1))<2
