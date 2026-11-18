import cv2
import numpy as np
from vision.custom_marker_refinement import refine_marker_candidate, calibrated_marker_candidates


def test_outer_black_edge_not_inner_white_edge():
    image=np.full((500,500),210,np.uint8)
    image[100:300,100:300]=0
    image[114:286,114:286]=255
    cv2.line(image,(114,114),(286,286),0,3)
    cv2.line(image,(286,114),(114,286),0,3)
    image[133:267,133:267]=0
    cv2.putText(image,'1',(175,240),cv2.FONT_HERSHEY_SIMPLEX,3,255,5)
    expected=np.array([[99.5,99.5],[299.5,99.5],[299.5,299.5],[99.5,299.5]])
    candidate={'corners_px':[[98,98],[301,98],[301,301],[98,301]],'candidate_index':0}
    result=refine_marker_candidate(image,candidate)
    assert result['outer_edge_refinement']['status']=='refined'
    assert np.max(np.linalg.norm(np.array(result['corners_px'])-expected,axis=1))<1.5
    assert result['candidate_index']==0


def test_no_edge_keeps_original_without_inventing_corners():
    candidate={'corners_px':[[50,50],[150,50],[150,150],[50,150]]}
    result=refine_marker_candidate(np.full((200,200),120,np.uint8),candidate)
    assert result['outer_edge_refinement']['status']=='kept_original'
    assert result['corners_px']==candidate['corners_px']


def test_display_uses_calibration_corners_instead_of_rejected_proposal():
    original=[[50,50],[150,50],[150,150],[50,150]]
    proposed=[[48,48],[152,48],[152,152],[48,152]]
    candidates=[{'corners_px':proposed,'candidate_index':0}]
    result=calibrated_marker_candidates(candidates,{'calibration_corners_px':[original]})
    assert result[0]['corners_px']==original
    assert result[0]['proposed_corners_px']==proposed
    assert candidates[0]['corners_px']==proposed
    assert result[0]['min_edge_px']==100
