import numpy as np

from scripts.evaluate_component_segmentation import rectangle_candidate


def test_perspective_quad_is_preserved():
    corners=np.array([[180,110],[530,90],[720,700],[30,720]],np.float32)
    contour=np.vstack([a+(b-a)*t for a,b in zip(corners,np.roll(corners,-1,axis=0)) for t in np.linspace(0,1,20,endpoint=False)])
    quad=rectangle_candidate(contour)
    assert quad is not None
    assert all(np.linalg.norm(corners-point,axis=1).min()<.01 for point in quad)


def test_large_notch_cannot_be_replaced_with_rectangle():
    polygon=np.array([[0,0],[500,0],[500,300],[250,300],[250,500],[0,500]],np.float32)
    assert rectangle_candidate(polygon) is None


def test_degenerate_mask_is_rejected():
    assert rectangle_candidate([[1,1],[2,2],[3,3],[4,4]]) is None
