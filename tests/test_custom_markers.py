"""Check pattern detection independently of tape dimensions and real photos."""
import cv2
import numpy as np

from vision.custom_markers import detect_custom_markers, detect_custom_markers_robust


def test_projected_card_and_solid_distractor():
    card = np.zeros((300, 300), np.uint8)
    card[20:280, 20:280] = 255
    cv2.line(card, (20, 20), (280, 280), 0, 4)
    cv2.line(card, (280, 20), (20, 280), 0, 4)
    card[50:250, 50:250] = 0
    cv2.putText(card, '1', (110, 210), cv2.FONT_HERSHEY_SIMPLEX, 4, 255, 8)
    expected = np.float32([[120, 170], [290, 150], [280, 290], [135, 310]])
    h = cv2.getPerspectiveTransform(np.float32([[0, 0], [299, 0], [299, 299], [0, 299]]), expected)
    frame = cv2.warpPerspective(card, h, (1000, 1000), borderValue=200)
    frame[500:640, 600:740] = 0
    candidates = detect_custom_markers(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    assert len(candidates) == 1
    assert np.max(np.linalg.norm(np.array(candidates[0]['corners_px'])-expected, axis=1)) < 4
    recovered = detect_custom_markers_robust(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    assert len(recovered) == 1  # Two contour passes must not double-count a card.
    assert recovered[0]['corners_px'] == candidates[0]['corners_px']


def test_blank_image_has_no_candidates():
    assert detect_custom_markers(np.full((800, 800, 3), 180, np.uint8)) == []
    assert detect_custom_markers_robust(np.full((800, 800, 3), 180, np.uint8)) == []


def test_invalid_search_parameters():
    import pytest
    image = np.full((100, 100, 3), 180, np.uint8)
    for value in (0, float('nan'), float('inf')):
        with pytest.raises(ValueError):
            detect_custom_markers(image, max_dimension=value)
    for value in (0, .5, float('nan')):
        with pytest.raises(ValueError):
            detect_custom_markers(image, contour_tolerance=value)


def test_recovery_does_not_change_existing_three_card_calibration(monkeypatch):
    import vision.custom_markers as module
    calls = []
    baseline = [{'corners_px': [[0, 0], [20, 0], [20, 20], [0, 20]]} for _ in range(3)]
    def detector(image, **kwargs):
        calls.append(kwargs)
        return [dict(item) for item in baseline]
    monkeypatch.setattr(module, 'detect_custom_markers', detector)
    result = module.detect_custom_markers_robust(None)
    assert len(calls) == 1
    assert [m['corners_px'] for m in result] == [m['corners_px'] for m in baseline]
