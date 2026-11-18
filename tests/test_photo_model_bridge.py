import json

import numpy as np
import pytest
from api import photo_model
from vision.visualization import save_image


@pytest.mark.parametrize('status,length,expected', [('REVIEW', 3700, True), ('RETAKE', 3700, False), ('PASS', 3700, None), ('REVIEW', float('nan'), None)])
def test_bridge_preserves_status_and_rejects_invalid_output(tmp_path, monkeypatch, status, length, expected):
    source = tmp_path/'source.png'
    save_image(source, np.full((320, 320, 3), 180, np.uint8))
    out = tmp_path/'run'; out.mkdir()
    def run(python, root, image, weights, output, marker_size_mm, log):
        assert marker_size_mm == 150
        assert image == out/'original.png' and output == out/'model'
        work = out/'model'; work.mkdir()
        save_image(work/'prediction.jpg', np.full((320, 320, 3), 180, np.uint8))
        (work/'prediction.json').write_text(json.dumps({'weight_sha256': 'test', 'measurement': {'status': status, 'measured_length_mm': length, 'measured_width_mm': 2600}, 'markers': []}), encoding='utf-8')
    monkeypatch.setattr(photo_model.runtime, 'predict', run)
    if expected is None:
        with pytest.raises(RuntimeError): photo_model.inspect_photo(source, out, 150, 'test')
    else:
        result = photo_model.inspect_photo(source, out, 150, 'test')
        assert result['result'] == status
        assert result['success'] is expected
        assert result['measured_length_mm'] == (3700 if expected else None)
        assert result['calibration'] is None  # Old manual editor is not compatible.
        assert result['process_visualization'] == {
            'kind': 'result_replay', 'image_width': 320, 'image_height': 320,
            'component_polygon_px': None, 'boundary_px': None,
        }


def test_replay_preserves_original_coordinates_without_inventing_dimensions(tmp_path, monkeypatch):
    source = tmp_path/'source.png'
    save_image(source, np.full((320, 480, 3), 180, np.uint8))
    out = tmp_path/'run'; out.mkdir()
    polygon = [[15.5, 30], [450, 31], [445, 290], [20, 288]]
    def run(python, root, image, weights, output, marker_size_mm, log):
        output.mkdir()
        save_image(output/'prediction.jpg', np.full((240, 360, 3), 180, np.uint8))
        (output/'prediction.json').write_text(json.dumps({
            'weight_sha256': 'test', 'candidates': [{'polygon_original_px': polygon}],
            'boundary_refinement': {'quad_original_px': None},
            'measurement': {'status': 'MANUAL_REQUIRED'}, 'markers': [],
        }), encoding='utf-8')
    monkeypatch.setattr(photo_model.runtime, 'predict', run)
    result = photo_model.inspect_photo(source, out, 150, 'test')
    assert result['process_visualization']['component_polygon_px'] == polygon
    assert result['process_visualization']['image_width'] == 480
    assert result['process_visualization']['boundary_px'] is None
    assert result['measured_length_mm'] is None and not result['success']
