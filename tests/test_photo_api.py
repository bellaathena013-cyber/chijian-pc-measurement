import json
import subprocess

import pytest
from fastapi.testclient import TestClient
from api import server, photo_model

PARAMS = {'component_id': '现场构件', 'marker_size_mm': 150, 'conditions_confirmed': True}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(server, 'RUNS', tmp_path/'runs')
    monkeypatch.setattr(photo_model, 'available', lambda: True)
    return TestClient(server.app)


def post(client, params=None, data=b'photo'):
    return client.post('/api/v1/photo-inspect', data={'payload': json.dumps(PARAMS if params is None else params)}, files={'image': ('photo.jpg', data, 'image/jpeg')})


def test_photo_result_and_history_keep_review(client, monkeypatch):
    def inspect(source, out, size, component):
        assert size == 150 and component == '现场构件'
        (out/'annotated.png').write_bytes(b'preview')
        return {'success': True, 'result': 'REVIEW', 'component_id': component, 'measurement_mode': 'photo_model', 'source_type': 'uploaded', 'measured_length_mm': 3677.1, 'measured_width_mm': 2629.2, 'images': {'annotated': str(out/'annotated.png')}}
    monkeypatch.setattr(photo_model, 'inspect_photo', inspect)
    response = post(client)
    assert response.status_code == 200
    body = response.json()
    assert body['result'] == 'REVIEW'
    assert 'design_length_mm' not in body
    assert client.get(body['download_url']).json() == body
    assert client.get(body['images']['annotated']).status_code == 200
    assert client.get('/api/runs').json()['items'][0]['measurement_mode'] == 'photo_model'
    assert client.get(f"/api/runs/{body['run_id']}/model.log").status_code == 404


@pytest.mark.parametrize('params', [{}, [], {**PARAMS, 'marker_size_mm': 0}, {**PARAMS, 'marker_size_mm': True}, {**PARAMS, 'marker_size_mm': float('nan')}])
def test_photo_parameters_rejected(client, params):
    assert post(client, params).status_code == 422


def test_missing_model_and_busy(client, monkeypatch):
    monkeypatch.setattr(photo_model, 'available', lambda: False)
    assert post(client).status_code == 503
    monkeypatch.setattr(photo_model, 'available', lambda: True)
    server.LOCK.acquire()
    try:
        assert post(client).status_code == 429
    finally:
        server.LOCK.release()


def test_bad_image_before_worker(client):
    assert post(client).status_code == 422
    assert client.get('/api/runs').json()['items'] == []


def test_timeout_releases_lock_and_no_history(client, monkeypatch):
    def fail(*args):
        raise subprocess.TimeoutExpired('model', 180)
    monkeypatch.setattr(photo_model, 'inspect_photo', fail)
    assert post(client).status_code == 504
    assert not server.LOCK.locked()
    assert client.get('/api/runs').json()['items'] == []
