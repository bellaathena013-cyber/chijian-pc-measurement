import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from api import server
from tests.synthetic import scene

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "RUNS", tmp_path / "runs")
    return TestClient(server.app)

def test_demo_and_manual_and_artifacts(client):
    payload = {"demo": True, "marker_size_mm": 150}
    r = client.post('/api/inspect', data={"payload": json.dumps(payload)})
    assert r.status_code == 200
    body = r.json()
    assert body['result'] == 'PASS'
    assert abs(body['measured_length_mm']-4000)<50
    assert body['source_type'] == 'simulation'
    assert client.get(body['images']['annotated']).headers['content-type'] == 'image/png'
    assert client.get(body['download_url']).json() == body
    payload['manual_corners'] = body['boundary_original_px']
    manual = client.post('/api/inspect', data={"payload":json.dumps(payload)}).json()
    assert manual['success'] and manual['measurement_mode'] == 'manual'
    assert manual['run_id'] != body['run_id']
    assert client.get(f"/api/runs/{body['run_id']}/parameters.json").status_code == 404

def test_upload_and_bad_image(client,tmp_path):
    image = tmp_path/'sample.png'
    scene(image, marker_size=150)
    payload = {"marker_size_mm":150,"layout":json.loads(Path('config/layout.example.json').read_text()),"conditions_confirmed":True}
    with image.open('rb') as f:
        r = client.post('/api/inspect',data={"payload":json.dumps(payload)},files={"image":('photo.png',f,'image/png')})
    assert r.json()['success'] and r.json()['source_type']=='uploaded'
    r=client.post('/api/inspect',data={"payload":json.dumps(payload)},files={"image":('bad.png',b'not an image','image/png')})
    assert r.json()['result']=='INVALID_INPUT'
    assert r.json()['measured_length_mm'] is None
    assert r.json()['images']=={}

@pytest.mark.parametrize('payload',[{},[],{"demo":True,"design_length_mm":-1},{"demo":True,"marker_size_mm":"bad"},{"demo":True,"manual_corners":[[1,2]]}])
def test_invalid_input(client,payload):
    assert client.post('/api/inspect',data={"payload":json.dumps(payload)}).status_code==422

def test_health_and_untrusted_host(client):
    assert client.get('/api/health').json()['status']=='ok'
    assert client.get('/').status_code==200
    assert client.get('/api/health',headers={'Host':'untrusted.example'}).status_code==400

def test_versioned_api_and_history(client):
    assert client.get('/api/runs').json()['items']==[]
    payload={"demo":True,"marker_size_mm":150,"component_id":"history-test"}
    response=client.post('/api/v1/inspect',data={"payload":json.dumps(payload)})
    assert response.status_code==200 and response.json()['success']
    item=client.get('/api/runs').json()['items'][0]
    assert item['run_id']==response.json()['run_id']
    assert item['component_id']=='history-test' and item['source_type']=='simulation'
    schema=client.get('/openapi.json').json()
    assert '/api/v1/inspect' in schema['paths']
    assert '/api/inspect' not in schema['paths']

def test_history_skips_incomplete_and_corrupt_records(client):
    for name,content in [('incomplete','{"success":true}'),('broken','{')]:
        folder=server.RUNS/name;folder.mkdir(parents=True)
        (folder/'result.json').write_text(content)
    assert client.get('/api/runs').json()['items']==[]
