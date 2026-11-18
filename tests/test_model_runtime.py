import subprocess
import sys

import pytest
from api.model_runtime import ModelRuntime


@pytest.fixture
def worker(tmp_path):
    folder=tmp_path/'scripts';folder.mkdir()
    (folder/'model_worker.py').write_text('''import json,sys,time,os
for line in sys.stdin:
 r=json.loads(line)
 mode=r['marker_size_mm']
 if mode==2: time.sleep(30)
 if mode==3: os._exit(1)
 print(json.dumps({'id':r['id'] if mode!=4 else 'wrong', 'ok':mode!=5}),flush=True)
''',encoding='utf8')
    runtime=ModelRuntime()
    yield runtime,tmp_path
    runtime.close()


def call(worker, mode=150, timeout=3):
    runtime,root=worker
    runtime.predict(sys.executable,root,root/'photo.png',root/'weights.pt',root/'output',mode,root/'log',timeout)


def test_reuses_process_and_closes_it(worker):
    call(worker)
    process=worker[0].process
    call(worker)
    assert worker[0].process is process
    worker[0].close()
    assert process.poll() is not None


def test_timeout_kills_worker_and_next_request_restarts(worker):
    call(worker)
    process=worker[0].process
    with pytest.raises(subprocess.TimeoutExpired): call(worker,2,.1)
    assert process.poll() is not None
    call(worker)
    assert worker[0].process is not process


@pytest.mark.parametrize('mode',[3,4,5])
def test_worker_failure_or_wrong_response_does_not_leak_into_next_request(worker,mode):
    with pytest.raises(RuntimeError): call(worker,mode)
    assert worker[0].process is None
    call(worker)


def test_warmup_is_ready_and_first_photo_reuses_worker(worker):
    runtime, root = worker
    assert not runtime.ready
    runtime.warmup(sys.executable, root, root/'weights.pt', root/'warmup.log')
    process = runtime.process
    assert runtime.ready
    call(worker)
    assert runtime.process is process
    runtime.close()
    assert not runtime.ready
