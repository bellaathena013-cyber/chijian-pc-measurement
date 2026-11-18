"""One local model subprocess, with bounded waits and restart after failure."""
import atexit
import json
import queue
import subprocess
import threading
import uuid


class ModelRuntime:
    def __init__(self):
        self.process = None
        self.responses = None
        self.lock = threading.Lock()
        self.ready = False

    def _stop(self):
        self.ready = False
        process, self.process = self.process, None
        if process is not None:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=10)
            process.stdin.close()
            process.stdout.close()

    def close(self):
        with self.lock:
            self._stop()

    def _start(self, python, root):
        self._stop()
        self.responses = queue.Queue()
        self.process = subprocess.Popen(
            [str(python), '-u', str(root/'scripts/model_worker.py')], cwd=root,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding='utf-8', bufsize=1,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        def read(process, responses):
            try:
                for line in process.stdout:
                    responses.put(json.loads(line))
            except (ValueError, OSError):
                pass
            finally:
                responses.put(None)
        threading.Thread(target=read, args=(self.process,self.responses), daemon=True).start()

    def predict(self, python, root, image, weights, output, marker_size_mm, log, timeout=180, operation="predict"):
        with self.lock:
            if self.process is None or self.process.poll() is not None:
                self._start(python, root)
            request = {'operation': operation, 'id': uuid.uuid4().hex, 'image': str(image), 'weights': str(weights),
                       'output': str(output), 'marker_size_mm': marker_size_mm, 'log': str(log)}
            try:
                self.process.stdin.write(json.dumps(request)+'\n')
                self.process.stdin.flush()
                response = self.responses.get(timeout=timeout)
                if not isinstance(response, dict) or response.get('id') != request['id'] or not response.get('ok'):
                    raise RuntimeError('本地模型处理失败，请查看本次运行日志。')
                self.ready = True
            except queue.Empty as exc:
                self._stop()
                raise subprocess.TimeoutExpired('local-model-worker', timeout) from exc
            except (OSError, RuntimeError):
                self._stop()
                raise

    def warmup(self, python, root, weights, log):
        self.predict(python, root, None, weights, None, 150, log, operation="warmup")

runtime = ModelRuntime()
atexit.register(runtime.close)
