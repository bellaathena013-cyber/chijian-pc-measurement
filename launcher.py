"""Recipient launcher: isolated setup, local service, browser, clean shutdown."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import webbrowser

ROOT = Path(__file__).resolve().parent
URL = 'http://127.0.0.1:8765'

def ready():
    try:
        with urllib.request.urlopen(URL + '/api/health', timeout=1) as r:
            return json.load(r) == {'status': 'ok', 'scope': 'local-prototype'}
    except (OSError, ValueError):
        return False

def main():
    os.chdir(ROOT)
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError('Please install Python 3.12 (64-bit).')
    if ready():
        webbrowser.open(URL)
        return
    env = ROOT / '.venv'
    python = env / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if not python.exists():
        print('Creating local Python environment...', flush=True)
        subprocess.run([sys.executable, '-m', 'venv', str(env)], check=True)
    requirements = ROOT / 'requirements.txt'
    digest = hashlib.sha256(requirements.read_bytes()).hexdigest()
    stamp = env / '.requirements-sha256'
    if not stamp.exists() or stamp.read_text() != digest:
        print('Installing dependencies. Internet access is required on first launch.', flush=True)
        subprocess.run([str(python), '-m', 'pip', 'install', '-r', str(requirements)], check=True)
        stamp.write_text(digest)
    process = subprocess.Popen([str(python), '-m', 'uvicorn', 'api.server:app', '--host', '127.0.0.1', '--port', '8765'], cwd=ROOT)
    try:
        for _ in range(60):
            if process.poll() is not None:
                raise RuntimeError('Server could not start; check the error above and port 8765.')
            if ready():
                print('Ready: ' + URL + '  (Ctrl+C to stop)', flush=True)
                webbrowser.open(URL)
                break
            time.sleep(.5)
        else:
            raise RuntimeError('Server startup timed out.')
        process.wait()
    except KeyboardInterrupt:
        pass
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=10)

if __name__ == '__main__':
    main()
