"""Local workbench and multipart API. Bind to loopback until deployment is reviewed."""
import json
import asyncio
import math
import subprocess
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware
from vision.pipeline import inspect
from vision.quality_check import Options
from vision.quality_check import InspectionError
from api import photo_model

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "outputs" / "web"
LOCK = threading.Lock()
@asynccontextmanager
async def lifespan(app):
    def prepare():
        try:
            log = ROOT/'outputs/model-warmup.log'
            log.parent.mkdir(parents=True, exist_ok=True)
            photo_model.runtime.warmup(photo_model.PYTHON, ROOT, photo_model.WEIGHTS, log)
        except Exception:
            import logging
            logging.exception('Local model warmup failed; a photo request can restart it')
    preparation = None
    if photo_model.available():
        preparation = threading.Thread(target=prepare, daemon=True)
        preparation.start()
    try:
        yield
    finally:
        if preparation is not None:
            await asyncio.to_thread(preparation.join)
        await asyncio.to_thread(photo_model.runtime.close)


app = FastAPI(title="PC 构件尺寸测量", version="0.1.0", lifespan=lifespan)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])
app.mount("/static", StaticFiles(directory=ROOT / "app"), name="static")

@app.get("/")
def index():
    return FileResponse(ROOT / "app" / "index.html")

@app.get("/history")
def history_page():
    return FileResponse(ROOT / "app" / "history.html")

@app.get("/api/health")
def health():
    process = photo_model.runtime.process
    return {"status": "ok", "scope": "local-prototype", "photo_model_available": photo_model.available(), "photo_model_ready": bool(photo_model.runtime.ready and process and process.poll() is None)}


@app.post('/api/v1/photo-inspect', summary='本地模型照片识别与试验测量')
def photo_inspect(payload: str = Form(...), image: UploadFile = File(...)):
    try:
        try:
            params = json.loads(payload)
            if not isinstance(params, dict):
                raise ValueError('参数必须是 JSON 对象。')
            size = params.get('marker_size_mm', 150)
            if isinstance(size, bool) or not isinstance(size, (int, float)) or not math.isfinite(size) or size <= 0:
                raise ValueError('标志外框边长必须是有效的正数。')
            component = params.get('component_id', '未命名构件')
            if not isinstance(component, str) or len(component) > 100:
                raise ValueError('构件编号最多 100 字。')
            if params.get('conditions_confirmed') is not True:
                raise ValueError('请确认标志黑色外框实测尺寸，且与构件上表面齐平。')
        except (ValueError, TypeError) as exc:
            raise HTTPException(422, str(exc)) from exc
        if not photo_model.available():
            raise HTTPException(503, '本地照片模型或运行环境缺失，请恢复训练环境与模型权重。')
        if not LOCK.acquire(blocking=False):
            raise HTTPException(429, '正在处理另一张照片，请稍后重试。')
        try:
            data = image.file.read(20 * 1024 * 1024 + 1)
            if len(data) > 20 * 1024 * 1024:
                raise HTTPException(413, '照片不能超过 20 MB。')
            run_id = uuid.uuid4().hex
            out = RUNS/run_id
            out.mkdir(parents=True)
            source = out/'upload.bin'
            source.write_bytes(data)
            try:
                result = photo_model.inspect_photo(source, out, size, component)
            except InspectionError as exc:
                raise HTTPException(422, str(exc)) from exc
            except subprocess.TimeoutExpired as exc:
                raise HTTPException(504, '照片处理超时，请缩小照片后重试。') from exc
            except (RuntimeError, OSError, ValueError, KeyError) as exc:
                raise HTTPException(500, '照片模型处理失败，未生成有效结果。请检查本机运行日志。') from exc
            result['run_id'] = run_id
            result['images'] = {k: f'/api/runs/{run_id}/{Path(v).name}' for k, v in result['images'].items()}
            result['download_url'] = f'/api/runs/{run_id}/result.json'
            (out/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
            return result
        finally:
            LOCK.release()
    finally:
        image.file.close()

@app.get("/api/runs")
def history():
    items = []
    if RUNS.exists():
        files = sorted(RUNS.glob("*/result.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for path in files:
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                # Only completed web responses are shown; pipeline writes an intermediate result.
                if record.get("run_id") != path.parent.name or "source_type" not in record:
                    continue
                items.append({key: record.get(key) for key in ("run_id", "component_id", "result", "source_type", "measurement_mode", "measured_length_mm", "measured_width_mm")}
                             | {"created_at": path.stat().st_mtime})
            except (OSError, ValueError):
                continue
            if len(items) == 30:
                break
    return {"items": items, "limit": 30}

@app.post("/api/inspect", include_in_schema=False)
@app.post("/api/v1/inspect", summary="测量构件照片", description="multipart：image 为照片，payload 为 JSON 字符串。HTTP 200 不代表质检通过，必须检查 success 和 result。人工角点采用 EXIF 归一化原图坐标。")
def inspect_upload(payload: str = Form(...), image: UploadFile | None = File(None)):
    try:
        p = json.loads(payload)
        if not isinstance(p, dict):
            raise ValueError()
        demo = p.get("demo") is True
        layout = json.loads((ROOT / "config/layout.example.json").read_text(encoding="utf-8")) if demo else p.get("layout")
        if not isinstance(layout, dict):
            raise ValueError("请提供现场布局 JSON。")
        if not demo and p.get("conditions_confirmed") is not True:
            raise ValueError("请先核对标志尺寸、布局和共面条件。")
        opts = Options(**{k: p[k] for k in ("design_length_mm", "design_width_mm", "marker_size_mm", "tolerance_mm", "component_id") if k in p})
        opts.validate()
        corners = p.get("manual_corners")
        if corners is not None and (not isinstance(corners, list) or len(corners) != 4 or any(not isinstance(v, list) or len(v) != 2 for v in corners)):
            raise ValueError("人工边界必须包含四个坐标点。")
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, str(exc) or "参数格式无效。") from exc
    if not LOCK.acquire(blocking=False):
        raise HTTPException(429, "正在处理另一张照片，请稍后重试。")
    try:
        run_id = uuid.uuid4().hex
        out = RUNS / run_id
        out.mkdir(parents=True)
        if demo:
            source = ROOT / "outputs/simulation/marker_150mm/input.png"
            if not source.exists():
                raise HTTPException(409, "演示图片缺失，请先运行 scripts/run_simulation.py。")
        else:
            if image is None:
                raise HTTPException(422, "请选择照片。")
            data = image.file.read(20 * 1024 * 1024 + 1)
            if len(data) > 20 * 1024 * 1024:
                raise HTTPException(413, "照片不能超过 20 MB。")
            source = out / "upload.bin"
            source.write_bytes(data)
        result = inspect(source, layout, opts, out, manual_corners=corners)
        result["run_id"] = run_id
        result["source_type"] = "simulation" if demo else "uploaded"
        result["images"] = {k: f"/api/runs/{run_id}/{Path(v).name}" for k, v in result["images"].items()}
        result["download_url"] = f"/api/runs/{run_id}/result.json"
        (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        return result
    finally:
        LOCK.release()
        if image is not None:
            image.file.close()

@app.get("/api/runs/{run_id}/{filename}")
def artifact(run_id: str, filename: str):
    if len(run_id) != 32 or any(c not in "0123456789abcdef" for c in run_id) or filename not in {"original.png", "annotated.png", "corrected.png", "corrected_annotated.png", "marker-details.png", "result.json"}:
        raise HTTPException(404)
    path = RUNS / run_id / filename
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(path, filename=filename if filename.endswith(".json") else None)
