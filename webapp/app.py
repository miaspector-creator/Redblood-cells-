"""Blood cell detector web app: upload smear images, get WBC / RBC / Platelet boxes and counts.

Run from the repo root:
    .venv/bin/python -m uvicorn webapp.app:app --port 8001
then open http://localhost:8001
"""
import io
import ipaddress
import os
import socket
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urljoin, urlparse

import requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"
SAMPLES_DIR = ROOT / "Data" / "TXL-PBC" / "images" / "test"
MODEL_CANDIDATES = [
    ROOT / "outputs" / "runs" / "txl_pbc" / "weights" / "best.pt",
    ROOT / "outputs" / "outputs" / "best.pt",
    ROOT / "outputs" / "best.pt",
]
# Predict at a low threshold once; the browser filters by the user's slider without re-running.
BASE_CONF = 0.05
IMGSZ = 640
MAX_BYTES = 25 * 1024 * 1024
MAX_SAMPLES = 12


def _model_path() -> Path:
    env = os.environ.get("BLOODCELL_MODEL")
    if env:
        return Path(env)
    for p in MODEL_CANDIDATES:
        if p.is_file():
            return p
    raise FileNotFoundError(
        "No trained weights found. Put best.pt at outputs/runs/txl_pbc/weights/best.pt "
        "or set BLOODCELL_MODEL=/path/to/best.pt"
    )


def _device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Detector:
    def __init__(self):
        from ultralytics import YOLO

        self.path = _model_path()
        self.model = YOLO(str(self.path))
        self.names = {int(k): v for k, v in self.model.names.items()}
        self.device = _device()
        self.lock = threading.Lock()
        # Warm up so the first real request isn't slow.
        self.predict(Image.new("RGB", (IMGSZ, IMGSZ), (230, 200, 200)))

    def predict(self, img: Image.Image) -> dict:
        w, h = img.size
        t0 = time.perf_counter()
        with self.lock:
            r = self.model.predict(img, conf=BASE_CONF, imgsz=IMGSZ, device=self.device, verbose=False)[0]
        ms = (time.perf_counter() - t0) * 1000
        boxes = []
        for (x1, y1, x2, y2), c, cls in zip(
            r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()
        ):
            boxes.append({
                "cls": self.names[int(cls)],
                "conf": round(c, 4),
                # normalized 0..1 so the browser can draw at any display size
                "box": [round(x1 / w, 5), round(y1 / h, 5), round(x2 / w, 5), round(y2 / h, 5)],
            })
        return {"width": w, "height": h, "inference_ms": round(ms, 1), "detections": boxes}


_detector: Detector | None = None
_detector_error: str | None = None


@asynccontextmanager
async def lifespan(_app):
    global _detector, _detector_error
    try:
        _detector = Detector()
    except Exception as e:  # keep the server up so the UI can show the problem
        _detector_error = str(e)
    yield


app = FastAPI(title="Blood Cell Detector", lifespan=lifespan)


def _get_detector() -> Detector:
    if _detector is None:
        raise HTTPException(503, _detector_error or "Model is still loading")
    return _detector


def _open_image(data: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img)
        return img.convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "Not a readable image (use PNG, JPEG, TIFF, BMP or WebP)")


@app.get("/api/info")
def info():
    if _detector is None:
        return {"ready": False, "error": _detector_error}
    return {
        "ready": True,
        "model": str(_detector.path.relative_to(ROOT)) if _detector.path.is_relative_to(ROOT) else str(_detector.path),
        "device": _detector.device,
        "classes": list(_detector.names.values()),
        "base_conf": BASE_CONF,
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    det = _get_detector()
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Image is larger than 25 MB")
    out = det.predict(_open_image(data))
    out["filename"] = file.filename
    return out


def _sample_names() -> list[str]:
    if not SAMPLES_DIR.is_dir():
        return []
    return sorted(p.name for p in SAMPLES_DIR.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})[:MAX_SAMPLES]


@app.get("/api/samples")
def samples():
    return {"samples": _sample_names()}


@app.get("/api/samples/{name}")
def sample(name: str):
    if name not in _sample_names():  # whitelist; blocks path traversal
        raise HTTPException(404, "Unknown sample")
    return FileResponse(SAMPLES_DIR / name)


# ---------- load an image from a web link ----------
# Browsers block cross-site image reads (CORS), so the server fetches the link instead.
FETCH_TIMEOUT = 15
MAX_REDIRECTS = 5
FETCH_HEADERS = {"User-Agent": "BloodCellDetector/1.0", "Accept": "image/*,*/*;q=0.5"}


class UrlIn(BaseModel):
    url: str


def _check_public(url: str) -> None:
    """Refuse non-http(s) links and anything that resolves to a private/local address."""
    u = urlparse(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        raise HTTPException(400, "Link must start with http:// or https://")
    try:
        infos = socket.getaddrinfo(u.hostname, u.port or (443 if u.scheme == "https" else 80))
    except socket.gaierror:
        raise HTTPException(400, f"Could not find the website {u.hostname}")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if not ip.is_global:
            raise HTTPException(400, "Links to local or private network addresses are not allowed")


def _download(url: str) -> tuple[bytes, str]:
    for _ in range(MAX_REDIRECTS + 1):
        _check_public(url)
        try:
            r = requests.get(url, headers=FETCH_HEADERS, timeout=FETCH_TIMEOUT, stream=True, allow_redirects=False)
        except requests.RequestException as e:
            raise HTTPException(502, f"Could not download the link ({e.__class__.__name__})")
        if r.is_redirect:
            url = urljoin(url, r.headers.get("location", ""))
            r.close()
            continue
        if r.status_code != 200:
            raise HTTPException(502, f"The website answered with HTTP {r.status_code}")
        if int(r.headers.get("content-length") or 0) > MAX_BYTES:
            raise HTTPException(413, "Image is larger than 25 MB")
        buf = bytearray()
        for chunk in r.iter_content(64 * 1024):
            buf += chunk
            if len(buf) > MAX_BYTES:
                raise HTTPException(413, "Image is larger than 25 MB")
        return bytes(buf), url
    raise HTTPException(502, "Too many redirects")


@app.post("/api/fetch-image")
def fetch_image(body: UrlIn):
    data, final_url = _download(body.url.strip())
    img = _open_image(data)  # 400 if it's a web page or other non-image
    name = unquote(PurePosixPath(urlparse(final_url).path).name) or "web-image"
    # Re-encode as PNG so every browser can preview it (TIFF, HEIC, etc.)
    out = io.BytesIO()
    img.save(out, format="PNG")
    stem = name.rsplit(".", 1)[0][:80] or "web-image"
    return Response(out.getvalue(), media_type="image/png",
                    headers={"X-Filename": f"{stem}.png".encode("ascii", "ignore").decode() or "web-image.png"})


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
