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

# Second model (optional): white blood cell type classifier trained on BloodMNIST.
WBC_DIR = ROOT / "outputs" / "wbc_classifier"
WBC_TYPES = ["neutrophil", "lymphocyte", "monocyte", "eosinophil", "basophil", "immature_granulocytes"]
WBC_CROP_SCALE = 1.35   # BloodMNIST cells fill ~70-80% of a centered frame with neighbours visible
WBC_SOURCE_PX = 64      # BloodMNIST resolution; crops are shrunk to this so they look like training images
WBC_RECHECK_S = 30


def _wbc_model_path() -> Path | None:
    env = os.environ.get("BLOODCELL_WBC_MODEL")
    if env:
        return Path(env) if Path(env).is_file() else None
    hits = sorted(WBC_DIR.rglob("wbc_best.pt"), key=lambda p: p.stat().st_mtime) if WBC_DIR.is_dir() else []
    return hits[-1] if hits else None


def _rel(p: Path) -> str:
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


class WbcClassifier:
    def __init__(self, path: Path, device: str):
        from ultralytics import YOLO

        self.path = path
        self.model = YOLO(str(path))
        self.device = device
        self.names = {int(k): v for k, v in self.model.names.items()}
        missing = [t for t in WBC_TYPES if t not in self.names.values()]
        if missing:
            raise ValueError(f"{path.name} is not a WBC classifier (missing classes {missing})")
        self.wbc_idx = [k for k, v in self.names.items() if v in WBC_TYPES]
        args = getattr(self.model, "ckpt", {}) or {}
        self.imgsz = int((args.get("train_args") or {}).get("imgsz", 224))
        self.classify([Image.new("RGB", (WBC_SOURCE_PX, WBC_SOURCE_PX), (230, 200, 200))])  # warm up

    def classify(self, crops: list[Image.Image]) -> list[dict]:
        out = []
        for r in self.model.predict(crops, imgsz=self.imgsz, device=self.device, verbose=False):
            p = r.probs.data.float().cpu().tolist()
            raw = max(range(len(p)), key=p.__getitem__)
            # The detector already said "WBC", so spread the score over the 6 white cell types only.
            sub = {self.names[k]: p[k] for k in self.wbc_idx}
            total = sum(sub.values()) or 1.0
            ranked = sorted(((k, v / total) for k, v in sub.items()), key=lambda kv: -kv[1])
            out.append({
                "type": ranked[0][0], "conf": round(ranked[0][1], 4),
                "top": [[k, round(v, 4)] for k, v in ranked[:3]],
                "raw": self.names[raw], "raw_conf": round(p[raw], 4),
            })
        return out


def crop_cell(img: Image.Image, box_px, bg) -> Image.Image:
    """Square crop centred on the box, padded with the slide background where it runs off the image."""
    x1, y1, x2, y2 = box_px
    side = max(x2 - x1, y2 - y1) * WBC_CROP_SCALE
    left, top = (x1 + x2 - side) / 2, (y1 + y2 - side) / 2
    s = max(1, round(side))
    canvas = Image.new("RGB", (s, s), bg)
    l, t = max(0, round(left)), max(0, round(top))
    r, b = min(img.width, round(left + side)), min(img.height, round(top + side))
    if r > l and b > t:
        canvas.paste(img.crop((l, t, r, b)), (l - round(left), t - round(top)))
    return canvas.resize((WBC_SOURCE_PX, WBC_SOURCE_PX), Image.BILINEAR)


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
        self.wbc: WbcClassifier | None = None
        self.wbc_error: str | None = None
        self._wbc_checked = 0.0
        self._wbc_path: Path | None = None
        # Warm up so the first real request isn't slow.
        self.predict(Image.new("RGB", (IMGSZ, IMGSZ), (230, 200, 200)))

    def ensure_wbc(self) -> WbcClassifier | None:
        """Load the WBC classifier when its weights appear (e.g. after the Kaggle run downloads),
        or reload it if a newer wbc_best.pt shows up. Checked at most every WBC_RECHECK_S seconds."""
        now = time.monotonic()
        if now - self._wbc_checked < WBC_RECHECK_S:
            return self.wbc
        self._wbc_checked = now
        path = _wbc_model_path()
        if path is None or path == self._wbc_path:
            return self.wbc
        try:
            with self.lock:
                self.wbc = WbcClassifier(path, self.device)
            self._wbc_path, self.wbc_error = path, None
        except Exception as e:
            self._wbc_path, self.wbc_error = path, f"{path.name}: {e}"
        return self.wbc

    def predict(self, img: Image.Image) -> dict:
        w, h = img.size
        wbc = self.ensure_wbc()
        t0 = time.perf_counter()
        with self.lock:
            r = self.model.predict(img, conf=BASE_CONF, imgsz=IMGSZ, device=self.device, verbose=False)[0]
            boxes = []
            wbc_px = []
            for (x1, y1, x2, y2), c, cls in zip(
                r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()
            ):
                d = {
                    "cls": self.names[int(cls)],
                    "conf": round(c, 4),
                    # normalized 0..1 so the browser can draw at any display size
                    "box": [round(x1 / w, 5), round(y1 / h, 5), round(x2 / w, 5), round(y2 / h, 5)],
                }
                boxes.append(d)
                if d["cls"] == "WBC":
                    wbc_px.append((d, (x1, y1, x2, y2)))
            if wbc and wbc_px:
                bg = img.resize((1, 1), Image.BILINEAR).getpixel((0, 0))   # average slide colour for padding
                for (d, _), t in zip(wbc_px, wbc.classify([crop_cell(img, px, bg) for _, px in wbc_px])):
                    d["wbc"] = t
        ms = (time.perf_counter() - t0) * 1000
        return {"width": w, "height": h, "inference_ms": round(ms, 1), "detections": boxes,
                "wbc_classifier": wbc is not None}


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
    wbc = _detector.ensure_wbc()
    return {
        "ready": True,
        "model": _rel(_detector.path),
        "device": _detector.device,
        "classes": list(_detector.names.values()),
        "base_conf": BASE_CONF,
        "wbc_classifier": {"ready": True, "model": _rel(wbc.path), "types": WBC_TYPES, "imgsz": wbc.imgsz}
        if wbc else {"ready": False, "error": _detector.wbc_error},
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
