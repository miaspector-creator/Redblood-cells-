"""Upload BloodMNIST as a private Kaggle dataset (once) and push the WBC classifier GPU kernel.

Separate from scripts/kaggle_launch.py (the TXL-PBC detector): different dataset, kernel and outputs.

    .venv/bin/python scripts/kaggle_launch_wbc.py
    .venv/bin/python scripts/monitor_wbc.py
"""
import argparse
import json
import tempfile
import time
import zipfile
from pathlib import Path

from common import (KAGGLE_USER, WBC_CLASSES, WBC_DATA_DIR, WBC_DATASET_SLUG, WBC_KERNEL_DIR,
                    WBC_KERNEL_SLUG, kaggle, load_env)


def ensure_dataset():
    ref = f"{KAGGLE_USER}/{WBC_DATASET_SLUG}"
    if kaggle("datasets", "status", ref, check=False).returncode == 0:
        print(f"dataset {ref} already exists")
        wait_until_ready(ref)
        return ref
    for split in ("train", "val", "test"):
        missing = [c for c in WBC_CLASSES if not (WBC_DATA_DIR / split / c).is_dir()]
        if missing:
            raise SystemExit(f"{WBC_DATA_DIR / split} is missing class folders: {missing}")
    with tempfile.TemporaryDirectory() as tmp:
        zpath = Path(tmp) / "bloodmnist.zip"
        n = 0
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(WBC_DATA_DIR.rglob("*")):
                if f.is_file() and not f.name.startswith("."):
                    zf.write(f, Path("bloodmnist") / f.relative_to(WBC_DATA_DIR))
                    n += 1
        (Path(tmp) / "dataset-metadata.json").write_text(json.dumps({
            "title": "BloodMNIST 64", "id": ref, "licenses": [{"name": "CC-BY-4.0"}],
        }))
        print(f"uploading {n} files ({zpath.stat().st_size / 1e6:.1f} MB zipped) as private dataset {ref} ...")
        print(kaggle("datasets", "create", "-p", tmp).stdout)
    wait_until_ready(ref)
    return ref


def wait_until_ready(ref, timeout_s=1800):
    """A new dataset must finish processing before a kernel can attach it as a source."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        out = kaggle("datasets", "status", ref, check=False).stdout.strip().lower()
        if "ready" in out:
            print(f"dataset {ref} is ready")
            return
        if "error" in out or "fail" in out:
            raise SystemExit(f"dataset {ref} failed to process: {out}")
        print(f"waiting for dataset {ref} to finish processing ({out or 'pending'}) ...", flush=True)
        time.sleep(20)
    raise SystemExit(f"dataset {ref} still not ready after {timeout_s // 60} min; rerun this script later")


def push_kernel(dataset_ref, accelerator):
    (WBC_KERNEL_DIR / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{KAGGLE_USER}/{WBC_KERNEL_SLUG}",
        "title": "BloodMNIST YOLO26 cls",
        "code_file": "train_cls.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [dataset_ref],
        "competition_sources": [],
        "kernel_sources": [],
    }, indent=2))
    print(kaggle("kernels", "push", "-p", str(WBC_KERNEL_DIR), "--accelerator", accelerator).stdout)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--accelerator", default="NvidiaTeslaT4")
    args = ap.parse_args()
    load_env()
    push_kernel(ensure_dataset(), args.accelerator)
    print("Monitor with: .venv/bin/python scripts/monitor_wbc.py")
