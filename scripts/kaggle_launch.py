"""Upload TXL-PBC as a private Kaggle dataset (once) and push the GPU training kernel."""
import argparse
import json
import shutil
import tempfile
from pathlib import Path

from common import (DATA_ZIP, DATASET_SLUG, KAGGLE_USER, KERNEL_DIR, KERNEL_SLUG,
                    kaggle, load_env)


def ensure_dataset():
    ref = f"{KAGGLE_USER}/{DATASET_SLUG}"
    if kaggle("datasets", "status", ref, check=False).returncode == 0:
        print(f"dataset {ref} already exists")
        return ref
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy(DATA_ZIP, Path(tmp) / DATA_ZIP.name)
        (Path(tmp) / "dataset-metadata.json").write_text(json.dumps({
            "title": "TXL-PBC", "id": ref, "licenses": [{"name": "unknown"}],
        }))
        print(f"uploading {DATA_ZIP.name} as private dataset {ref} ...")
        print(kaggle("datasets", "create", "-p", tmp).stdout)
    return ref


def push_kernel(dataset_ref, accelerator):
    (KERNEL_DIR / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{KAGGLE_USER}/{KERNEL_SLUG}",
        "title": "TXL-PBC YOLO26",
        "code_file": "train.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [dataset_ref],
        "competition_sources": [],
        "kernel_sources": [],
    }, indent=2))
    print(kaggle("kernels", "push", "-p", str(KERNEL_DIR), "--accelerator", accelerator).stdout)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--accelerator", default="NvidiaTeslaT4")
    args = ap.parse_args()
    load_env()
    push_kernel(ensure_dataset(), args.accelerator)
    print("Monitor with: .venv/bin/python scripts/monitor.py")
