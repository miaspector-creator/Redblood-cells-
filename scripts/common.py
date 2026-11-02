"""Shared paths and Kaggle CLI helpers."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_ZIP = ROOT / "Data" / "TXL-PBC.zip"
DATA_DIR = ROOT / "Data" / "TXL-PBC"
KERNEL_DIR = ROOT / "kaggle" / "kernel"
OUTPUTS = ROOT / "outputs"
REPORTS = ROOT / "reports"
DASHBOARD = ROOT / "dashboard"
NAMES = ["WBC", "RBC", "Platelets"]

KAGGLE_USER = os.environ.get("KAGGLE_USERNAME", "miaspector")
DATASET_SLUG = "txl-pbc"
KERNEL_SLUG = "txl-pbc-yolo26"


def load_env():
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
    if "KAGGLE_API_TOKEN" not in os.environ:
        sys.exit("KAGGLE_API_TOKEN missing: add it to .env")


def kaggle(*args, check=True):
    exe = Path(sys.executable).parent / "kaggle"
    res = subprocess.run([str(exe), *args], capture_output=True, text=True)
    if check and res.returncode != 0:
        raise RuntimeError(f"kaggle {' '.join(args)} failed:\n{res.stdout}\n{res.stderr}")
    return res
