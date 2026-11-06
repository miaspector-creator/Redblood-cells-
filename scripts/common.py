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

# Second model: white blood cell type classifier (BloodMNIST). Kept separate from the detector above.
WBC_DATA_DIR = ROOT / "Data" / "bloodmnist"
WBC_KERNEL_DIR = ROOT / "kaggle" / "kernel_wbc"
WBC_OUTPUTS = OUTPUTS / "wbc_classifier"
WBC_DATASET_SLUG = "bloodmnist-64"
WBC_KERNEL_SLUG = "bloodmnist-yolo26-cls"
WBC_CLASSES = ["basophil", "eosinophil", "erythroblast", "immature_granulocytes",
               "lymphocyte", "monocyte", "neutrophil", "platelet"]


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
