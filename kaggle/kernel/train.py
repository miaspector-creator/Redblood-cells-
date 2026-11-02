"""Fine-tune YOLO26 on TXL-PBC (WBC / RBC / Platelets). Runs as a Kaggle GPU kernel.

Progress is printed as single-line markers that scripts/monitor.py parses from the
kernel log while the run is in progress:
    @@STAGE <name>
    @@EPOCH key=value key=value ...
"""
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

MODEL = os.environ.get("YOLO_MODEL", "yolo26s.pt")
EPOCHS = int(os.environ.get("EPOCHS", "100"))
IMGSZ = int(os.environ.get("IMGSZ", "640"))
BATCH = int(os.environ.get("BATCH", "16"))
PATIENCE = int(os.environ.get("PATIENCE", "30"))
SEED = 42

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
OUT = WORK / "outputs"
NAMES = ["WBC", "RBC", "Platelets"]


def stage(name):
    print(f"@@STAGE {name} t={time.time():.0f}", flush=True)


def find_dataset_root():
    for p in INPUT.rglob("data.yaml"):
        if (p.parent / "images" / "train").is_dir():
            return p.parent
    for z in INPUT.rglob("*.zip"):
        dest = WORK / "dataset"
        with zipfile.ZipFile(z) as zf:
            zf.extractall(dest)
        for p in dest.rglob("data.yaml"):
            return p.parent
    raise FileNotFoundError("TXL-PBC data.yaml not found under /kaggle/input")


def main():
    stage("setup")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "ultralytics"], check=True)
    import torch
    from ultralytics import YOLO, __version__ as ul_version

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    print(f"ultralytics={ul_version} torch={torch.__version__} gpu={gpu}", flush=True)

    root = find_dataset_root()
    data_yaml = WORK / "txl_pbc.yaml"
    data_yaml.write_text(
        f"path: {root}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"nc: {len(NAMES)}\nnames: {NAMES}\n"
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "run_config.json").write_text(json.dumps({
        "model": MODEL, "epochs": EPOCHS, "imgsz": IMGSZ, "batch": BATCH,
        "patience": PATIENCE, "seed": SEED, "gpu": gpu,
        "ultralytics": ul_version, "torch": torch.__version__,
    }, indent=2))

    model = YOLO(MODEL)

    def on_fit_epoch_end(trainer):
        row = {"epoch": trainer.epoch + 1, "epochs": trainer.epochs}
        row.update(trainer.label_loss_items(trainer.tloss, prefix="train"))
        row.update(trainer.metrics or {})
        row.update({f"lr/pg{i}": g["lr"] for i, g in enumerate(trainer.optimizer.param_groups)})
        parts = [f"{k}={float(v):.6g}" for k, v in row.items()]
        print("@@EPOCH " + " ".join(parts), flush=True)

    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

    stage("training")
    model.train(
        data=str(data_yaml), epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, patience=PATIENCE,
        device=0 if gpu != "none" else "cpu", seed=SEED, workers=4,
        project=str(WORK / "runs"), name="txl_pbc", exist_ok=True, plots=True, verbose=False,
    )
    run_dir = Path(model.trainer.save_dir)

    stage("testing")
    best = YOLO(run_dir / "weights" / "best.pt")
    test = best.val(data=str(data_yaml), split="test", imgsz=IMGSZ, batch=BATCH,
                    project=str(WORK / "runs"), name="test", exist_ok=True, plots=True)
    box = test.box
    per_class = {}
    for i, c in enumerate(box.ap_class_index):
        p, r, ap50, ap = box.class_result(i)
        per_class[NAMES[int(c)]] = {"precision": p, "recall": r, "mAP50": ap50, "mAP50-95": ap}
    metrics = {
        "split": "test",
        "overall": {"precision": box.mp, "recall": box.mr, "mAP50": box.map50, "mAP50-95": box.map},
        "per_class": per_class,
        "confusion_matrix": test.confusion_matrix.matrix.tolist(),
        "confusion_labels": NAMES + ["background"],
        "speed_ms": test.speed,
    }
    (OUT / "test_metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    print("@@TEST " + json.dumps(metrics["overall"], default=float), flush=True)

    stage("packaging")
    shutil.copy(run_dir / "results.csv", OUT / "results.csv")
    shutil.copy(run_dir / "weights" / "best.pt", OUT / "best.pt")
    for d, prefix in ((run_dir, "train_"), (Path(test.save_dir), "test_")):
        for f in d.glob("*.png"):
            shutil.copy(f, OUT / f"{prefix}{f.name}")
        for f in d.glob("*.jpg"):
            shutil.copy(f, OUT / f"{prefix}{f.name}")
    stage("done")


if __name__ == "__main__":
    main()
