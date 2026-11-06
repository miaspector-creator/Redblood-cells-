"""Fine-tune a YOLO26 classifier on BloodMNIST (8 blood cell types). Runs as a Kaggle GPU kernel.

This is the second model of the project. It is kept fully separate from the TXL-PBC detector
(kaggle/kernel/train.py): different kernel, dataset, output folder and weight file name.

Progress is printed as single-line markers that scripts/monitor_wbc.py parses from the log:
    @@STAGE <name>
    @@EPOCH key=value key=value ...
    @@TEST {json}

Environment overrides (used for local smoke tests):
    KAGGLE_INPUT, KAGGLE_WORK, YOLO_MODEL, EPOCHS, IMGSZ, BATCH, PATIENCE, FRACTION, DEVICE, SKIP_PIP=1
"""
import csv
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

MODEL = os.environ.get("YOLO_MODEL", "yolo26s-cls.pt")
EPOCHS = int(os.environ.get("EPOCHS", "60"))
IMGSZ = int(os.environ.get("IMGSZ", "224"))      # 64 px source images, upsampled for the pretrained backbone
BATCH = int(os.environ.get("BATCH", "128"))
PATIENCE = int(os.environ.get("PATIENCE", "15"))
FRACTION = float(os.environ.get("FRACTION", "1.0"))
SEED = 42

INPUT = Path(os.environ.get("KAGGLE_INPUT", "/kaggle/input"))
WORK = Path(os.environ.get("KAGGLE_WORK", "/kaggle/working"))
OUT = WORK / "outputs"
CLASSES = ["basophil", "eosinophil", "erythroblast", "immature_granulocytes",
           "lymphocyte", "monocyte", "neutrophil", "platelet"]
SPLITS = ("train", "val", "test")
IMG_EXT = {".png", ".jpg", ".jpeg"}


def stage(name):
    print(f"@@STAGE {name} t={time.time():.0f}", flush=True)


def _is_root(p: Path) -> bool:
    return all((p / s / CLASSES[0]).is_dir() for s in SPLITS)


def find_dataset_root() -> Path:
    for t in INPUT.rglob("train"):
        if t.is_dir() and _is_root(t.parent):
            return t.parent
    for z in INPUT.rglob("*.zip"):
        dest = WORK / "dataset"
        with zipfile.ZipFile(z) as zf:
            zf.extractall(dest)
        for t in dest.rglob("train"):
            if t.is_dir() and _is_root(t.parent):
                return t.parent
    raise FileNotFoundError(f"BloodMNIST train/val/test class folders not found under {INPUT}")


def images(root: Path, split: str):
    for c in CLASSES:
        for f in sorted((root / split / c).iterdir()):
            if f.suffix.lower() in IMG_EXT:
                yield f, c


def evaluate(model, root: Path, split: str, device):
    """Predict every image in a split and compute accuracy, per-class P/R/F1 and a confusion matrix
    (rows = true class, columns = predicted class)."""
    files = list(images(root, split))
    n = len(CLASSES)
    cm = [[0] * n for _ in range(n)]
    rows = []
    t0 = time.perf_counter()
    for i in range(0, len(files), 256):
        chunk = files[i:i + 256]
        results = model.predict([str(f) for f, _ in chunk], imgsz=IMGSZ, device=device, batch=64, verbose=False)
        for (f, true_c), r in zip(chunk, results):
            pred_c = model.names[int(r.probs.top1)]
            t, p = CLASSES.index(true_c), CLASSES.index(pred_c)
            cm[t][p] += 1
            top2 = [model.names[int(k)] for k in r.probs.top5[:2]]
            rows.append({"file": f"{split}/{true_c}/{f.name}", "true": true_c, "pred": pred_c,
                         "conf": round(float(r.probs.top1conf), 4), "top2_hit": true_c in top2})
    ms = (time.perf_counter() - t0) * 1000 / max(1, len(files))

    per_class, f1s, recalls = {}, [], []
    for k, c in enumerate(CLASSES):
        tp = cm[k][k]
        support = sum(cm[k])
        predicted = sum(cm[j][k] for j in range(n))
        prec = tp / predicted if predicted else 0.0
        rec = tp / support if support else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[c] = {"precision": prec, "recall": rec, "f1": f1, "support": support}
        f1s.append(f1)
        recalls.append(rec)
    total = sum(map(sum, cm))
    overall = {
        "accuracy": sum(cm[k][k] for k in range(n)) / total,
        "top2_accuracy": sum(r["top2_hit"] for r in rows) / total,
        "macro_f1": sum(f1s) / n,
        "balanced_accuracy": sum(recalls) / n,
        "images": total,
    }
    return {"split": split, "overall": overall, "per_class": per_class, "confusion_matrix": cm,
            "labels": CLASSES, "ms_per_image": ms}, rows


def main():
    stage("setup")
    if os.environ.get("SKIP_PIP") != "1":
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "ultralytics"], check=True)
    import torch
    from ultralytics import YOLO, __version__ as ul_version

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    device = os.environ.get("DEVICE") or (0 if gpu != "none" else "cpu")
    print(f"ultralytics={ul_version} torch={torch.__version__} gpu={gpu} device={device}", flush=True)

    root = find_dataset_root()
    counts = {s: {c: sum(1 for f in (root / s / c).iterdir() if f.suffix.lower() in IMG_EXT) for c in CLASSES}
              for s in SPLITS}
    print("@@DATA " + json.dumps(counts), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "run_config.json").write_text(json.dumps({
        "task": "classify", "dataset": "BloodMNIST (64x64)", "model": MODEL, "epochs": EPOCHS, "imgsz": IMGSZ,
        "batch": BATCH, "patience": PATIENCE, "fraction": FRACTION, "seed": SEED, "gpu": gpu,
        "ultralytics": ul_version, "torch": torch.__version__, "classes": CLASSES, "counts": counts,
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
        data=str(root), epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, patience=PATIENCE, fraction=FRACTION,
        device=device, seed=SEED, workers=4, flipud=0.5,   # cells have no "up": vertical flips are valid
        project=str(WORK / "runs"), name="bloodmnist_cls", exist_ok=True, plots=True, verbose=False,
    )
    run_dir = Path(model.trainer.save_dir)

    stage("testing")
    best = YOLO(run_dir / "weights" / "best.pt")
    val_metrics, _ = evaluate(best, root, "val", device)
    test_metrics, rows = evaluate(best, root, "test", device)
    metrics = {"val": val_metrics, "test": test_metrics}
    (OUT / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
    with (OUT / "test_predictions.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("@@TEST " + json.dumps(test_metrics["overall"]), flush=True)

    stage("packaging")
    shutil.copy(run_dir / "results.csv", OUT / "results.csv")
    shutil.copy(run_dir / "weights" / "best.pt", OUT / "wbc_best.pt")   # distinct name from the detector's best.pt
    for f in list(run_dir.glob("*.png")) + list(run_dir.glob("*.jpg")):
        shutil.copy(f, OUT / f"train_{f.name}")
    stage("done")


if __name__ == "__main__":
    main()
