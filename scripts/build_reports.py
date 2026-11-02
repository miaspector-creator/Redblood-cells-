"""Build reports/training_report.md and reports/performance_report.md from the downloaded
Kaggle outputs, and write the "performance" section of dashboard/data.json."""
import csv
import json
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import NAMES, OUTPUTS, REPORTS
from dashboard_data import update

FIG = REPORTS / "figures"


def find(name):
    hits = sorted(OUTPUTS.rglob(name))
    return hits[0] if hits else None


def read_results():
    rows = []
    with open(find("results.csv")) as f:
        for r in csv.DictReader(f):
            rows.append({k.strip(): float(v) for k, v in r.items() if v.strip()})
    return rows


def plot_curves(rows):
    ep = [r["epoch"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for key, color in (("box_loss", "#4a6fa5"), ("cls_loss", "#d9485f"), ("dfl_loss", "#2f9e8f")):
        if f"train/{key}" in rows[0]:
            axes[0].plot(ep, [r[f"train/{key}"] for r in rows], color=color, label=f"train {key}")
            axes[0].plot(ep, [r[f"val/{key}"] for r in rows], color=color, ls="--", label=f"val {key}")
    axes[0].set_title("Loss")
    axes[0].legend(frameon=False, fontsize=8)
    for key, color in (("mAP50(B)", "#4a6fa5"), ("mAP50-95(B)", "#7c5cd6"),
                       ("precision(B)", "#2f9e8f"), ("recall(B)", "#e08a2c")):
        axes[1].plot(ep, [r[f"metrics/{key}"] for r in rows], color=color, label=key.replace("(B)", ""))
    axes[1].set_title("Validation metrics")
    axes[1].set_ylim(0, 1)
    axes[1].legend(frameon=False, fontsize=8)
    for a in axes:
        a.set_xlabel("epoch")
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "training_curves.png", dpi=120)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(find("run_config.json").read_text())
    rows = read_results()
    test = json.loads(find("test_metrics.json").read_text())
    plot_curves(rows)
    for src, dst in (("test_confusion_matrix_normalized.png", "confusion_matrix.png"),
                     ("test_BoxPR_curve.png", "pr_curve.png"),
                     ("test_val_batch0_pred.jpg", "test_predictions.jpg"),
                     ("test_val_batch0_labels.jpg", "test_labels.jpg")):
        if p := find(src):
            shutil.copy(p, FIG / dst)

    best = max(rows, key=lambda r: r["metrics/mAP50-95(B)"])
    last = rows[-1]
    stopped_early = len(rows) < cfg["epochs"]
    L = ["# Training Progress Report", "",
         "## Run configuration", "",
         "| Setting | Value |", "|---|---|"]
    for k in ("model", "epochs", "imgsz", "batch", "patience", "seed", "gpu", "ultralytics", "torch"):
        L.append(f"| {k} | {cfg[k]} |")
    L += ["", "## Progress", "",
          f"- Epochs run: **{len(rows)}** of {cfg['epochs']}"
          + (f" (early stopping, patience {cfg['patience']})" if stopped_early else ""),
          f"- Best epoch (validation mAP50-95): **{int(best['epoch'])}**, "
          f"mAP50 {best['metrics/mAP50(B)']:.3f}, mAP50-95 {best['metrics/mAP50-95(B)']:.3f}",
          f"- Final epoch: train box loss {last['train/box_loss']:.3f}, val box loss {last['val/box_loss']:.3f}", "",
          "![Training curves](figures/training_curves.png)", "",
          "## Validation metrics by epoch (every 10th + last)", "",
          "| Epoch | train box | train cls | val box | val cls | P | R | mAP50 | mAP50-95 |",
          "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        if int(r["epoch"]) % 10 == 0 or r is last or r is best:
            L.append(f"| {int(r['epoch'])} | {r['train/box_loss']:.3f} | {r['train/cls_loss']:.3f} | "
                     f"{r['val/box_loss']:.3f} | {r['val/cls_loss']:.3f} | {r['metrics/precision(B)']:.3f} | "
                     f"{r['metrics/recall(B)']:.3f} | {r['metrics/mAP50(B)']:.3f} | {r['metrics/mAP50-95(B)']:.3f} |")
    (REPORTS / "training_report.md").write_text("\n".join(L) + "\n")

    o = test["overall"]
    P = ["# Model Performance Report", "",
         f"Best checkpoint (`best.pt`, epoch {int(best['epoch'])}) evaluated on the **held-out test split** "
         "(126 images, 1,881 cells), which was never used for training or model selection.", "",
         "## Overall", "",
         "| Precision | Recall | mAP50 | mAP50-95 |", "|---:|---:|---:|---:|",
         f"| {o['precision']:.3f} | {o['recall']:.3f} | {o['mAP50']:.3f} | {o['mAP50-95']:.3f} |", "",
         "## Per class", "",
         "| Class | Precision | Recall | mAP50 | mAP50-95 |", "|---|---:|---:|---:|---:|"]
    for n in NAMES:
        c = test["per_class"].get(n)
        if c:
            P.append(f"| {n} | {c['precision']:.3f} | {c['recall']:.3f} | {c['mAP50']:.3f} | {c['mAP50-95']:.3f} |")
    sp = test.get("speed_ms", {})
    P += ["", f"Inference speed on {cfg['gpu']}: {sp.get('inference', 0):.1f} ms/image "
          f"(preprocess {sp.get('preprocess', 0):.1f} ms, postprocess {sp.get('postprocess', 0):.1f} ms).", "",
          "## Confusion matrix (normalized)", "", "![Confusion matrix](figures/confusion_matrix.png)", "",
          "## Precision–recall curve", "", "![PR curve](figures/pr_curve.png)", "",
          "## Test predictions vs. ground truth", "",
          "| Ground truth | Prediction |", "|---|---|",
          "| ![labels](figures/test_labels.jpg) | ![preds](figures/test_predictions.jpg) |"]
    (REPORTS / "performance_report.md").write_text("\n".join(P) + "\n")

    update("performance", {
        "config": cfg, "test": test, "best_epoch": int(best["epoch"]), "epochs_run": len(rows),
        "figures": {k: f"../reports/figures/{k}.{ext}" for k, ext in
                    (("confusion_matrix", "png"), ("pr_curve", "png"), ("test_predictions", "jpg"),
                     ("training_curves", "png"))},
    })
    print("wrote training_report.md and performance_report.md")


if __name__ == "__main__":
    main()
