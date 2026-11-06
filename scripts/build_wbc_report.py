"""Build reports/wbc_classifier_report.md (+ figures in reports/figures/wbc/) from the downloaded
WBC classifier outputs. Does not touch the detector's reports.

    .venv/bin/python scripts/build_wbc_report.py
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from common import REPORTS, WBC_CLASSES, WBC_DATA_DIR, WBC_OUTPUTS

NICE = {"basophil": "Basophil", "eosinophil": "Eosinophil", "erythroblast": "Erythroblast",
        "immature_granulocytes": "Immature granulocyte", "lymphocyte": "Lymphocyte",
        "monocyte": "Monocyte", "neutrophil": "Neutrophil", "platelet": "Platelet"}


def find(root: Path, name: str):
    hits = sorted(root.rglob(name))
    return hits[0] if hits else None


def read_results(path):
    with open(path) as f:
        return [{k.strip(): float(v) for k, v in r.items() if v and v.strip()} for r in csv.DictReader(f)]


def plot_curves(rows, out):
    ep = [r["epoch"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(ep, [r["train/loss"] for r in rows], color="#4a6fa5", label="train loss")
    if "val/loss" in rows[0]:
        ax[0].plot(ep, [r["val/loss"] for r in rows], color="#d9485f", ls="--", label="val loss")
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend(frameon=False)
    ax[1].plot(ep, [r["metrics/accuracy_top1"] for r in rows], color="#7c5cd6", label="val top-1 accuracy")
    ax[1].plot(ep, [r["metrics/accuracy_top5"] for r in rows], color="#2f9e8f", ls="--", label="val top-5 accuracy")
    ax[1].set_title("Validation accuracy"); ax[1].set_xlabel("epoch"); ax[1].set_ylim(top=1.0); ax[1].legend(frameon=False)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False); a.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def plot_confusion(m, out):
    cm = m["confusion_matrix"]
    norm = [[v / max(1, sum(row)) for v in row] for row in cm]
    labels = [NICE[c] for c in m["labels"]]
    fig, ax = plt.subplots(figsize=(8.5, 7.2))
    im = ax.imshow(norm, cmap="Purples", vmin=0, vmax=1)
    for i, row in enumerate(norm):
        for j, v in enumerate(row):
            if cm[i][j]:
                ax.text(j, i, f"{v:.2f}\n({cm[i][j]})", ha="center", va="center", fontsize=7.5,
                        color="white" if v > .55 else "#222")
    ax.set_xticks(range(len(labels)), labels, rotation=40, ha="right"); ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Test confusion matrix (row-normalized; counts in brackets)")
    fig.colorbar(im, fraction=.046, pad=.04); fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def plot_errors(pred_csv, out, n=24):
    with open(pred_csv) as f:
        wrong = [r for r in csv.DictReader(f) if r["true"] != r["pred"]]
    wrong.sort(key=lambda r: -float(r["conf"]))
    wrong = [r for r in wrong if (WBC_DATA_DIR / r["file"]).exists()][:n]
    if not wrong:
        return 0
    cols = 8; rws = (len(wrong) + cols - 1) // cols
    fig, axes = plt.subplots(rws, cols, figsize=(cols * 1.7, rws * 2.1))
    for ax in axes.flat:
        ax.axis("off")
    for ax, r in zip(axes.flat, wrong):
        ax.imshow(Image.open(WBC_DATA_DIR / r["file"]).convert("RGB"))
        ax.set_title(f"true: {NICE[r['true']]}\npred: {NICE[r['pred']]} {float(r['conf']):.0%}", fontsize=6.5)
    fig.suptitle("Most confident mistakes on the test split", fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    return len(wrong)


def pct(x):
    return f"{x * 100:.1f}%"


def main(outputs: Path = WBC_OUTPUTS, reports: Path = REPORTS):
    fig_dir = reports / "figures" / "wbc"
    fig_dir.mkdir(parents=True, exist_ok=True)
    metrics = json.loads(find(outputs, "test_metrics.json").read_text())
    cfg = json.loads(find(outputs, "run_config.json").read_text())
    rows = read_results(find(outputs, "results.csv"))
    test, val = metrics["test"], metrics["val"]

    plot_curves(rows, fig_dir / "curves.png")
    plot_confusion(test, fig_dir / "confusion_matrix.png")
    n_err = plot_errors(find(outputs, "test_predictions.csv"), fig_dir / "mistakes.png")

    best = max(rows, key=lambda r: r["metrics/accuracy_top1"])
    o = test["overall"]
    pc = test["per_class"]
    worst = min(pc, key=lambda c: pc[c]["recall"])
    cm, labels = test["confusion_matrix"], test["labels"]
    k = labels.index(worst)
    confused = max((j for j in range(len(labels)) if j != k), key=lambda j: cm[k][j])

    lines = [
        "# White Blood Cell Classifier Report",
        "",
        "Second model of the project: classifies a single-cell image into one of 8 blood cell types. "
        "It is separate from the TXL-PBC detector (tag `detector-v1`), which finds and counts cells in smear images.",
        "",
        f"- **Model:** `{cfg['model']}` fine-tuned on {cfg['dataset']}, image size {cfg['imgsz']} px",
        f"- **Training:** {len(rows)} epochs run (max {cfg['epochs']}, early-stop patience {cfg['patience']}), "
        f"batch {cfg['batch']}, GPU: {cfg['gpu']}",
        f"- **Best epoch:** {int(best['epoch'])} (val top-1 accuracy {pct(best['metrics/accuracy_top1'])})",
        f"- **Weights:** `outputs/wbc_classifier/.../wbc_best.pt`",
        "",
        "## Test results",
        "",
        f"Held-out **test split** ({o['images']:,} images), never used for training or model selection.",
        "",
        "| Accuracy | Balanced accuracy | Macro F1 | Top-2 accuracy |",
        "|---:|---:|---:|---:|",
        f"| {pct(o['accuracy'])} | {pct(o['balanced_accuracy'])} | {o['macro_f1']:.3f} | {pct(o['top2_accuracy'])} |",
        "",
        f"Validation split: accuracy {pct(val['overall']['accuracy'])}, macro F1 {val['overall']['macro_f1']:.3f}.",
        "",
        "## Per class (test)",
        "",
        "| Class | Precision | Recall | F1 | Test images |",
        "|---|---:|---:|---:|---:|",
    ]
    for c in WBC_CLASSES:
        r = pc[c]
        lines.append(f"| {NICE[c]} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | {r['support']} |")
    lines += [
        "",
        f"Weakest class by recall: **{NICE[worst]}** ({pct(pc[worst]['recall'])}); "
        f"it is most often mistaken for **{NICE[labels[confused]]}** ({cm[k][confused]} of {pc[worst]['support']} test images).",
        "",
        "## Confusion matrix",
        "",
        "![Confusion matrix](figures/wbc/confusion_matrix.png)",
        "",
        "## Training curves",
        "",
        "![Training curves](figures/wbc/curves.png)",
        "",
    ]
    if n_err:
        lines += ["## Most confident mistakes", "", "![Mistakes](figures/wbc/mistakes.png)", ""]
    lines += [
        "## Limitations",
        "",
        "- BloodMNIST images are centered single cells at 64 × 64 px from one source. Accuracy on cells "
        "cropped from other smears (for example TXL-PBC detections or a live microscope camera) is not measured "
        "here and may be lower, since those images differ in scale, stain and framing.",
        "- Classes are imbalanced in training (849 lymphocytes vs 2,330 neutrophils); see per-class recall above.",
        "- For research and education only; not a diagnostic device.",
        "",
    ]
    (reports / "wbc_classifier_report.md").write_text("\n".join(lines))
    print(f"wrote {reports / 'wbc_classifier_report.md'} and figures in {fig_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", type=Path, default=WBC_OUTPUTS)
    ap.add_argument("--reports", type=Path, default=REPORTS)
    a = ap.parse_args()
    main(a.outputs, a.reports)
