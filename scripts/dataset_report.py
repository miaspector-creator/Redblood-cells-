"""Extract TXL-PBC locally, validate it, and write reports/dataset_report.md + figures.

Also writes the "dataset" section of dashboard/data.json.
"""
import json
import random
import zipfile
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from common import DASHBOARD, DATA_DIR, DATA_ZIP, NAMES, REPORTS
from dashboard_data import update

COLORS = {"WBC": "#2a78d6", "RBC": "#eb6834", "Platelets": "#1baf7a"}  # match dashboard
SPLITS = ["train", "val", "test"]
FIG = REPORTS / "figures"


def extract():
    if not (DATA_DIR / "data.yaml").exists():
        with zipfile.ZipFile(DATA_ZIP) as zf:
            bad = zf.testzip()
            if bad:
                raise SystemExit(f"{DATA_ZIP} is corrupt (first bad entry: {bad})")
            zf.extractall(DATA_DIR.parent)


def scan():
    stats = {}
    problems = []
    for split in SPLITS:
        imgs = sorted((DATA_DIR / "images" / split).glob("*.png"))
        s = {"images": len(imgs), "boxes": Counter(), "per_image": [], "sizes": Counter(),
             "modes": Counter(), "box_wh": {n: [] for n in NAMES}}
        for img in imgs:
            lbl = DATA_DIR / "labels" / split / (img.stem + ".txt")
            with Image.open(img) as im:
                w, h = im.size
                s["sizes"][f"{w}x{h}"] += 1
                s["modes"][im.mode] += 1
            if not lbl.exists():
                problems.append(f"{split}/{img.name}: missing label")
                continue
            n = 0
            for ln, line in enumerate(lbl.read_text().split("\n"), 1):
                if not line.strip():
                    continue
                parts = line.split()
                try:
                    c, x, y, bw, bh = int(parts[0]), *map(float, parts[1:5])
                except (ValueError, IndexError):
                    problems.append(f"{split}/{lbl.name}:{ln}: malformed row")
                    continue
                if not (0 <= c < len(NAMES)) or not all(0 <= v <= 1 for v in (x, y, bw, bh)):
                    problems.append(f"{split}/{lbl.name}:{ln}: out of range")
                    continue
                s["boxes"][NAMES[c]] += 1
                s["box_wh"][NAMES[c]].append((bw * w, bh * h))
                n += 1
            s["per_image"].append(n)
        stats[split] = s
    return stats, problems


def draw_samples(k=6, seed=0):
    random.seed(seed)
    imgs = random.sample(sorted((DATA_DIR / "images" / "train").glob("*.png")), k)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
    for ax, p in zip(axes.flat, imgs):
        im = Image.open(p).convert("RGB")
        d = ImageDraw.Draw(im)
        W, H = im.size
        for line in (DATA_DIR / "labels" / "train" / (p.stem + ".txt")).read_text().split("\n"):
            if line.strip():
                c, x, y, w, h = line.split()[:5]
                x, y, w, h = float(x) * W, float(y) * H, float(w) * W, float(h) * H
                d.rectangle([x - w / 2, y - h / 2, x + w / 2, y + h / 2],
                            outline=COLORS[NAMES[int(c)]], width=max(2, W // 250))
        ax.imshow(im)
        ax.set_title(p.name, fontsize=9)
        ax.axis("off")
    handles = [plt.Line2D([], [], color=c, lw=3, label=n) for n, c in COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG / "dataset_samples.jpg", dpi=110)
    plt.close(fig)


def plot_stats(stats):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = range(len(SPLITS))
    bottom = [0] * len(SPLITS)
    for n in NAMES:
        vals = [stats[s]["boxes"][n] for s in SPLITS]
        ax.bar(x, vals, bottom=bottom, color=COLORS[n], label=n, width=0.6)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_xticks(list(x), SPLITS)
    ax.set_ylabel("annotated cells")
    ax.set_title("Cells per split and class")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "class_distribution.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    allp = sum((stats[s]["per_image"] for s in SPLITS), [])
    axes[0].hist(allp, bins=30, color="#4a6fa5")
    axes[0].set_xlabel("cells per image")
    axes[0].set_ylabel("images")
    axes[0].set_title("Annotation density")
    for n in NAMES:
        wh = sum((stats[s]["box_wh"][n] for s in SPLITS), [])
        axes[1].scatter([a for a, _ in wh], [b for _, b in wh], s=3, alpha=0.35, color=COLORS[n], label=n)
    axes[1].set_xlabel("box width (px)")
    axes[1].set_ylabel("box height (px)")
    axes[1].set_title("Box sizes by class")
    axes[1].legend(frameon=False, markerscale=4)
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "box_statistics.png", dpi=120)
    plt.close(fig)


def mean_wh(stats, n):
    wh = sum((stats[s]["box_wh"][n] for s in SPLITS), [])
    return (sum(a for a, _ in wh) / len(wh), sum(b for _, b in wh) / len(wh)) if wh else (0, 0)


def write_report(stats, problems):
    tot_img = sum(stats[s]["images"] for s in SPLITS)
    tot = {n: sum(stats[s]["boxes"][n] for s in SPLITS) for n in NAMES}
    tot_box = sum(tot.values())
    sizes = sum((stats[s]["sizes"] for s in SPLITS), Counter())
    modes = sum((stats[s]["modes"] for s in SPLITS), Counter())
    allp = sum((stats[s]["per_image"] for s in SPLITS), [])

    L = ["# TXL-PBC Dataset Report", "",
         "Peripheral blood smear images annotated for object detection in YOLO format. "
         "Each cell has a bounding box and one of three classes: WBC, RBC, Platelets.", "",
         "## Splits", "",
         "| Split | Images | Cells | WBC | RBC | Platelets | Cells / image |",
         "|---|---:|---:|---:|---:|---:|---:|"]
    for s in SPLITS:
        b = stats[s]["boxes"]
        n = sum(b.values())
        L.append(f"| {s} | {stats[s]['images']:,} | {n:,} | {b['WBC']:,} | {b['RBC']:,} | "
                 f"{b['Platelets']:,} | {n / stats[s]['images']:.1f} |")
    L.append(f"| **total** | **{tot_img:,}** | **{tot_box:,}** | {tot['WBC']:,} | {tot['RBC']:,} | "
             f"{tot['Platelets']:,} | {tot_box / tot_img:.1f} |")
    L += ["", "## Class balance", "",
          "| Class | Cells | Share | Mean box (px, w × h) |", "|---|---:|---:|---:|"]
    for n in NAMES:
        w, h = mean_wh(stats, n)
        L.append(f"| {n} | {tot[n]:,} | {tot[n] / tot_box:.1%} | {w:.0f} × {h:.0f} |")
    L += ["", f"RBCs dominate the annotations. Each image holds between {min(allp)} and {max(allp)} "
          f"cells (median {sorted(allp)[len(allp) // 2]}).", "",
          "![Class distribution](figures/class_distribution.png)", "",
          "![Box statistics](figures/box_statistics.png)", "",
          "## Images", "",
          f"- {len(sizes)} distinct resolutions; most common: "
          + ", ".join(f"{k} ({v})" for k, v in sizes.most_common(4)),
          "- Color modes: " + ", ".join(f"{k} ({v})" for k, v in modes.most_common())
          + ". RGBA images are converted to 3-channel on load.", "",
          "![Sample training images with ground-truth boxes](figures/dataset_samples.jpg)", "",
          "## Validation", "",
          "Archive CRC check passed; every image has a matching label file; every label row "
          "was checked for 5 fields, class id in 0–2 and coordinates in [0, 1]."
          if not problems else f"**{len(problems)} problems found:**"]
    L += [f"- {p}" for p in problems[:50]]
    L += ["", "> Note: the original `Data/TXL-PBC.zip` in this folder was corrupted (binary bytes "
          "replaced by U+FFFD during a text-mode transfer). It was replaced with a fresh download "
          "from the same source (SHA-256 `ac1a0281…5403f`)."]
    (REPORTS / "dataset_report.md").write_text("\n".join(L) + "\n")

    update("dataset", {
        "name": "TXL-PBC", "classes": NAMES, "images": tot_img, "cells": tot_box,
        "splits": {s: {"images": stats[s]["images"], **{n: stats[s]["boxes"][n] for n in NAMES}}
                   for s in SPLITS},
        "mean_box_px": {n: [round(v) for v in mean_wh(stats, n)] for n in NAMES},
        "cells_per_image": {"min": min(allp), "max": max(allp),
                            "median": sorted(allp)[len(allp) // 2]},
        "resolutions": len(sizes), "problems": len(problems),
        "sample_image": "../reports/figures/dataset_samples.jpg",
    })


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    DASHBOARD.mkdir(exist_ok=True)
    extract()
    stats, problems = scan()
    plot_stats(stats)
    draw_samples()
    write_report(stats, problems)
    print(f"wrote {REPORTS / 'dataset_report.md'} ({len(problems)} problems)")
