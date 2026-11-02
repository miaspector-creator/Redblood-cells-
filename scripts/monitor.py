"""Poll the Kaggle kernel, stream per-epoch metrics into dashboard/data.json, and fetch
outputs + build reports when the run finishes.

    .venv/bin/python scripts/monitor.py            # poll until the run ends
    .venv/bin/python scripts/monitor.py --once     # single refresh
"""
import argparse
import re
import time

from common import KAGGLE_USER, KERNEL_SLUG, OUTPUTS, kaggle, load_env
from dashboard_data import load, update

REF = f"{KAGGLE_USER}/{KERNEL_SLUG}"
EPOCH_RE = re.compile(r"@@EPOCH ((?:[\w/().-]+=-?[\d.eE+-]+ ?)+)")
KV_RE = re.compile(r"([\w/().-]+)=(-?[\d.eE+-]+)")
STAGE_RE = re.compile(r"@@STAGE (\w+)")
ERROR_RE = re.compile(r"(Traceback \(most recent call last\)|\w*Error: [^\\\"\n]{0,200})")
TERMINAL = {"complete", "error", "cancelAcknowledged", "cancel_acknowledged"}


def parse_status(text):
    m = re.search(r'status "?(?:KernelWorkerStatus\.)?(\w+)', text)
    return m.group(1).lower() if m else "unknown"


def parse_log(log):
    epochs = {}
    for m in EPOCH_RE.finditer(log):
        row = {k: float(v) for k, v in KV_RE.findall(m.group(1))}
        if "epoch" in row:
            epochs[int(row["epoch"])] = row
    stages = STAGE_RE.findall(log)
    errors = [e for e in ERROR_RE.findall(log) if not e.startswith("Traceback")]
    return [epochs[k] for k in sorted(epochs)], (stages[-1] if stages else None), errors[-3:]


def refresh():
    status = parse_status(kaggle("kernels", "status", REF).stdout)
    log = kaggle("kernels", "logs", REF, check=False).stdout
    epochs, stage, errors = parse_log(log)
    prev = load().get("training", {})
    if not epochs and prev.get("kernel") == REF and status not in TERMINAL:
        epochs = prev.get("epochs", [])  # log endpoint can lag; keep what we had
    last = epochs[-1] if epochs else {}
    best = max(epochs, key=lambda r: r.get("metrics/mAP50-95(B)", 0)) if epochs else {}
    update("training", {
        "kernel": REF, "url": f"https://www.kaggle.com/code/{REF}",
        "status": status, "stage": stage, "errors": errors,
        "epoch": int(last.get("epoch", 0)), "epochs_total": int(last.get("epochs", 0)),
        "best_epoch": int(best.get("epoch", 0)), "epochs": epochs,
        "polled_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    })
    print(f"[{time.strftime('%H:%M:%S')}] {status} stage={stage} epoch={last.get('epoch', 0):.0f}"
          f"/{last.get('epochs', 0):.0f} mAP50-95={last.get('metrics/mAP50-95(B)', 0):.3f}", flush=True)
    return status


def fetch_outputs():
    OUTPUTS.mkdir(exist_ok=True)
    kaggle("kernels", "output", REF, "-p", str(OUTPUTS), "-o", "-q")
    import build_reports
    build_reports.main()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--interval", type=int, default=60)
    args = ap.parse_args()
    load_env()
    while True:
        status = refresh()
        if status in TERMINAL or args.once:
            break
        time.sleep(args.interval)
    if status == "complete":
        fetch_outputs()
        print("outputs downloaded; reports and dashboard updated")
