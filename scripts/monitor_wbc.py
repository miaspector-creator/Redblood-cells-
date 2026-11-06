"""Poll the WBC classifier Kaggle kernel, record per-epoch metrics, and download outputs + build the
report when it finishes. Separate from scripts/monitor.py (the detector) and never touches its files.

    .venv/bin/python scripts/monitor_wbc.py            # poll until the run ends
    .venv/bin/python scripts/monitor_wbc.py --once     # single refresh
"""
import argparse
import json
import time

from common import KAGGLE_USER, WBC_KERNEL_SLUG, WBC_OUTPUTS, kaggle, load_env
from monitor import TERMINAL, parse_log, parse_status

REF = f"{KAGGLE_USER}/{WBC_KERNEL_SLUG}"
PROGRESS = WBC_OUTPUTS / "progress.json"
ACC = "metrics/accuracy_top1"


def refresh():
    status = parse_status(kaggle("kernels", "status", REF).stdout)
    log = kaggle("kernels", "logs", REF, check=False).stdout
    epochs, stage, errors = parse_log(log)
    # Ultralytics prints one extra validation of the best weights after the last epoch; drop it.
    epochs = [r for r in epochs if r.get("epoch", 0) <= r.get("epochs", 1e9)]
    prev = json.loads(PROGRESS.read_text()) if PROGRESS.exists() else {}
    if not epochs and status not in TERMINAL:
        epochs = prev.get("epochs", [])   # the log endpoint can lag; keep what we had
    last = epochs[-1] if epochs else {}
    best = max(epochs, key=lambda r: r.get(ACC, 0)) if epochs else {}
    WBC_OUTPUTS.mkdir(parents=True, exist_ok=True)
    PROGRESS.write_text(json.dumps({
        "kernel": REF, "url": f"https://www.kaggle.com/code/{REF}", "status": status, "stage": stage,
        "errors": errors, "epoch": int(last.get("epoch", 0)), "epochs_total": int(last.get("epochs", 0)),
        "best_epoch": int(best.get("epoch", 0)), "best_val_accuracy": best.get(ACC), "epochs": epochs,
        "polled_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }, indent=1))
    print(f"[{time.strftime('%H:%M:%S')}] {status} stage={stage} epoch={last.get('epoch', 0):.0f}"
          f"/{last.get('epochs', 0):.0f} val_acc={last.get(ACC, 0):.4f} best={best.get(ACC, 0):.4f}"
          + (f" errors={errors}" if errors else ""), flush=True)
    return status


def fetch_outputs():
    kaggle("kernels", "output", REF, "-p", str(WBC_OUTPUTS), "-o", "-q")
    import build_wbc_report
    build_wbc_report.main()


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
        print(f"outputs downloaded to {WBC_OUTPUTS}; report written to reports/wbc_classifier_report.md")
