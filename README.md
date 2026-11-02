# Blood cell detection with YOLO26

Fine-tunes an Ultralytics **YOLO26** detector (`yolo26s.pt`) to locate and classify
cells in peripheral blood smear images as **WBC**, **RBC** or **Platelets**, using the
TXL-PBC dataset (1,260 images, 18,143 boxes). Training runs on a Kaggle GPU; progress
and results are shown in a local HTML dashboard.

## Layout

| Path | Purpose |
|---|---|
| `kaggle/kernel/train.py` | Training + test evaluation script that runs on Kaggle |
| `scripts/dataset_report.py` | Extracts and validates the dataset, writes `reports/dataset_report.md` |
| `scripts/kaggle_launch.py` | Uploads TXL-PBC as a private Kaggle dataset and starts the GPU kernel |
| `scripts/monitor.py` | Polls the run, updates the dashboard, downloads outputs when finished |
| `scripts/build_reports.py` | Writes `reports/training_report.md` and `reports/performance_report.md` |
| `dashboard/index.html` | Dashboard (reads `dashboard/data.json`, refreshes every 20 s) |
| `reports/` | Markdown reports and figures |

## Run it

```bash
python3 -m venv .venv
.venv/bin/pip install kaggle six pillow matplotlib numpy pyyaml
cp .env.example .env               # add your Kaggle API token

.venv/bin/python scripts/dataset_report.py   # needs Data/TXL-PBC.zip
.venv/bin/python scripts/kaggle_launch.py    # starts training on a Kaggle T4
.venv/bin/python scripts/monitor.py          # keep running until the run ends

.venv/bin/python -m http.server 8000         # then open http://localhost:8000/dashboard/
```

Training settings (model, epochs, image size, batch, patience) are at the top of
`kaggle/kernel/train.py`. The held-out `test` split is used only for the final evaluation
of the best checkpoint; model selection uses `val`.

## Notes

- `Data/TXL-PBC.zip` must be an intact copy. A copy that went through a text-mode
  transfer (bytes replaced with `EF BF BD`) cannot be repaired; download it again.
- `Data/Dataset_NeuNN` and `Data/bloodmnist` are image-classification datasets with no
  bounding boxes, so they are not used for detection training.
- Trained weights (`outputs/best.pt`) are not committed; they are in the Kaggle kernel output.
