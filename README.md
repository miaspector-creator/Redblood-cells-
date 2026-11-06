**Live dashboard:** https://miaspector-creator.github.io/Redblood-cells-/dashboard/
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

## Web app (use the trained model)

`webapp/` is a local web app. Add smear images by uploading from your computer, drag and
drop (including straight from another browser tab), pasting, or pasting an image link
(the server downloads it; links to private/local network addresses are refused). Sample
smears from the TXL-PBC test split can be downloaded from the page. For each image you get WBC / RBC / Platelet boxes,
counts, WBC-per-1,000-RBC, a live confidence slider, and CSV / JSON / annotated-PNG export.
It needs the trained weights at `outputs/runs/txl_pbc/weights/best.pt` (or set
`BLOODCELL_MODEL=/path/to/best.pt`) and runs on Apple GPU (MPS), CUDA or CPU.

```bash
.venv/bin/pip install -r webapp/requirements.txt
.venv/bin/python -m uvicorn webapp.app:app --port 8001   # then open http://localhost:8001
```

**Live camera:** press *Live camera* to count straight from a microscope camera (USB camera,
HDMI capture adapter, or a phone on an eyepiece adapter that shows up as a webcam). Frames are
analyzed about 9–10 times a second; *Pause* freezes a frame and *Capture* saves it with its counts.
Browsers only allow camera access on `localhost` or `https://`, so open the app on the computer the
camera is plugged into.

Research and education use only; it is not a diagnostic device.

Training settings (model, epochs, image size, batch, patience) are at the top of
`kaggle/kernel/train.py`. The held-out `test` split is used only for the final evaluation
of the best checkpoint; model selection uses `val`.

## Notes

- `Data/TXL-PBC.zip` must be an intact copy. A copy that went through a text-mode
  transfer (bytes replaced with `EF BF BD`) cannot be repaired; download it again.
- `Data/Dataset_NeuNN` and `Data/bloodmnist` are image-classification datasets with no
  bounding boxes, so they are not used for detection training.
- Trained weights (`outputs/best.pt`) are not committed; they are in the Kaggle kernel output.
