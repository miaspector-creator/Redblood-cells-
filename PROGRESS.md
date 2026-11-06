# Project progress

Status snapshot for picking the work back up later (last updated 2026-11-06).

## Done

| What | Where | Git |
|---|---|---|
| Cell detector (WBC / RBC / platelets), YOLO26s on TXL-PBC. Test: P 0.974, R 0.948, mAP50 0.990 | `kaggle/kernel/train.py`, `reports/performance_report.md` | tag `detector-v1` |
| Web app: upload, drag and drop, paste, image links, samples, live microscope camera, strictness presets, zoom, CSV/JSON/PNG export | `webapp/` | `a4c8cc1` |
| WBC type classifier pipeline (YOLO26s-cls on BloodMNIST, 8 classes) + app integration: WBC differential, type tags, All cells / White cells switch | `kaggle/kernel_wbc/`, `scripts/*_wbc.py` | `a75a5f6` |
| Guided 100-cell WBC differential (motion tracking, count each cell once, report with 95% ranges, CSV/PDF) | `webapp/` | `843a1d2` |
| Master's project disclaimer (start page, footer, live view, help, reports, README) | `webapp/`, `README.md` | `843a1d2` |
| Hugging Face Space deployment files (prepared, **not deployed**) | `deploy/space/`, `scripts/deploy_space.py` | `55991d8` |

| WBC classifier trained on Kaggle T4 (51 epochs, early stop). Test: accuracy 99.0%, macro F1 0.992; weakest: neutrophil vs immature granulocyte | `outputs/wbc_classifier/outputs/wbc_best.pt` (git-ignored), `reports/wbc_classifier_report.md` | tag `wbc-classifier-v1` |

## In progress

- **GitHub Releases:** files ready in `outputs/release/` (git-ignored). Create it at
  https://github.com/miaspector-creator/Redblood-cells-/releases/new?tag=detector-v1 and attach
  `detector-v1_yolo26s_txl-pbc.pt`, using `detector-v1_release_notes.md` as the description.
  Same for the classifier at https://github.com/miaspector-creator/Redblood-cells-/releases/new?tag=wbc-classifier-v1
  with `wbc-classifier-v1_yolo26s-cls_bloodmnist.pt` and `wbc-classifier-v1_release_notes.md`.
- **Real-microscope check:** classifier accuracy on cells cropped from smears (TXL-PBC, own microscope)
  is not measured; differently stained slides are least reliable.

## On hold / ideas

- **Hugging Face deployment:** on hold. Needs an `HF_TOKEN` (write) in `.env` and a public/private
  decision. Sample images are excluded by default because the TXL-PBC license is unknown.
- **Sign-in and saving:** preferred approach is Google sign-in saving to each user's own Google Drive
  (nothing stored on this computer). Not started.
- **Other feature ideas:** correct-the-AI labeling mode, teaching/quiz mode, image quality check,
  explainability heatmaps, RBC size analysis.

## Run the app locally

```bash
.venv/bin/python -m uvicorn webapp.app:app --port 8001 --reload --reload-dir webapp
# open http://localhost:8001
```
