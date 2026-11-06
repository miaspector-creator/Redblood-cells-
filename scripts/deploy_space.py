"""Publish the web app + both trained models to a Hugging Face Space (Docker, CPU).

    .venv/bin/python scripts/deploy_space.py --dry-run            # stage files locally and list them
    .venv/bin/python scripts/deploy_space.py                      # create/update <you>/blood-cell-counter
    .venv/bin/python scripts/deploy_space.py --private            # only you (and invited people) can open it
    .venv/bin/python scripts/deploy_space.py --samples            # also publish 12 TXL-PBC sample images

Needs a Hugging Face access token with write permission in .env as HF_TOKEN=hf_...
(or in the HF_TOKEN environment variable). Nothing is uploaded with --dry-run.
"""
import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

from common import ROOT, WBC_OUTPUTS

DETECTOR_CANDIDATES = [ROOT / "outputs" / "runs" / "txl_pbc" / "weights" / "best.pt", ROOT / "outputs" / "outputs" / "best.pt"]
SAMPLES = ROOT / "Data" / "TXL-PBC" / "images" / "test"
SPACE_FILES = ROOT / "deploy" / "space"


def token():
    tok = os.environ.get("HF_TOKEN")
    env = ROOT / ".env"
    if not tok and env.exists():
        for line in env.read_text().splitlines():
            if line.strip().startswith("HF_TOKEN="):
                tok = line.split("=", 1)[1].strip()
    if not tok:
        sys.exit("HF_TOKEN missing: create a write token at https://huggingface.co/settings/tokens and add HF_TOKEN=... to .env")
    return tok


def find_weights():
    det = next((p for p in DETECTOR_CANDIDATES if p.is_file()), None)
    wbc = sorted(WBC_OUTPUTS.rglob("wbc_best.pt"), key=lambda p: p.stat().st_mtime) if WBC_OUTPUTS.is_dir() else []
    if not det:
        sys.exit("detector weights (best.pt) not found")
    if not wbc:
        sys.exit("WBC classifier weights (wbc_best.pt) not found in outputs/wbc_classifier/: finish training first")
    return det, wbc[-1]


def stage(dst: Path, with_samples: bool):
    det, wbc = find_weights()
    shutil.copytree(ROOT / "webapp", dst / "webapp", ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "requirements.txt"))
    for f in ("Dockerfile", "requirements.txt", "README.md"):
        shutil.copy(SPACE_FILES / f, dst / f)
    (dst / "models" / "detector").mkdir(parents=True)
    (dst / "models" / "wbc").mkdir(parents=True)
    shutil.copy(det, dst / "models" / "detector" / "best.pt")
    shutil.copy(wbc, dst / "models" / "wbc" / "wbc_best.pt")
    (dst / "samples").mkdir()
    if with_samples:
        for p in sorted(x for x in SAMPLES.iterdir() if x.suffix.lower() in {".png", ".jpg", ".jpeg"})[:12]:
            shutil.copy(p, dst / "samples" / p.name)
    else:
        (dst / "samples" / ".keep").write_text("")
    (dst / ".gitattributes").write_text("*.pt filter=lfs diff=lfs merge=lfs -text\n")
    return det, wbc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", help="Space id, e.g. your-name/blood-cell-counter (default: <your username>/blood-cell-counter)")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--samples", action="store_true", help="include 12 TXL-PBC test images as on-page samples")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="space_"))
    det, wbc = stage(tmp, a.samples)
    files = sorted(p.relative_to(tmp) for p in tmp.rglob("*") if p.is_file())
    size = sum((tmp / f).stat().st_size for f in files) / 1e6
    print(f"staged {len(files)} files ({size:.1f} MB) in {tmp}\n  detector: {det}\n  wbc:      {wbc}")
    for f in files:
        print("   ", f)
    if a.dry_run:
        print("dry run: nothing uploaded")
        return

    from huggingface_hub import HfApi
    api = HfApi(token=token())
    space = a.space or f"{api.whoami()['name']}/blood-cell-counter"
    api.create_repo(space, repo_type="space", space_sdk="docker", private=a.private, exist_ok=True)
    api.upload_folder(folder_path=str(tmp), repo_id=space, repo_type="space",
                      commit_message="Deploy Blood Cell Counter", delete_patterns=["*"])
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\nuploaded. Build logs and app: https://huggingface.co/spaces/{space}")
    print(f"Direct app URL (after the build finishes): https://{space.replace('/', '-').lower()}.hf.space")


if __name__ == "__main__":
    main()
