# Hugging Face Space (Docker SDK) for the Blood Cell Counter web app. CPU-only.
# Built by Hugging Face from the folder that scripts/deploy_space.py uploads.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Spaces run as uid 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    YOLO_AUTOINSTALL=false \
    YOLO_OFFLINE=1 \
    BLOODCELL_DEVICE=cpu \
    BLOODCELL_MODEL=/home/user/app/models/detector/best.pt \
    BLOODCELL_WBC_MODEL=/home/user/app/models/wbc/wbc_best.pt \
    BLOODCELL_SAMPLES=/home/user/app/samples
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY --chown=user . .
EXPOSE 7860
CMD ["uvicorn", "webapp.app:app", "--host", "0.0.0.0", "--port", "7860"]
