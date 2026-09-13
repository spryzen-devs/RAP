FROM python:3.11-slim

# System deps needed by opencv-python-headless / Pillow at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1 \
    MODEL_WEIGHTS_PATH=/app/runs/ppe/rtdetr_ppe_v1/weights/best.pt \
    DETECT_CONF_THRESHOLD=0.35

EXPOSE 8000

# Trained weights are mounted at runtime (see docker-compose.yml) rather than
# baked into the image — keeps the image reproducible-buildable without
# requiring a GPU or the dataset at build time.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
