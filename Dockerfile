# ============================================================
# LIVO — 124M Multimodal Transformer
# Docker image for training and inference
# ============================================================

# --- Stage 1: Base with CUDA support ---
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS base

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3

# Set working directory
WORKDIR /app

# --- Stage 2: Install dependencies ---
FROM base AS deps

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- Stage 3: Copy project ---
FROM deps AS final

# Copy entire LIVO project
COPY . .

# Create directories for checkpoints and configs
RUN mkdir -p /app/checkpoints /app/data

# Default: show help
CMD ["python", "-c", "print('\\n🧠 LIVO — 124M Multimodal Transformer\\n'); \
print('Available commands:'); \
print('  Train tokenizer:  python scripts/train_tokenizer.py --dataset wikitext --dataset-config wikitext-103-raw-v1 --vocab-size 50000'); \
print('  Train model:      python scripts/train.py --dataset wikitext --dataset-config wikitext-103-raw-v1 --tokenizer configs/tokenizer.json'); \
print('  Generate text:    python scripts/generate.py --checkpoint checkpoints/latest.pt --tokenizer configs/tokenizer.json'); \
print('  Verify system:    python verify_125m.py'); \
print()"]
