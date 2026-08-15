# Python 3.12, matching the project's actual development environment
# (venv/pyvenv.cfg: Python 3.12.13) -- not just "a recent 3.x".
FROM python:3.12-slim

WORKDIR /app

# build-essential: numba/llvmlite (umap-learn's dependency) and a few
# scipy/torch-geometric extras can need a C/C++ toolchain at install time
# even when a prebuilt wheel isn't available for the exact platform.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install torch from PyTorch's own CPU-only wheel index first: the default
# PyPI wheel is CUDA-enabled and drags in ~5-6GB of nvidia-* runtime
# libraries (cublas, cudnn, nccl, etc.) this project never uses -- there's
# no GPU in this environment (see config.py's CARE_GNN_CONFIG note on
# full-graph training instead of mini-batch, motivated by the same
# no-GPU constraint). `pip install -r requirements.txt` below then finds
# torch already satisfies its `>=2.1.0` constraint and leaves it alone,
# so the rest of the dependency set still resolves normally from PyPI.
RUN pip install --no-cache-dir --default-timeout=180 --retries 10 \
    --index-url https://download.pytorch.org/whl/cpu \
    torch
RUN pip install --no-cache-dir --default-timeout=180 --retries 10 -r requirements.txt

# data/raw is intentionally NOT baked in here: it's gitignored and
# Kaggle-license-restricted (see .dockerignore, which excludes it from the
# build context entirely so it can never end up in an image layer even if
# present on the host running `docker build`). It is supplied at runtime as
# a bind-mounted volume -- see docker-compose.yml.
COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
