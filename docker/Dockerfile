# ===== Stage 1: Base Image with CUDA Support =====
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=America/New_York

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    build-essential \
    curl \
    git \
    libopenblas-dev \
    liblapack-dev \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# ===== Stage 2: Dependencies =====
FROM base AS dependencies

WORKDIR /app

# Copy dependency files first for better layer caching
COPY requirements.txt pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download Qlib data skeleton (optional, for build-time caching)
RUN python -c "import qlib; print(f'Qlib version: {qlib.__version__}')" || true

# ===== Stage 3: Development Image =====
FROM dependencies AS development

# Install dev dependencies
RUN pip install --no-cache-dir \
    pytest>=8.0.0 \
    pytest-cov>=4.0.0 \
    pytest-asyncio>=0.23.0 \
    black>=24.0.0 \
    isort>=5.13.0 \
    ruff>=0.2.0 \
    mypy>=1.8.0 \
    ipython>=8.0.0 \
    jupyterlab>=4.0.0

# Create non-root user for security
RUN groupadd -r quant && useradd -r -g quant -m -s /bin/bash quant \
    && mkdir -p /app/data /app/models /app/logs \
    && chown -R quant:quant /app

# Copy project source code
COPY --chown=quant:quant . .

USER quant

# Expose Jupyter and FastAPI ports
EXPOSE 8888 8000

# Default: start Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# ===== Stage 4: Production Image (minimal) =====
FROM dependencies AS production

RUN groupadd -r quant && useradd -r -g quant -m -s /bin/bash quant \
    && mkdir -p /app/data /app/models /app/logs \
    && chown -R quant:quant /app

COPY --chown=quant:quant . .

USER quant

# Production: run FastAPI server
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
