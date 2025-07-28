# --------------------------
# Extreme size optimization build - targeting <1GB
# --------------------------

# Build stage with Debian slim
FROM python:3.10-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install minimal build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Minimal pip setup
RUN pip install --no-cache-dir --upgrade pip

# Install minimal PyTorch first (CPU only, no extras)
RUN pip install --no-cache-dir \
    torch==2.0.0+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --no-deps

# Install core numerical libraries
RUN pip install --no-cache-dir numpy==1.24.4

# Install dependencies with minimal extras
RUN pip install --no-cache-dir \
    pdfplumber==0.11.0 \
    --no-deps

# Install transformers with minimal dependencies
RUN pip install --no-cache-dir \
    transformers==4.35.2 \
    --no-deps

# Install sentence-transformers without full dependencies  
RUN pip install --no-cache-dir \
    sentence-transformers==2.7.0 \
    --no-deps

# Install only essential dependencies manually
RUN pip install --no-cache-dir \
    pillow \
    pdfminer.six \
    charset-normalizer \
    requests \
    urllib3 \
    certifi \
    huggingface-hub \
    tokenizers \
    regex \
    tqdm \
    packaging \
    filelock \
    typing-extensions

# Clean up pip cache
RUN pip cache purge

# --------------------------
# Ultra-minimal runtime
# --------------------------
FROM python:3.10-slim AS runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Install only essential runtime libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    poppler-utils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/* \
    && rm -rf /var/cache/apt/* \
    && rm -rf /usr/share/doc/* \
    && rm -rf /usr/share/man/* \
    && rm -rf /usr/share/locale/*

# Copy virtual environment
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Copy only essential files
COPY app.py .
COPY input/challenge1b_input.json ./input/
COPY input/docs/ ./input/docs/

# Copy model files (these are the largest - optimize if possible)
COPY model/all-MiniLM-L6-v2/ ./model/all-MiniLM-L6-v2/

# Create output directory
RUN mkdir -p output

# Create non-root user
RUN useradd --create-home --uid 1000 app \
    && chown -R app:app /app
USER app

# Optimized command
CMD ["python", "-O", "app.py"]
