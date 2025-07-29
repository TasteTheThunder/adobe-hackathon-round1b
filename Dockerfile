# Ultra-minimal Python image for PDF processing under 1GB
FROM python:3.10-alpine AS builder

# Install minimal build dependencies for pdfplumber
RUN apk add --no-cache gcc musl-dev

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Final stage 
FROM python:3.10-alpine

# Copy virtual environment
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Copy application file
COPY app.py .

# Create non-root user and set permissions
RUN addgroup -g 1001 -S appgroup && \
    adduser -S -D -H -u 1001 -h /app -s /sbin/nologin -G appgroup appuser && \
    chown -R appuser:appgroup /app

USER appuser

CMD ["python", "app.py"]
