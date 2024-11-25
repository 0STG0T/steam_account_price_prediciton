# Build stage
FROM python:3.12-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY docker/requirements.txt .
RUN pip install --user -r requirements.txt

# Final stage
FROM python:3.12-slim

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create directory structure
WORKDIR /app
RUN mkdir -p /app/data /app/models/onnx /app/upload

# Copy project files
COPY docker/*.py /app/docker/
COPY docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh /app/docker/cli.py

# Set environment variables
ENV PYTHONPATH=/app

ENTRYPOINT ["/app/entrypoint.sh"]
