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
RUN mkdir -p /app/data /app/models /app/notebooks

# Copy project files
COPY project/main /app/main
COPY project/idea_notebooks /app/notebooks

# Set up Jupyter
RUN pip install --user jupyter

# Copy entrypoint script
COPY docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Expose Jupyter port
EXPOSE 8888

ENTRYPOINT ["/app/entrypoint.sh"]
