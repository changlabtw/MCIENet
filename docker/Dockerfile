# Development stage optimization: Using single-stage build to speed up development
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# Set working directory
WORKDIR /app

# Copy requirements.txt first to leverage Docker cache
COPY ../requirements.txt .

# Install system dependencies and Python packages
RUN apt-get update && apt-get install -y \
    git \
    wget \
    bedtools \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -r requirements.txt

# Monitoring tools (optional)
RUN apt-get update && apt-get install -y \
    htop \
    tmux \
    && pip install --no-cache-dir gpustat

# Copy project files
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

# Default command: start interactive shell
CMD ["/bin/bash"]

# Note: This is a development configuration. For production, use multi-stage build to reduce final image size
