# Use Python 3.9 slim image as base
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install CUDA dependencies (for GPU support)
RUN apt-get update && apt-get install -y \
    wget \
    gnupg2 \
    && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb \
    && dpkg -i cuda-keyring_1.0-1_all.deb \
    && apt-get update \
    && apt-get install -y cuda-toolkit-12-0 \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for the RAG system
RUN pip install --no-cache-dir \
    motor \
    pymongo \
    hnswlib \
    transformers \
    torch \
    numpy \
    pandas \
    scikit-learn \
    sentence-transformers

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/models /app/data /app/logs /app/storage

# Set permissions
RUN chmod -R 755 /app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
