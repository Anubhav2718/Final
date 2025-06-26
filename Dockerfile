# Use Python slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required by faiss, uvicorn[standard], etc.
RUN apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    libomp-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose port
EXPOSE 8000

# Set environment variables (you can override with --env or --env-file at runtime)
ENV PORT=8000

# Run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
