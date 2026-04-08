FROM python:3.11-slim

WORKDIR /app

# Install minimal system deps needed for pandas C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY . .

# Make sure the project root is on PYTHONPATH so imports work
ENV PYTHONPATH=/app

# HF Spaces uses port 7860
EXPOSE 7860

# Start the FastAPI server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
