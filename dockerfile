FROM python:3.11-slim

# Set work directory
WORKDIR /app

# Install system dependencies (for pandas/numpy/scikit-learn)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all app files, but exclude dev files
COPY . .

# Remove dev-only files to prevent accidental deployment
RUN rm -f main_dev.py testing.py

# Expose FastAPI port
EXPOSE 8000

# Start the app with production entrypoint
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
