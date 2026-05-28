FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY model/ ./model/
COPY apiCall/api.py ./apiCall/api.py

# Create temp dir for uploads
RUN mkdir -p /app/tmp

ENV PYTHONUNBUFFERED=1
ENV TMPDIR=/app/tmp

EXPOSE 8000

CMD ["uvicorn", "apiCall.api:app", "--host", "0.0.0.0", "--port", "8000"]
