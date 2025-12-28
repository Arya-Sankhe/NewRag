FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/project:/app/backend

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy ALL requirements first for caching
COPY project/requirements.txt /app/project/requirements.txt
COPY backend/requirements.txt /app/backend/requirements.txt

# Install project requirements (base dependencies)
RUN pip install --no-cache-dir -r /app/project/requirements.txt

# Install backend requirements (FastAPI specific)
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy project code
COPY project /app/project
COPY backend /app/backend

# Expose port
EXPOSE 8000

# Working directory for running the app
WORKDIR /app/backend

# Run FastAPI
CMD ["python", "main.py"]
