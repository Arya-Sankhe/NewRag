FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project requirements first for caching
COPY project/requirements.txt /app/project/
RUN pip install --no-cache-dir -r /app/project/requirements.txt

# Copy backend requirements
COPY backend/requirements.txt /app/backend/
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy project code
COPY project /app/project
COPY backend /app/backend

# Set Python path
ENV PYTHONPATH=/app/project:/app/backend

# Expose port
EXPOSE 8000

# Working directory for running the app
WORKDIR /app/backend

# Run FastAPI - use python directly to run main.py
CMD ["python", "main.py"]


