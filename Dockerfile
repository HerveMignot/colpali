# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the project files into the container
COPY /api .

# Install git for package installation (required for hatch)
RUN apt-get update && apt-get install -y git git-lfs

# Install Hugging Face Hub library
RUN pip install huggingface_hub

# Install dependencies
RUN pip install --no-cache-dir fastapi uvicorn colpali-engine

# Download models snapshots
ENV MODEL_CACHE_PATH=/app/huggingface 
RUN python /app/download_models.py

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
