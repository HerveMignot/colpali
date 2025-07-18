# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the project files into the container
COPY . .

# Install git for package installation (required for hatch)
RUN apt-get update && apt-get install -y git

# Install dependencies
RUN pip install --no-cache-dir ".[all]" fastapi uvicorn "fast-plaid @ git+https://github.com/Lightning-AI/fast-plaid.git"

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
