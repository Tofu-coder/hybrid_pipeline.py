#This is the first Dockerfile that was running in the terminal alongside VS. This was a test version in which it was able to run well for agent_llama.py

# Use a stable Python base image with good compatibility
FROM python:3.10-slim

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies needed by TensorFlow and your scripts
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install required Python packages
RUN pip install --upgrade pip && pip install --no-cache-dir \
    tensorflow \
    ollama \
    pandas \
    numpy

# Set working directory inside container
WORKDIR /app

# Copy your local files into the container
COPY . .

# Make results and data directories in case they aren't in repo
RUN mkdir -p data/raw results prompts

# Command to run when the container starts
CMD ["python", "agent_llama.py"]


