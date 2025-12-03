# Use slim Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*


# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variable to detect Docker
ENV RUNNING_IN_DOCKER=1

# Set entrypoint
ENTRYPOINT [ "python", "hand_tracking.py" ] 

