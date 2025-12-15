# Use slim Python base image
# THIS IS BUILDING THE PYTHON BACKEND (in the BACKEND FOLDER)

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY ./backend /app/backend

COPY ./requirements.txt /app/requirements.txt

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*


# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variable to detect Docker
ENV RUNNING_IN_DOCKER=1

EXPOSE 8080

# Set entrypoint
CMD [ "python", "/app/backend/server.py" ] 

