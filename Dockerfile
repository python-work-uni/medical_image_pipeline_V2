# Use a lightweight Python base image
# This matches the "python:3.9-slim" recommendation [cite: 34]
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required for OpenCV
# cv2 requires 'libgl1' which isn't in python-slim by default
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy your dependency file first (for caching layers)
COPY requirements.txt .

# Install Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# Copy your source code and test folders
COPY src/ ./src/
COPY tests/ ./tests/

# Default command: Run the preprocessing script
CMD ["python", "src/preprocess.py"]