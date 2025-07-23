# Use a slim Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all files
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install tensorflow==2.13.0 keras==2.13.1
RUN pip install -r requirements.txt

# Expose the port Render will use
EXPOSE 10000

# Run the app using gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "app:app"]
