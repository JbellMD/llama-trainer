# Use AMD ROCm base image
FROM rocm/pytorch:rocm5.6_ubuntu20.04_2.0.1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install --upgrade pip
COPY requirements.txt .
RUN pip3 install -r requirements.txt
RUN pip3 install mlflow

# Copy project files
COPY . .

# Expose port for API
EXPOSE 8000

# Command to run the application
CMD ["python3", "app.py"]
