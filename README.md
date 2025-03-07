# LLaMA-Trainer: TinyLLaMA-1.1B for ML Workflows

This project provides a setup for fine-tuning and deploying TinyLLaMA-1.1B as an AI agent to handle machine learning workflows and automation tasks. It's optimized for AMD hardware (Ryzen 7 8845HS, Radeon 780M) and includes Docker support with ROCm.

## Features

- **Optimized for AMD Hardware**: Uses ROCm for GPU acceleration
- **Memory Efficient**: Implements 4-bit quantization and gradient checkpointing
- **Tool Integration**: Includes MLflow, Git, and file system tools
- **Fine-Tuning**: Parameter-efficient fine-tuning with LoRA
- **API with Rate Limiting**: FastAPI server with monitoring and rate limiting
- **ML Workflow Templates**: Pre-defined templates for common ML tasks
- **Performance Metrics**: Tracks token generation speed and memory usage

## Project Structure

```
llama-trainer/
├── Dockerfile          # Docker setup with ROCm support
├── app.py              # FastAPI application with LangChain agent
├── finetune.py         # Script for fine-tuning TinyLLaMA
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Setup Instructions

### Option 1: Docker Setup (Recommended for Deployment)

1. **Build the Docker image**:
   ```bash
   docker build -t llama-trainer .
   ```

2. **Run the Docker container**:
   ```bash
   docker run -p 8000:8000 -p 5000:5000 --device=/dev/kfd --device=/dev/dri --group-add video llama-trainer
   ```

### Option 2: Local Setup (Alternative if Docker/ROCm issues arise)

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Fine-tune the model**:
   ```bash
   python finetune.py
   ```

3. **Run the API server**:
   ```bash
   python app.py
   ```

## Creating a Dataset for Fine-Tuning

Create a `dataset.jsonl` file with examples in the following format:

```json
{"instruction": "Your ML-related instruction here", "response": "Expected model response"}
```

Examples:
- ML preprocessing tasks
- Model training code
- Data analysis scripts
- ML workflow automation

## Using the API

Once the server is running, you can interact with the agent through the API:

```python
import requests

response = requests.post(
    "http://localhost:8000/run-agent",
    json={
        "prompt": "Write a Python script to preprocess text data for NLP",
        "refine": True,
        "template": "preprocess"
    }
)

print(response.json())
```

## AMD GPU Considerations

For AMD Ryzen 7 8845HS with Radeon 780M:

1. **Install ROCm**: Follow AMD's [installation guide](https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html)
2. **Memory Management**: The 16GB RAM constraint is addressed through:
   - 4-bit quantization
   - Gradient checkpointing
   - Dynamic batch sizing
3. **Temperature Management**: Monitor system temperature during training

## Troubleshooting

- **ROCm Detection Issues**: If ROCm doesn't detect your GPU, try updating to the latest drivers
- **Out of Memory Errors**: Reduce batch size or use CPU-only mode with `device_map="cpu"`
- **Docker Permission Issues**: Ensure you have proper permissions for Docker and GPU access

## Advanced Usage

### MLflow Tracking

MLflow server is accessible at http://localhost:5000

### Custom Templates

Add your own templates to the `ML_TEMPLATES` dictionary in `app.py`.


## How to use the project

## For direct training without evolution:

run_train.bat --dataset dataset_fixed.jsonl --batch_size 4 --epochs 3

## For running the genetic algorithm:

run_evolution.bat --population_size 10 --generations 5
## For inference with a trained model:

python run_inference.py --adapter_path ./results/epoch_3 --interactive