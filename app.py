import os
import logging
import uvicorn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.agents import initialize_agent, Tool
from langchain.tools import ShellTool
from langchain.tools.file_management import ReadFileTool, WriteFileTool
from mlflow.tracking import MlflowClient
import mlflow
from fastapi import FastAPI, HTTPException
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from fastapi.middleware.cors import CORSMiddleware
import psutil
import time
import torch
from pydantic import BaseModel
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="ML Agent API")

# Rate limiting (10 requests per minute per IP)
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MLflow setup
mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()

# Load TinyLLaMA with quantization and gradient checkpointing
model_name = "TinyLLaMA/TinyLLaMA-1.1B-Chat-v1.0"
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True,
    gradient_checkpointing=True  # Memory optimization
)

# Create a HuggingFace pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.2,
    device_map="auto"
)
llm = HuggingFacePipeline(pipeline=pipe)

# Define tools
tools = [
    ShellTool(),
    ReadFileTool(),
    WriteFileTool(),
    Tool(
        name="MLflowLog",
        func=lambda x: mlflow.log_metric("agent_response_time", float(x)),
        description="Log a metric to MLflow"
    ),
    Tool(
        name="GitCommit",
        func=lambda x: os.system(f"git commit -m '{x}'"),
        description="Commit changes to git with a message"
    )
]

# Initialize LangChain agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True
)

# Predefined ML workflow templates
ML_TEMPLATES = {
    "preprocess": "import pandas as pd\nfrom sklearn.preprocessing import StandardScaler\n# Load dataset\ndf = pd.read_csv(\"dataset.csv\")\ndf = df.dropna()\n# Scale features\nscaler = StandardScaler()\nscaled_features = scaler.fit_transform(df[['feature1', 'feature2']])\nscaled_df = pd.DataFrame(scaled_features, columns=['feature1', 'feature2'])\n# Save preprocessed data\nscaled_df.to_csv(\"preprocessed_dataset.csv\", index=False)",
    "train_model": "import tensorflow as tf\n# Define a simple model\nmodel = tf.keras.Sequential([\n    tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),\n    tf.keras.layers.Dense(1, activation='sigmoid')\n])\nmodel.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])\n# Train model\nmodel.fit(X_train, y_train, epochs=10, batch_size=32)\nmodel.save('trained_model.h5')"
}

# Pydantic model for request
class AgentRequest(BaseModel):
    prompt: str
    refine: bool = False
    template: str = ""

# API endpoint with rate limiting
@app.post("/run-agent", dependencies=[Limiter.shared_limiter.limit("10/minute")])
async def run_agent(request: AgentRequest):
    try:
        prompt = request.prompt
        refine = request.refine
        template = request.template

        # Apply template if specified
        if template and template in ML_TEMPLATES:
            prompt = f"Using the {template} template, {prompt}"

        # Start timing
        start_time = time.time()

        # Initial agent response
        response = agent.run(prompt)
        duration = time.time() - start_time

        # Memory and performance metrics
        memory_usage = psutil.virtual_memory().percent
        token_speed = 512 / duration if duration > 0 else 0  # Approx tokens/second

        # Log to MLflow
        with mlflow.start_run():
            mlflow.log_metric("response_time", duration)
            mlflow.log_metric("memory_usage", memory_usage)

        # Refine response if requested
        if refine:
            refined_prompt = f"Refine this response: {response}. Ensure it is accurate for ML workflows."
            refined_response = agent.run(refined_prompt)
            response = refined_response

        return {
            "response": response,
            "metrics": {
                "response_time": duration,
                "memory_usage_percent": memory_usage,
                "token_speed_tokens_per_second": token_speed
            }
        }
    except Exception as e:
        logger.error(f"Error in run-agent: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Test inference on startup
@app.on_event("startup")
async def startup_event():
    test_prompt = "Write a Python script to preprocess a dataset for machine learning."
    logger.info(f"Testing inference with prompt: {test_prompt}")
    response = agent.run(test_prompt)
    logger.info(f"Response: {response}")

# Run the FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
