"""
Main entry point for running the TinyLLaMA agent.
This script provides a simple command-line interface for interacting 
with the agent without going through the API.
"""

import argparse
import logging
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.agents import initialize_agent, Tool
from langchain.tools.file_management import ReadFileTool, WriteFileTool
from utils.metrics import PerformanceTracker, log_system_info
from utils.prompt_templates import format_ml_task_prompt, get_workflow_template
import os
import torch

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run TinyLLaMA agent")
    
    parser.add_argument(
        "--prompt", 
        type=str, 
        help="Prompt for the agent"
    )
    
    parser.add_argument(
        "--prompt_file", 
        type=str, 
        help="File containing the prompt"
    )
    
    parser.add_argument(
        "--workflow", 
        type=str, 
        choices=["data_preprocessing", "model_training", "model_inference"],
        help="Use a predefined workflow template"
    )
    
    parser.add_argument(
        "--workflow_params", 
        type=str, 
        help="JSON string with parameters for the workflow template"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        help="File to write output to"
    )
    
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto",
        choices=["cpu", "cuda", "rocm", "auto"],
        help="Device to run inference on"
    )
    
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.7,
        help="Sampling temperature"
    )
    
    parser.add_argument(
        "--max_length", 
        type=int, 
        default=512,
        help="Maximum generation length"
    )
    
    parser.add_argument(
        "--log_metrics", 
        action="store_true", 
        help="Log performance metrics"
    )
    
    return parser.parse_args()

def load_model(device="auto", temperature=0.7, max_length=512):
    """Load the TinyLLaMA model."""
    logger.info("Loading TinyLLaMA model...")
    
    # Load tokenizer
    model_name = "TinyLLaMA/TinyLLaMA-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Quantization configuration for memory efficiency
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map=device,
        trust_remote_code=True,
        gradient_checkpointing=True  # Memory optimization
    )
    
    # Create pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=max_length,
        temperature=temperature,
        top_p=0.9,
        repetition_penalty=1.2,
        device_map=device
    )
    
    llm = HuggingFacePipeline(pipeline=pipe)
    logger.info("Model loaded successfully!")
    
    return llm

def initialize_tools():
    """Initialize tools for the agent."""
    tools = [
        ReadFileTool(),
        WriteFileTool(),
    ]
    return tools

def main():
    """Main entry point."""
    args = parse_args()
    
    # Log system information
    if args.log_metrics:
        system_info = log_system_info()
        logger.info(f"System info: {system_info}")
    
    # Load the model
    llm = load_model(
        device=args.device,
        temperature=args.temperature,
        max_length=args.max_length
    )
    
    # Initialize tools
    tools = initialize_tools()
    
    # Initialize agent
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent="zero-shot-react-description",
        verbose=True
    )
    
    # Get the prompt
    prompt = None
    
    if args.prompt:
        prompt = args.prompt
    elif args.prompt_file:
        with open(args.prompt_file, 'r') as f:
            prompt = f.read()
    elif args.workflow:
        import json
        workflow_params = {}
        if args.workflow_params:
            workflow_params = json.loads(args.workflow_params)
        
        template = get_workflow_template(args.workflow, **workflow_params)
        prompt = f"Explain the following code and suggest any improvements:\n\n```python\n{template}\n```"
    else:
        prompt = input("Enter your prompt: ")
    
    # Initialize performance tracker if metrics are enabled
    tracker = None
    if args.log_metrics:
        tracker = PerformanceTracker()
        start_time = tracker.start_tracking()
    
    # Run agent
    logger.info(f"Running agent with prompt: {prompt[:100]}...")
    response = agent.run(prompt)
    
    # Log metrics if enabled
    if args.log_metrics and tracker:
        metrics = tracker.record_inference(start_time, len(response))
        logger.info(f"Performance metrics: {metrics}")
    
    # Print the response
    print("\n" + "="*50 + "\n")
    print(response)
    print("\n" + "="*50 + "\n")
    
    # Write output to file if specified
    if args.output:
        with open(args.output, 'w') as f:
            f.write(response)
        logger.info(f"Output written to {args.output}")

if __name__ == "__main__":
    main()
