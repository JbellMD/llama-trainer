"""
A simple example of using the TinyLLaMA agent from Python code.
"""

import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.agents import initialize_agent, Tool
from langchain.tools.file_management import ReadFileTool, WriteFileTool
from utils.metrics import PerformanceTracker, log_system_info

def main():
    """Run a simple example of the agent."""
    print("Loading TinyLLaMA model (this may take a moment)...")
    
    # Load model and tokenizer
    model_name = "TinyLLaMA/TinyLLaMA-1.1B-Chat-v1.0"
    
    # Quantization configuration
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
        trust_remote_code=True
    )
    
    # Create pipeline
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
    
    # Set up tools
    tools = [
        ReadFileTool(),
        WriteFileTool()
    ]
    
    # Initialize agent
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent="zero-shot-react-description",
        verbose=True
    )
    
    # Log system information
    system_info = log_system_info()
    print(f"System information:\n{system_info}")
    
    # Initialize performance tracker
    tracker = PerformanceTracker()
    
    # Example prompts
    example_prompts = [
        "Create a function to preprocess a CSV dataset for a machine learning task",
        "Explain how neural networks work and provide a simple example",
        "Write code to implement a basic decision tree classifier"
    ]
    
    # Run agent with each prompt
    for i, prompt in enumerate(example_prompts):
        print(f"\n\n{'='*50}")
        print(f"Example {i+1}: {prompt}")
        print(f"{'='*50}\n")
        
        # Track performance
        start_time = tracker.start_tracking()
        
        # Run agent
        response = agent.run(prompt)
        
        # Record metrics
        metrics = tracker.record_inference(start_time, len(response))
        
        # Print response and metrics
        print(f"\nResponse:\n{response}")
        print(f"\nPerformance metrics:")
        print(f"- Duration: {metrics['duration']:.2f} seconds")
        print(f"- Memory usage: {metrics['memory_percent']:.2f}%")
        print(f"- Tokens per second: {metrics['tokens_per_second']:.2f}")
    
    # Print summary statistics
    print("\n\nSummary statistics:")
    tracker.log_summary()

if __name__ == "__main__":
    # Create the examples directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
    main()
