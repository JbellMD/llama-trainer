from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import torch
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def finetune_model():
    logger.info("Starting fine-tuning process...")
    
    # Create directories for outputs if they don't exist
    os.makedirs("./finetuned_model", exist_ok=True)
    
    # Load model and tokenizer
    model_name = "TinyLLaMA/TinyLLaMA-1.1B-Chat-v1.0"
    logger.info(f"Loading model: {model_name}")
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add padding token if doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        gradient_checkpointing=True  # Memory optimization
    )
    
    # Check if dataset exists, otherwise use a small sample dataset
    data_file = "dataset.jsonl"
    if not os.path.exists(data_file):
        logger.warning(f"Dataset file {data_file} not found. Creating a sample dataset.")
        
        # Create a small sample dataset for demonstration
        sample_data = [
            {"instruction": "Create a function to preprocess image data for a CNN.", 
             "response": "```python\ndef preprocess_images(image_paths, target_size=(224, 224)):\n    \"\"\"Preprocess images for CNN input.\n    \n    Args:\n        image_paths: List of paths to images\n        target_size: Target size for resizing\n        \n    Returns:\n        Preprocessed images as numpy array\n    \"\"\"\n    from PIL import Image\n    import numpy as np\n    \n    processed_images = []\n    for img_path in image_paths:\n        # Load and resize image\n        img = Image.open(img_path).resize(target_size)\n        # Convert to array and normalize\n        img_array = np.array(img) / 255.0\n        processed_images.append(img_array)\n    \n    return np.array(processed_images)\n```"},
            {"instruction": "Write code to train a random forest classifier.", 
             "response": "```python\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.metrics import accuracy_score\nimport pandas as pd\n\n# Load data\ndf = pd.read_csv('data.csv')\nX = df.drop('target', axis=1)\ny = df['target']\n\n# Split data\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n\n# Train model\nrf = RandomForestClassifier(n_estimators=100, random_state=42)\nrf.fit(X_train, y_train)\n\n# Evaluate\npredictions = rf.predict(X_test)\naccuracy = accuracy_score(y_test, predictions)\nprint(f'Accuracy: {accuracy:.4f}')\n```"}
        ]
        
        import json
        with open(data_file, 'w') as f:
            for item in sample_data:
                f.write(json.dumps(item) + '\n')
        
        logger.info(f"Created sample dataset with {len(sample_data)} examples.")
    
    # Prepare dataset
    logger.info("Loading and preprocessing dataset...")
    dataset = load_dataset("json", data_files=data_file, split="train")
    
    def preprocess_function(examples):
        inputs = examples["instruction"]
        targets = examples["response"]
        model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
        labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    logger.info(f"Dataset prepared with {len(tokenized_dataset)} examples")
    
    # Apply LoRA for efficient fine-tuning
    logger.info("Applying LoRA for parameter-efficient fine-tuning...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Training arguments with dynamic batching
    training_args = TrainingArguments(
        output_dir="./finetuned_model",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        save_steps=50,
        save_total_limit=2,
        logging_steps=10,
        learning_rate=2e-5,
        fp16=True,
        remove_unused_columns=False,
        dataloader_num_workers=2,  # Dynamic batching
        auto_find_batch_size=True  # Adjust batch size dynamically
    )
    
    # Initialize Trainer
    logger.info("Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    
    # Start fine-tuning
    logger.info("Starting fine-tuning...")
    trainer.train()
    
    # Save the fine-tuned model
    logger.info("Saving fine-tuned model...")
    model.save_pretrained("./finetuned_model")
    tokenizer.save_pretrained("./finetuned_model")
    
    logger.info("Fine-tuning complete!")
    return "./finetuned_model"

if __name__ == "__main__":
    finetune_model()
