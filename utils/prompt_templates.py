"""
Prompt templates for the TinyLLaMA agent.
"""

from typing import Dict, List, Optional, Any, Union
import json

# Base system prompt for the agent
SYSTEM_PROMPT = """You are an AI assistant specialized in machine learning and coding tasks.
Your purpose is to help users with machine learning workflows, code generation, and data analysis.
You are running on a system with AMD hardware (Ryzen 7 8845HS, Radeon 780M) with 16GB RAM.

You have access to the following tools:
- Shell commands (for executing code)
- File reading and writing
- Git operations
- MLflow for experiment tracking

When providing code solutions:
1. Be concise and follow Python best practices
2. Consider memory constraints of the system
3. Provide explanations for complex parts
4. Use libraries like scikit-learn, TensorFlow, or PyTorch appropriately
"""

# Template for general ML task prompts
ML_TASK_TEMPLATE = """{system_prompt}

User's task: {user_task}

Please provide a solution that includes:
- Explanation of the approach
- Required libraries and dependencies
- Complete, working code
- Potential optimizations

"""

# Template for code refinement
CODE_REFINEMENT_TEMPLATE = """{system_prompt}

Original code:
```python
{original_code}
```

Feedback/issues to address:
{feedback}

Please refine the code to address these issues while maintaining its core functionality.
"""

# Templates for ML workflows
ML_WORKFLOW_TEMPLATES = {
    "data_preprocessing": """import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv("{dataset_path}")

# Define numeric and categorical features
numeric_features = {numeric_features}
categorical_features = {categorical_features}

# Create preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Split the data
X = data.drop('{target_column}', axis=1)
y = data['{target_column}']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess the data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Save preprocessed data if needed
np.save('X_train_processed.npy', X_train_processed)
np.save('X_test_processed.npy', X_test_processed)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)

print(f"Preprocessed training data shape: {X_train_processed.shape}")
print(f"Preprocessed testing data shape: {X_test_processed.shape}")
""",

    "model_training": """import numpy as np
from sklearn.{model_type} import {model_class}
from sklearn.metrics import {metrics}
import pickle
import mlflow

# Load preprocessed data
X_train = np.load('X_train_processed.npy')
X_test = np.load('X_test_processed.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

# Initialize model
model = {model_class}({model_params})

# Train model
with mlflow.start_run():
    mlflow.log_params({model_params})
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    {metric_calculations}
    
    # Log metrics
    {metric_logging}
    
    # Save model
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    mlflow.sklearn.log_model(model, "model")
    
    print("Model trained and saved successfully!")
""",

    "model_inference": """import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Function for inference
def predict(data):
    \"\"\"
    Make predictions on new data.
    
    Args:
        data: DataFrame or array of features
        
    Returns:
        Array of predictions
    \"\"\"
    # Preprocess the data (must match training preprocessing)
    # This is a simplified example - you'll need to adjust based on your actual preprocessing
    if isinstance(data, pd.DataFrame):
        # Handle DataFrame input
        data = data.values
    
    # Make predictions
    predictions = model.predict(data)
    return predictions

# Example usage
if __name__ == "__main__":
    # Example new data (replace with your actual data loading)
    new_data = pd.read_csv("new_data.csv")
    
    # Make predictions
    predictions = predict(new_data)
    
    # Save or display results
    results = pd.DataFrame({"prediction": predictions})
    results.to_csv("predictions.csv", index=False)
    print(f"Generated {len(predictions)} predictions")
"""
}


def format_ml_task_prompt(user_task: str) -> str:
    """
    Format a prompt for a general ML task.
    
    Args:
        user_task: The user's ML task description
        
    Returns:
        Formatted prompt
    """
    return ML_TASK_TEMPLATE.format(
        system_prompt=SYSTEM_PROMPT,
        user_task=user_task
    )


def format_code_refinement_prompt(original_code: str, feedback: str) -> str:
    """
    Format a prompt for code refinement.
    
    Args:
        original_code: The original code to refine
        feedback: Feedback or issues to address
        
    Returns:
        Formatted prompt
    """
    return CODE_REFINEMENT_TEMPLATE.format(
        system_prompt=SYSTEM_PROMPT,
        original_code=original_code,
        feedback=feedback
    )


def get_workflow_template(template_name: str, **kwargs) -> str:
    """
    Get a workflow template with parameters filled in.
    
    Args:
        template_name: Name of the template
        **kwargs: Template parameters
        
    Returns:
        Formatted template or empty string if template not found
    """
    if template_name not in ML_WORKFLOW_TEMPLATES:
        return ""
    
    template = ML_WORKFLOW_TEMPLATES[template_name]
    try:
        return template.format(**kwargs)
    except KeyError as e:
        missing_key = str(e).strip("'")
        raise ValueError(f"Missing required parameter for template '{template_name}': {missing_key}")


# Example usage
if __name__ == "__main__":
    # Example of using the ML task template
    prompt = format_ml_task_prompt("Create a model to classify images of cats vs dogs")
    print(prompt)
    print("\n" + "-"*50 + "\n")
    
    # Example of using the code refinement template
    refined_prompt = format_code_refinement_prompt(
        "def train_model(X, y):\n    model = RandomForest()\n    model.fit(X, y)\n    return model",
        "The model training function doesn't have proper error handling or hyperparameters"
    )
    print(refined_prompt)
    print("\n" + "-"*50 + "\n")
    
    # Example of using a workflow template
    try:
        preprocessing_code = get_workflow_template(
            "data_preprocessing",
            dataset_path="data.csv",
            numeric_features=["age", "income", "score"],
            categorical_features=["education", "occupation"],
            target_column="target"
        )
        print(preprocessing_code)
    except ValueError as e:
        print(f"Error: {e}")
