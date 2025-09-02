import torch
from src.models.nn_baseline.nn_model import NNModel
from src.models.nn_baseline.math_scaler import MathScaler
import yaml
import os
import cohere
from dotenv import load_dotenv
import json
import numpy as np

# Load environment variables
load_dotenv()
co = cohere.ClientV2(os.getenv("COHERE_API_KEY"))

with open(os.path.join(os.path.dirname(__file__), "config.yaml"), "r") as f:
    config = yaml.safe_load(f)

def get_embedding(query):
    """Generate embedding for the query using Cohere's embed-v4 model."""
    response = co.embed(
        texts=[query],  # Wrap query in a list
        model="embed-v4.0",
        input_type="classification",
        embedding_types=["float"],
        output_dimension=config["input_dim"]
    )
    return torch.tensor(response.embeddings.float[0], dtype=torch.float, device=config["device"]).unsqueeze(0)

def predict_answer(model, query, scaler, input_embedding=None, verbose=False):
    """Predict numerical answer for a math query using the neural network model."""
    model.eval()
    with torch.no_grad():
        # Use provided embedding or generate new one
        if input_embedding is None:
            input_embedding = get_embedding(query)
        if verbose:
            print("Input embedding shape:", input_embedding.shape)
        
        # Get prediction from the model
        prediction = model(input_embedding)
        if verbose:
            print("Raw prediction:", prediction.item())
        
        # Apply inverse scaling if needed
        if scaler.method != 'no_scale':
            actual_prediction = scaler.inverse_transform(prediction)
            if verbose:
                print("Descaled prediction:", actual_prediction.item())
            return actual_prediction.item()
        else:
            return prediction.item()

def evaluate_model_accuracy(model, scaler, device, tolerance=0.01):
    """Evaluate model accuracy on validation dataset."""
    # Load validation data
    with open(config["val_data_file"], 'r') as f:
        val_data = json.load(f)
    
    correct = 0
    total = len(val_data)
    
    print(f"\nEvaluating model on {total} validation examples...")
    print(f"Using tolerance: {tolerance}")
    print(f"Scaler method: {scaler.method}")
    
    for i, example in enumerate(val_data):
        # Get the query and expected answer
        query = example["input"]
        expected_answer = float(example["answer"])
        embedding = torch.tensor(example["embedding"], dtype=torch.float, device=device).unsqueeze(0)
        
        # Generate model's prediction
        model_prediction = predict_answer(model, query, scaler, embedding)
        
        # Compare predictions (using tolerance for numerical comparison)
        if abs(model_prediction - expected_answer) <= tolerance:
            correct += 1
        
        # Print progress every 10 examples
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{total} examples. Current accuracy: {correct/(i+1):.2%}")
            print(f"Example {i+1}: Query='{query}', Expected={expected_answer:.4f}, Predicted={model_prediction:.4f}")
    
    accuracy = correct / total
    print(f"\nFinal accuracy: {accuracy:.2%} ({correct}/{total} correct)")
    return accuracy

def evaluate_model_mae(model, scaler, device):
    """Evaluate model using Mean Absolute Error."""
    # Load validation data
    with open(config["val_data_file"], 'r') as f:
        val_data = json.load(f)
    
    total_error = 0
    total = len(val_data)
    
    print(f"\nEvaluating model MAE on {total} validation examples...")
    print(f"Scaler method: {scaler.method}")
    
    for i, example in enumerate(val_data):
        # Get the query and expected answer
        query = example["input"]
        expected_answer = float(example["answer"])
        embedding = torch.tensor(example["embedding"], dtype=torch.float, device=device).unsqueeze(0)
        
        # Generate model's prediction
        model_prediction = predict_answer(model, query, scaler, embedding)
        
        # Calculate absolute error
        error = abs(model_prediction - expected_answer)
        total_error += error
        
        # Print progress every 10 examples
        if (i + 1) % 10 == 0:
            current_mae = total_error / (i + 1)
            print(f"Processed {i + 1}/{total} examples. Current MAE: {current_mae:.4f}")
    
    mae = total_error / total
    print(f"\nFinal MAE: {mae:.4f}")
    return mae

if __name__ == "__main__":
    # Load model
    model = NNModel(
        input_dim=config["input_dim"],
        hidden_dims=config["hidden_dims"]
    )
    model.load_state_dict(torch.load(config["model_save_path"]))
    model.to(config["device"])
    model.eval()
    
    # Load math scaler
    scaler_path = config["model_save_path"].replace(".pth", "_scaler.json")
    scaler = MathScaler()
    scaler.load(scaler_path)
    
    print("Neural Network Baseline Model loaded successfully!")
    print(f"Model architecture: {config['input_dim']} -> {config['hidden_dims']}")
    print(f"Math scaler method: {scaler.method}")
    
    # Evaluate model accuracy with different tolerances
    print("\n" + "="*50)
    print("ACCURACY EVALUATION")
    print("="*50)
    
    for tolerance in [0.01, 0.1, 1.0, 10.0, 100.0]:
        print(f"\nTolerance: {tolerance}")
        accuracy = evaluate_model_accuracy(model, scaler, config["device"], tolerance=tolerance)
    
    # Evaluate model using MAE
    print("\n" + "="*50)
    print("MEAN ABSOLUTE ERROR EVALUATION")
    print("="*50)
    mae = evaluate_model_mae(model, scaler, config["device"])
    
    # Test with a sample query
    print("\n" + "="*50)
    print("SAMPLE INFERENCE")
    print("="*50)
    test_query = "What is 15 * 23?"
    print(f"Query: {test_query}")
    prediction = predict_answer(model, test_query, scaler, verbose=True)
    print(f"Predicted answer: {prediction:.4f}")
    print(f"Expected answer: {15 * 23}")
