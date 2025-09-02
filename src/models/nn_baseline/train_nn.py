from src.models.nn_baseline.nn_model import NNModel
import yaml
import torch
from src.models.nn_baseline.dataLoader import get_math_dataloader, get_math_val_dataloader
from src.models.nn_baseline.math_scaler import MathScaler
import os
import matplotlib.pyplot as plt
import json

def calculate_validation_accuracy(model, val_dataloader, scaler, tolerance=0.1):
    """Calculate validation accuracy with given tolerance."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in val_dataloader:
            x_val, y_val = batch
            
            # Get model predictions
            predictions = model(x_val)
            
            # Apply inverse scaling to get actual values
            if scaler.method != 'no_scale':
                actual_predictions = scaler.inverse_transform(predictions)
                actual_targets = scaler.inverse_transform(y_val)
            else:
                actual_predictions = predictions
                actual_targets = y_val
            
            # Calculate accuracy with tolerance
            for pred, target in zip(actual_predictions, actual_targets):
                pred_val = pred.item()
                target_val = target.item()
                
                # Handle division by zero for percentage calculation
                if abs(target_val) < 1e-8:
                    # If target is very close to zero, check if prediction is also close to zero
                    if abs(pred_val) < tolerance:
                        correct += 1
                else:
                    # Calculate percentage error
                    error_percent = abs(pred_val - target_val) / abs(target_val)
                    if error_percent <= tolerance:
                        correct += 1
                
                total += 1
    
    return correct / total if total > 0 else 0.0

def train(config):
    train_data_file = config["train_data_file"]
    val_data_file = config["val_data_file"]
    input_dim = config["input_dim"]
    hidden_dims = config["hidden_dims"]
    num_epochs = config["num_epochs"]
    batch_size = config["batch_size"]
    learning_rate = config["learning_rate"]
    device_name = config["device"]
    model_save_path = config["model_save_path"]
    graph_save_path = config["graph_save_path"]
    
    # Early stopping parameters
    early_stopping_patience = config.get("early_stopping_patience", 5)
    min_delta = config.get("min_delta", 0.001)
    model = NNModel(input_dim, hidden_dims)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    device = torch.device(device_name)
    print(f"Using device: {device}")
    
    # Move model to device
    if torch.cuda.is_available() and device_name == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        # Enable gradient checkpointing for memory efficiency
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled")
    else:
        print("Using CPU")
    model.to(device)
    
    # Create and fit math scaler on training data
    scaler = MathScaler(method='log_scale')  # Using corrected log scaling for wide range of mathematical values
    train_dl = get_math_dataloader(train_data_file, device, batch_size, normalizer=scaler, fit_normalizer=True)
    val_dl = get_math_val_dataloader(val_data_file, device, 1, normalizer=scaler, fit_normalizer=False)
    
    # Save scaler parameters
    scaler_path = config["model_save_path"].replace(".pth", "_scaler.json")
    scaler.save(scaler_path)
    print(f"Math scaler parameters saved to {scaler_path}")
    print(f"Math scaler method: {scaler.method}")
    
    # Lists to track losses for plotting
    train_losses = []
    val_losses = []
    epochs = []
    
    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for i in range(num_epochs):
        train_loss = 0
        for batch in train_dl:
            X_train, Y_train = batch
            optimizer.zero_grad()
            output = model(X_train)
            loss = criterion(output, Y_train)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        val_loss = 0
        with torch.no_grad():
            for batch in val_dl:
                x_val, y_val = batch
                output = model(x_val)
                val_loss += criterion(output, y_val).item()
        # Calculate average losses
        avg_train_loss = train_loss / len(train_dl)
        avg_val_loss = val_loss / len(val_dl)
        
        # Calculate validation accuracy with 10% tolerance
        val_accuracy = calculate_validation_accuracy(model, val_dl, scaler, tolerance=0.1)
        
        # Store losses for plotting
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        epochs.append(i + 1)
        
        print(f"Epoch {i+1}, Train loss: {avg_train_loss:.4f}, Val loss: {avg_val_loss:.4f}, Val accuracy (10%): {val_accuracy:.2%}")
        
        # Early stopping logic
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f"  → New best validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"  → No improvement for {patience_counter} epochs")
        
        # Check for early stopping
        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {i+1} epochs!")
            print(f"Best validation loss: {best_val_loss:.4f}")
            break
    
    # Create and save the loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss', color='blue', linewidth=2)
    plt.plot(epochs, val_losses, label='Validation Loss', color='red', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Validation Loss Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(graph_save_path, dpi=300, bbox_inches='tight')
    print(f"Loss plot saved to {graph_save_path}")
    plt.close()  # Close the figure to free memory
    
    # Load the best model state if early stopping was used
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model state (val loss: {best_val_loss:.4f})")
    
    torch.save(model.state_dict(), config["model_save_path"])
    print(f"Model saved to {config['model_save_path']}")
if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    train(config)