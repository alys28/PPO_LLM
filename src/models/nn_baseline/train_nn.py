from src.models.nn_baseline.nn_model import NNModel
import yaml
import torch
from src.models.nn_baseline.dataLoader import get_math_dataloader, get_math_val_dataloader
import os

def train(config):
    train_data_file = config["train_data_file"]
    val_data_file = config["val_data_file"]
    input_dim = config["input_dim"]
    hidden_dims = config["hidden_dims"]
    num_epochs = config["num_epochs"]
    batch_size = config["batch_size"]
    learning_rate = config["learning_rate"]
    device_name = config["device"]
    model = NNModel(input_dim, hidden_dims)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    device = torch.device(device_name)
    print(f"Using device: {device}")
    # Move model to device
    if torch.cuda.is_available() and device_name == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")
    model.to(device)
    train_dl = get_math_dataloader(train_data_file, device, batch_size)
    val_dl = get_math_val_dataloader(val_data_file, device, 1)
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
        print(f"Epoch {i+1}, Train loss: {train_loss / len(train_dl)}, Val loss: {val_loss / len(val_dl)}")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    train(config)