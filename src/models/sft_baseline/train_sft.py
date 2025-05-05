import torch
from sft_model import SFT_Model
# import module in parent directory
from dataLoader import DataLoader # Fix this


def train_sft(num_epochs=10, batch_size=32):
    """Train the SFT model."""
    model = SFT_Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    num_epochs = 10
    # Data loading
    data_loader = DataLoader(batch_size, shuffle=True)
    # Validation
    val_data_loader = DataLoader(batch_size, shuffle=False)
    for epoch in range(num_epochs):
        for batch in data_loader:
            inputs, labels = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        # Validation
        val_loss = 0
        for val_batch in val_data_loader:
            val_inputs, val_labels = val_batch
            with torch.no_grad():
                val_outputs = model(val_inputs)
                val_loss += criterion(val_outputs, val_labels).item()
        val_loss /= len(val_data_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}, Validation Loss: {val_loss}")
    torch.save(model.state_dict(), "sft_model.pth")
    print("Model saved to sft_model.pth")
