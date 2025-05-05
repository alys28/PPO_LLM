import torch
from sft_model import SFT_Model
# import module in parent directory
from dataLoader import get_math_dataloader, get_math_val_dataloader # Fix this


def train_sft(num_epochs=10, batch_size=32):
    """Train the SFT model."""
    model = SFT_Model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    # Data loading
    data_loader = get_math_dataloader(batch_size)
    # Validation
    val_data_loader = get_math_val_dataloader(batch_size, shuffle=False)
    for epoch in range(num_epochs):
        for batch in data_loader:
            model.train()
            inputs, labels = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()
        # Validation
        model.eval()
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


# Inference
def generate_answer(model, input_embedding, max_len=10):
    model.eval()
    generated_tokens = []
    input_tokens = torch.zeros(1, 1).long()  # Start with a dummy input (e.g., <start> token)
    
    # Feed tokens into the model sequentially
    for _ in range(max_len):
        output = model(input_tokens)  # Model predicts next token based on input tokens
        next_token = output.argmax(dim=-1)[:, -1]  # Get the most probable next token
        generated_tokens.append(next_token.item())
        
        # Update input_tokens for next step (autoregressive prediction)
        input_tokens = torch.cat([input_tokens, next_token.unsqueeze(1)], dim=1)

    return generated_tokens