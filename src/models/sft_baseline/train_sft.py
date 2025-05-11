import torch
from src.models.sft_baseline.sft_model import SFT_Model
from src.models.dataLoader import get_math_dataloader, get_math_val_dataloader
import yaml
import os
from src.models.tokenizer import Tokenizer
def train_sft(train_data_file, val_data_file, vocab, device_name, input_dim, max_seq_len, num_heads, num_epochs=10, batch_size=32, learning_rate=0.001, model_save_path="sft_model.pth"):
    """Train the SFT model."""
    assert max_seq_len > 0, "max_seq_len must be greater than 0"
    assert num_heads > 0, "num_heads must be greater than 0"
    assert num_epochs > 0, "num_epochs must be greater than 0"
    assert batch_size > 0, "batch_size must be greater than 0"
    assert learning_rate > 0, "learning_rate must be greater than 0"
    assert model_save_path, "model_save_path must be provided"
    tokenizer = Tokenizer(vocab)
    vocab_size = len(tokenizer)
    model = SFT_Model(vocab_size, input_dim, max_seq_len, num_heads)
    device = torch.device(device_name)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    # Data loading
    # seq_len = answer_len + embedding_len = answer_len + 1
    data_loader = get_math_dataloader(train_data_file, device, vocab, batch_size, max_seq_len)
    # Validation
    val_data_loader = get_math_val_dataloader(val_data_file, device, vocab, batch_size, max_seq_len)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in data_loader:
            input_embedding, decoder_input, target, causal_mask, key_padding_mask = batch
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(input_embedding, decoder_input, causal_mask, key_padding_mask)
            
            # Reshape logits and target for loss calculation
            logits = logits.view(-1, logits.size(-1))  # [batch_size * seq_len, vocab_size]
            target = target.view(-1)  # [batch_size * seq_len]
            
            # Calculate loss
            loss = criterion(logits, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(data_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_batch in val_data_loader:
                val_input_embedding, val_decoder_input, val_target, val_causal_mask, val_key_padding_mask = val_batch
                
                # Forward pass
                val_logits = model(val_input_embedding, val_decoder_input, val_causal_mask, val_key_padding_mask)
                
                # Reshape for loss calculation
                val_logits = val_logits.view(-1, val_logits.size(-1))
                val_target = val_target.view(-1)
                
                # Calculate validation loss
                val_loss += criterion(val_logits, val_target).item()
        
        val_loss /= len(val_data_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_loss:.4f}, Validation Loss: {val_loss:.4f}")
    
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")


# Inference
def generate_answer(model, input_embedding, causal_mask, key_padding_mask, max_len=10):
    model.eval()
    with torch.no_grad():
        logits = model(input_embedding, causal_mask, key_padding_mask)
        # Get the most probable next token
        next_token = logits.argmax(dim=-1)[:, -1]
        return next_token


if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    train_sft(config["train_data_file"], config["val_data_file"], config["vocab"], config["device"], config["input_dim"], config["max_seq_len"], config["num_heads"], config["num_epochs"], config["batch_size"], config["learning_rate"], config["model_save_path"])
