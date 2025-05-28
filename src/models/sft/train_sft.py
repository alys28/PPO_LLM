import torch
from src.models.sft_baseline.sft_model import SFT_Model
from src.models.dataLoader import get_math_dataloader, get_math_val_dataloader
import yaml
import os
import json
from src.models.tokenizer import Tokenizer
# from src.models.sft_baseline.inference_sft import get_embedding

def train_sft(train_data_file, embedding_dim, num_transformer_layers, val_data_file, vocab, device_name, input_dim, max_seq_len, num_heads, num_epochs=10, batch_size=32, learning_rate=0.001, model_save_path="sft_model.pth"):
    """Train the SFT model."""
    assert max_seq_len > 0, "max_seq_len must be greater than 0"
    assert num_heads > 0, "num_heads must be greater than 0"
    assert num_epochs > 0, "num_epochs must be greater than 0"
    assert batch_size > 0, "batch_size must be greater than 0"
    assert learning_rate > 0, "learning_rate must be greater than 0"
    assert model_save_path, "model_save_path must be provided"
    tokenizer = Tokenizer(vocab)
    vocab_size = len(tokenizer)
    model = SFT_Model(vocab_size, embedding_dim, input_dim, max_seq_len, num_heads, num_transformer_layers)
    device = torch.device(device_name)
    print(f"Using device: {device}")
    # Move model to device
    if torch.cuda.is_available() and device_name == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    # Data loading
    # seq_len = answer_len + embedding_len = answer_len + 1
    data_loader = get_math_dataloader(train_data_file, device, vocab, batch_size, max_seq_len)
    # Validation
    val_data_loader = get_math_val_dataloader(val_data_file, device, vocab, batch_size, max_seq_len)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(data_loader):
            input_embedding, decoder_input, target, causal_mask, key_padding_mask = batch
            optimizer.zero_grad()
            
            logits = model(input_embedding, decoder_input, causal_mask, key_padding_mask)
            
            # Reshape logits and target for loss calculation
            logits = logits.view(-1, logits.size(-1))  # [batch_size * seq_len, vocab_size]
            target = target.view(-1)  # [batch_size * seq_len]
            
            # Calculate loss
            loss = criterion(logits, target)
            
            # Add penalty for early end token prediction
            pred_tokens = torch.argmax(logits, dim=-1)
            target_tokens = target
            # Find where end token is predicted before target end token
            # early_end_mask = (pred_tokens == tokenizer.end_token_id) & (target_tokens != tokenizer.end_token_id)
            # if early_end_mask.any():
            #     loss += 0.1 * early_end_mask.float().mean()
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

            # Print sample from batch every 10 batches
            # if batch_idx % 10 == 0:
            #     # Get predictions for first item in batch
            #     sample_logits = logits.view(batch_size, -1, vocab_size)[0]  # [seq_len, vocab_size]
            #     sample_target = target.view(batch_size, -1)[0]  # [seq_len]
                
            #     # Get predicted tokens
            #     predicted_tokens = torch.argmax(sample_logits.detach(), dim=-1) # Detach to avoid gradient computation
                
            #     # Decode sequences
            #     predicted_text = tokenizer.decode(predicted_tokens.tolist())
            #     target_text = tokenizer.decode(sample_target.tolist())
                
            #     print(f"\nBatch {batch_idx} Sample:")
            #     print(f"Target: {target_text}")
            #     print(f"Predicted: {predicted_text}")
            #     print(f"Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(data_loader)
        # scheduler.step()
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_batch in val_data_loader:
                val_input_embedding, val_decoder_input, val_target, val_causal_mask, val_key_padding_mask = val_batch
                current_batch_size = val_input_embedding.size(0)  # Get actual batch size
                
                # Forward pass
                val_logits = model(val_input_embedding, val_decoder_input, val_causal_mask, val_key_padding_mask)
                
                # Reshape for loss calculation
                val_logits = val_logits.view(-1, val_logits.size(-1))
                val_target = val_target.view(-1)
                
                # Calculate validation loss
                val_loss += criterion(val_logits, val_target).item()

                # # Print validation sample
                # sample_logits = val_logits.view(current_batch_size, -1, vocab_size)[0]
                # sample_target = val_target.view(current_batch_size, -1)[0]
                # predicted_tokens = torch.argmax(sample_logits.detach(), dim=-1) # Detach to avoid gradient computation
                # predicted_text = tokenizer.decode(predicted_tokens.tolist())
                # target_text = tokenizer.decode(sample_target.tolist())
                # print(f"\nValidation Sample:")
                # print(f"Target: {target_text}")
                # print(f"Predicted: {predicted_text}")
            # Try test question
            test_question = "What is -800 + 10?"
            test_answer = "-790"
            test_input_embedding = json.load(open(os.path.join("data", "embeddings", "cohere_embeddings_train.json")))[0]["embedding"]
            test_input_embedding = torch.tensor(test_input_embedding, dtype=torch.float, device=device).unsqueeze(0)
            current_tokens = torch.tensor([[tokenizer.start_token_id]], dtype=torch.long, device=device)
            for _ in range(5):
                logits = model(test_input_embedding, current_tokens)
                next_token_logits = logits[:, -1, :]  # Get predictions for next token
                probs = torch.softmax(next_token_logits, dim=-1)  # Convert logits to probabilities
                next_token = torch.argmax(probs, dim=-1, keepdim=True)
                # Append predicted token
                current_tokens = torch.cat([current_tokens, next_token], dim=1)
                if next_token.item() == tokenizer.end_token_id:
                    break
            # # Decode the generated sequence
            generated_text = tokenizer.decode(current_tokens[0].tolist())
            print("generated_text", generated_text)
        val_loss /= len(val_data_loader)

        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_loss:.4f}, Validation Loss: {val_loss:.4f}")
    
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")



if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    train_sft(config["train_data_file"], config["embedding_dim"], config["num_transformer_layers"], config["val_data_file"], config["vocab"], config["device"], config["input_dim"], config["max_seq_len"], config["num_heads"], config["num_epochs"], config["batch_size"], config["learning_rate"], config["model_save_path"])
    