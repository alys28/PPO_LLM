import torch
from src.models.sft_baseline.sft_model import SFT_Model
from src.models.dataLoader import get_math_dataloader, get_math_val_dataloader
import yaml
import os
from src.models.tokenizer import Tokenizer
from src.models.sft_baseline.inference_sft import get_embedding

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
            test_input_embedding = [
      -0.18945312,
      -0.061035156,
      0.06298828,
      0.064941406,
      -0.09472656,
      -0.033203125,
      -0.05859375,
      -0.14746094,
      0.0625,
      0.08886719,
      0.056396484,
      -0.016235352,
      -0.15820312,
      -0.072753906,
      -0.032470703,
      0.1328125,
      0.012084961,
      0.012390137,
      0.027954102,
      -0.0079956055,
      0.07763672,
      0.018676758,
      0.005554199,
      -0.12988281,
      -0.028564453,
      0.057373047,
      0.07421875,
      -0.045410156,
      0.02368164,
      0.017578125,
      -0.061523438,
      -0.0126953125,
      -0.04296875,
      0.11035156,
      -0.13671875,
      0.1328125,
      -0.052490234,
      -0.09277344,
      0.05517578,
      0.044189453,
      0.010437012,
      -0.030761719,
      0.009277344,
      0.010864258,
      0.063964844,
      0.02319336,
      0.071777344,
      0.0010681152,
      0.026367188,
      -0.05029297,
      -0.071777344,
      -0.01574707,
      -0.076171875,
      -0.10058594,
      0.0546875,
      0.029785156,
      0.05419922,
      0.14648438,
      0.05810547,
      0.056152344,
      -0.04736328,
      -0.049316406,
      -0.076171875,
      0.006225586,
      0.13476562,
      -0.026489258,
      0.006225586,
      0.12109375,
      -0.015380859,
      -0.016845703,
      -0.03515625,
      0.11230469,
      -0.041259766,
      -0.021484375,
      0.08886719,
      0.0028533936,
      -0.0390625,
      -0.080566406,
      0.02746582,
      -0.004486084,
      0.01574707,
      -0.04345703,
      0.11035156,
      -0.11230469,
      0.14355469,
      0.056396484,
      0.063964844,
      -0.12792969,
      -0.023071289,
      -0.007537842,
      0.023803711,
      0.018188477,
      0.055908203,
      -0.030761719,
      0.019042969,
      -0.034179688,
      0.0020599365,
      0.030273438,
      -0.02355957,
      -0.008544922,
      0.016357422,
      0.064941406,
      -0.07763672,
      -0.045654297,
      -0.068359375,
      0.1328125,
      -0.015075684,
      0.044921875,
      0.0077209473,
      -0.0074157715,
      0.01977539,
      -0.13085938,
      -0.06298828,
      -0.12597656,
      0.022583008,
      -0.023071289,
      0.014343262,
      0.041259766,
      -0.026367188,
      -0.0546875,
      -0.05859375,
      0.04638672,
      -0.103515625,
      -0.030883789,
      0.051757812,
      -0.111816406,
      -0.08154297,
      -0.0048217773,
      0.030273438,
      0.061035156,
      -0.106933594,
      0.051513672,
      0.08642578,
      -0.011413574,
      -0.017333984,
      0.057373047,
      0.010009766,
      0.0009994507,
      0.005004883,
      -0.1328125,
      -0.07470703,
      -0.017089844,
      0.021484375,
      0.017944336,
      -0.12060547,
      0.10205078,
      0.052490234,
      0.033203125,
      -0.088378906,
      0.022705078,
      -0.00062942505,
      0.0067749023,
      -0.091308594,
      0.045654297,
      -0.016357422,
      0.0021514893,
      0.091308594,
      0.049316406,
      -0.045410156,
      0.051757812,
      0.032714844,
      -0.071777344,
      0.0055236816,
      0.017578125,
      0.04345703,
      -0.051513672,
      0.011779785,
      -0.09033203,
      -0.033203125,
      -0.12988281,
      -0.026733398,
      0.04321289,
      -0.02368164,
      0.01373291,
      0.107421875,
      0.057373047,
      -0.05810547,
      0.0154418945,
      0.0019073486,
      -0.041015625,
      0.012634277,
      0.026733398,
      0.052490234,
      -0.059326172,
      0.0076904297,
      -0.014282227,
      0.016479492,
      0.114746094,
      0.044433594,
      0.008728027,
      -0.040039062,
      0.0703125,
      -0.02355957,
      -0.024414062,
      -0.06591797,
      0.09033203,
      0.023071289,
      0.029296875,
      -0.045410156,
      -0.033203125,
      0.08544922,
      -0.0038452148,
      -0.041992188,
      -0.09277344,
      0.04638672,
      -0.0066223145,
      -0.056396484,
      -0.083984375,
      0.039794922,
      -0.014038086,
      0.076171875,
      0.020507812,
      -0.016235352,
      0.0055236816,
      -0.042236328,
      0.06298828,
      -0.02709961,
      -0.023071289,
      0.06298828,
      0.02734375,
      -0.032470703,
      -0.05834961,
      -0.016845703,
      0.13183594,
      0.037109375,
      0.0043029785,
      0.055664062,
      -0.11035156,
      -0.011413574,
      -0.06298828,
      -0.0057678223,
      0.013000488,
      -0.099121094,
      0.0012207031,
      -0.06738281,
      0.030883789,
      -0.07763672,
      -0.008972168,
      -0.103515625,
      0.03857422,
      0.0021820068,
      -0.035888672,
      0.016845703,
      0.040771484,
      0.018432617,
      0.1171875,
      0.0021820068,
      0.0066223145,
      -0.12109375,
      0.0012741089,
      -0.026977539,
      0.0059509277,
      0.03515625,
      0.063964844,
      0.017333984,
      0.08642578
    ] 
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
    