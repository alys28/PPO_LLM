import torch
from src.models.sft_baseline.sft_model import SFT_Model
import yaml
import os
import cohere
from dotenv import load_dotenv
from src.models.tokenizer import Tokenizer
import json

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

# Inference
def generate_answer(model, tokenizer, query, max_len, device, input_embedding=None, verbose=False):
    model.eval()
    with torch.no_grad():
        # Use provided embedding or generate new one
        if input_embedding is None:
            input_embedding = get_embedding(query)
        if verbose:
            print("Input embedding shape:", input_embedding.shape)
        
        # Initialize with empty sequence
        current_tokens = torch.tensor([[tokenizer.start_token_id]], dtype=torch.long, device=device)
        for i in range(1, max_len):
            logits = model(input_embedding, current_tokens)
            if verbose:
                print(f"Step {i} - Logits shape:", logits.shape)
            next_token_logits = logits[:, -1, :]  # Get predictions for next token
            probs = torch.softmax(next_token_logits, dim=-1)  # Convert logits to probabilities
            if verbose:
                print(f"Step {i} - Probabilities:", probs)
            next_token = torch.argmax(probs, dim=-1, keepdim=True)
            if verbose:
                print(f"Step {i} - Next token:", next_token.item())
            # Append predicted token
            current_tokens = torch.cat([current_tokens, next_token], dim=1)
            if next_token.item() == tokenizer.end_token_id:
                break
        # Decode the generated sequence
        generated_text = tokenizer.decode(current_tokens[0].tolist())
        return generated_text

def evaluate_model_accuracy(model, tokenizer, device):
    """Evaluate model accuracy on validation dataset."""
    # Load validation data
    with open(config["val_data_file"], 'r') as f:
        val_data = json.load(f)
    
    correct = 0
    total = len(val_data)
    
    print(f"\nEvaluating model on {total} validation examples...")
    
    for i, example in enumerate(val_data):
        # Get the query and expected answer
        query = example["input"]
        expected_answer = str(example["answer"])
        embedding = torch.tensor(example["embedding"], dtype=torch.float, device=device).unsqueeze(0)
        tokenized_answer = tokenizer.encode(expected_answer, add_start_token=True, add_end_token=True, add_pad_token=False)
        
        answer_to_compare = tokenizer.decode(tokenized_answer)
        
        # Generate model's answer
        model_answer = generate_answer(model, tokenizer, query, config["max_seq_len"], device, embedding)
        print(f"Actual answer: {answer_to_compare}")
        print(f"Model answer: {model_answer}")
        # Compare answers (strip whitespace and convert to string for comparison)
        if model_answer.strip() == answer_to_compare.strip():
            correct += 1
        
        # Print progress every 10 examples
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{total} examples. Current accuracy: {correct/(i+1):.2%}")
    
    accuracy = correct / total
    print(f"\nFinal accuracy: {accuracy:.2%} ({correct}/{total} correct)")
    return accuracy

if __name__ == "__main__":
    tokenizer = Tokenizer(config["vocab"])
    model = SFT_Model(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=config["embedding_dim"],
        input_dim=config["input_dim"],
        max_seq_len=config["max_seq_len"],
        num_heads=config["num_heads"],
        num_transformer_layers=config["num_transformer_layers"]
    )
    model.load_state_dict(torch.load(config["model_save_path"]))
    model.to(config["device"])
    model.eval()

    # Evaluate model accuracy
    accuracy = evaluate_model_accuracy(model, tokenizer, config["device"])
    
    # Test with Cohere embedding
    # print("\nTesting with Cohere embedding:")
    # query = "What is -491 * 620?"
    # output = generate_answer(model, tokenizer, query, config["max_seq_len"], config["device"])
    # print(f"Query: {query}")
    # print(f"Answer: {output}")