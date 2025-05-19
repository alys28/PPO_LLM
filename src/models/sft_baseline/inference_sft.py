import torch
from src.models.sft_baseline.sft_model import SFT_Model
import yaml
import os
import cohere
from dotenv import load_dotenv
from src.models.tokenizer import Tokenizer
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
def generate_answer(model, tokenizer, query, max_len, device, input_embedding=None):
    model.eval()
    with torch.no_grad():
        # Use provided embedding or generate new one
        if input_embedding is None:
            input_embedding = get_embedding(query)
        print("Input embedding shape:", input_embedding.shape)
        
        # Initialize with empty sequence
        current_tokens = torch.tensor([[tokenizer.start_token_id]], dtype=torch.long, device=device)
        for i in range(max_len):
            logits = model(input_embedding, current_tokens)
            print(f"Step {i} - Logits shape:", logits.shape)
            next_token_logits = logits[:, -1, :]  # Get predictions for next token
            probs = torch.softmax(next_token_logits, dim=-1)  # Convert logits to probabilities
            print(f"Step {i} - Probabilities:", probs)
            next_token = torch.argmax(probs, dim=-1, keepdim=True)
            print(f"Step {i} - Next token:", next_token.item())
            # Append predicted token
            current_tokens = torch.cat([current_tokens, next_token], dim=1)
            if next_token.item() == tokenizer.end_token_id:
                break
        # Decode the generated sequence
        generated_text = tokenizer.decode(current_tokens[0].tolist())
        return generated_text




if __name__ == "__main__":
    tokenizer = Tokenizer(config["vocab"])
    model = model = SFT_Model(
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

    # Test with the same case that works in training
    # test_input_embedding = get_embedding(test_question) 
    # test_input_embedding = torch.tensor(test_input_embedding, dtype=torch.float, device=config["device"]).unsqueeze(0)
    
    # # Test with hardcoded embedding
    # print("\nTesting with hardcoded embedding:")
    # output = generate_answer(model, tokenizer, test_question, config["max_seq_len"], config["device"], test_input_embedding)
    # print(f"Query: {test_question}")
    # print(f"Expected: {test_answer}")
    # print(f"Got: {output}")

    # Test with Cohere embedding
    print("\nTesting with Cohere embedding:")
    query = "What is 5 * 5?"
    output = generate_answer(model, tokenizer, query, config["max_seq_len"], config["device"])
    print(f"Query: {query}")
    print(f"Answer: {output}")