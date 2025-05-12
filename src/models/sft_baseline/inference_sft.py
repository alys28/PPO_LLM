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
        current_tokens = torch.tensor([[]], dtype=torch.long, device=device)
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
    model = SFT_Model(tokenizer.vocab_size, config["input_dim"], config["max_seq_len"], config["num_heads"])
    model.load_state_dict(torch.load(config["model_save_path"]))
    model.to(config["device"])
    model.eval()

    # Test with the same case that works in training
    test_question = "What is -800 + 10?"
    test_answer = "-790"
    test_input_embedding = [
        -0.18945312, -0.061035156, 0.06298828, 0.064941406, -0.09472656, -0.033203125, -0.05859375, -0.14746094,
        0.0625, 0.08886719, 0.056396484, -0.016235352, -0.15820312, -0.072753906, -0.032470703, 0.1328125,
        0.012084961, 0.012390137, 0.027954102, -0.0079956055, 0.07763672, 0.018676758, 0.005554199, -0.12988281,
        -0.028564453, 0.057373047, 0.07421875, -0.045410156, 0.02368164, 0.017578125, -0.061523438, -0.0126953125,
        -0.04296875, 0.11035156, -0.13671875, 0.1328125, -0.052490234, -0.09277344, 0.05517578, 0.044189453,
        0.010437012, -0.030761719, 0.009277344, 0.010864258, 0.063964844, 0.02319336, 0.071777344, 0.0010681152,
        0.026367188, -0.05029297, -0.071777344, -0.01574707, -0.076171875, -0.10058594, 0.0546875, 0.029785156,
        0.05419922, 0.14648438, 0.05810547, 0.056152344, -0.04736328, -0.049316406, -0.076171875, 0.006225586,
        0.13476562, -0.026489258, 0.006225586, 0.12109375, -0.015380859, -0.016845703, -0.03515625, 0.11230469,
        -0.041259766, -0.021484375, 0.08886719, 0.0028533936, -0.0390625, -0.080566406, 0.02746582, -0.004486084,
        0.01574707, -0.04345703, 0.11035156, -0.11230469, 0.14355469, 0.056396484, 0.063964844, -0.12792969,
        -0.023071289, -0.007537842, 0.023803711, 0.018188477, 0.055908203, -0.030761719, 0.019042969, -0.034179688,
        0.0020599365, 0.030273438, -0.02355957, -0.008544922, 0.016357422, 0.064941406, -0.07763672, -0.045654297,
        -0.068359375, 0.1328125, -0.015075684, 0.044921875, 0.0077209473, -0.0074157715, 0.01977539, -0.13085938,
        -0.06298828, -0.12597656, 0.022583008, -0.023071289, 0.014343262, 0.041259766, -0.026367188, -0.0546875,
        -0.05859375, 0.04638672, -0.103515625, -0.030883789, 0.051757812, -0.111816406, -0.08154297, -0.0048217773,
        0.030273438, 0.061035156, -0.106933594, 0.051513672, 0.08642578, -0.011413574, -0.017333984, 0.057373047,
        0.010009766, 0.0009994507, 0.005004883, -0.1328125, -0.07470703, -0.017089844, 0.021484375, 0.017944336,
        -0.12060547, 0.10205078, 0.052490234, 0.033203125, -0.088378906, 0.022705078, -0.00062942505, 0.0067749023,
        -0.091308594, 0.045654297, -0.016357422, 0.0021514893, 0.091308594, 0.049316406, -0.045410156, 0.051757812,
        0.032714844, -0.071777344, 0.0055236816, 0.017578125, 0.04345703, -0.051513672, 0.011779785, -0.09033203,
        -0.033203125, -0.12988281, -0.026733398, 0.04321289, -0.02368164, 0.01373291, 0.107421875, 0.057373047,
        -0.05810547, 0.0154418945, 0.0019073486, -0.041015625, 0.012634277, 0.026733398, 0.052490234, -0.059326172,
        0.0076904297, -0.014282227, 0.016479492, 0.114746094, 0.044433594, 0.008728027, -0.040039062, 0.0703125,
        -0.02355957, -0.024414062, -0.06591797, 0.09033203, 0.023071289, 0.029296875, -0.045410156, -0.033203125,
        0.08544922, -0.0038452148, -0.041992188, -0.09277344, 0.04638672, -0.0066223145, -0.056396484, -0.083984375,
        0.039794922, -0.014038086, 0.076171875, 0.020507812, -0.016235352, 0.0055236816, -0.042236328, 0.06298828,
        -0.02709961, -0.023071289, 0.06298828, 0.02734375, -0.032470703, -0.05834961, -0.016845703, 0.13183594,
        0.037109375, 0.0043029785, 0.055664062, -0.11035156, -0.011413574, -0.06298828, -0.0057678223, 0.013000488,
        -0.099121094, 0.0012207031, -0.06738281, 0.030883789, -0.07763672, -0.008972168, -0.103515625, 0.03857422,
        0.0021820068, -0.035888672, 0.016845703, 0.040771484, 0.018432617, 0.1171875, 0.0021820068, 0.0066223145,
        -0.12109375, 0.0012741089, -0.026977539, 0.0059509277, 0.03515625, 0.063964844, 0.017333984, 0.08642578
    ]
    test_input_embedding = torch.tensor(test_input_embedding, dtype=torch.float, device=config["device"]).unsqueeze(0)
    
    # Test with hardcoded embedding
    print("\nTesting with hardcoded embedding:")
    output = generate_answer(model, tokenizer, test_question, config["max_seq_len"], config["device"], test_input_embedding)
    print(f"Query: {test_question}")
    print(f"Expected: {test_answer}")
    print(f"Got: {output}")

    # Test with Cohere embedding
    print("\nTesting with Cohere embedding:")
    query = "What is 5 + 5?"
    output = generate_answer(model, tokenizer, query, config["max_seq_len"], config["device"])
    print(f"Query: {query}")
    print(f"Answer: {output}")