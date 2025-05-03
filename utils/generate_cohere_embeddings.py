import cohere
import os
from dotenv import load_dotenv
import json
from pathlib import Path


def generate_cohere_embeddings(input_file, output_file, api_key):
    """Generate embeddings for a dataset using Cohere API."""
    co = cohere.ClientV2(api_key=api_key)
    # open JSON file
    with open(input_file, "r") as f:
        data = json.load(f)
    # create output directory if it doesn't exist
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
    # create output file`
    for entry in data:
        text = entry["input"]
        print(f"Generating embedding for: {text}")
        response = co.embed(
            texts = [text],
            model = "embed-v4.0",
            input_type="classification",
            embedding_types=["float"],
            output_dimension=256
        )
        print(response)
        break
    # write embeddings to output file
    with open(output_file, "w") as f:
        for i, entry in enumerate(data):
            entry["embedding"] = response.embeddings[i]
        json.dump(data, f, indent=2)
    print(f"Embeddings saved to {output_file}")
if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("COHERE_API_KEY")
    parent_dir = Path(__file__).resolve().parent.parent
    input_file = os.path.join(parent_dir, "data", "math_dataset.json")
    output_file = os.path.join(parent_dir, "data","cohere_embeddings.json")
    generate_cohere_embeddings(input_file, output_file, api_key)

