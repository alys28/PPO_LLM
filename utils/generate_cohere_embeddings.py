import cohere
import os
from dotenv import load_dotenv
import json
from pathlib import Path


def generate_cohere_embeddings(input_file, output_file, api_key, batch_size=96):
    """Generate embeddings for a dataset using Cohere API.
    
    Args:
        input_file (str): Path to input JSON file
        output_file (str): Path to output JSON file
        api_key (str): Cohere API key
        batch_size (int): Number of texts to process in each batch
    """
    co = cohere.ClientV2(api_key=api_key)
    
    # open JSON file
    with open(input_file, "r") as f:
        data = json.load(f)
    
    # create output directory if it doesn't exist
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
    
    # Process data in batches
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        texts = [entry["input"] for entry in batch]
        print(f"Generating embeddings for batch {i//batch_size + 1}/{(len(data) + batch_size - 1)//batch_size} (batch size: {len(batch)})")
        
        response = co.embed(
            texts=texts,
            model="embed-v4.0",
            input_type="classification",
            embedding_types=["float"],
            output_dimension=256
        )
        
        # Store embeddings in the corresponding entries
        for j, entry in enumerate(batch):
            entry["embedding"] = response.embeddings.float[j]
    
    # write embeddings to output file
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Embeddings saved to {output_file}")

if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("COHERE_API_KEY")
    parent_dir = Path(__file__).resolve().parent.parent
    input_file = os.path.join(parent_dir, "data", "math_dataset_val.json")
    output_file = os.path.join(parent_dir, "data","cohere_embeddings_val.json")
    generate_cohere_embeddings(input_file, output_file, api_key)
