import cohere
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.preprocessing import normalize
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("COHERE_API_KEY")
co = cohere.ClientV2(api_key=api_key)


expressions = [
    "800 + 10", "-800 + 10",
    "800 * 10", "-800 * 10",
    "800 / 10", "-800 / 10",
    "800 - 10", "-800 - 10",
    "10 + 800", "10 - 800",
    "123 + 456", "456 - 123",
    "15 * 3", "15 / 3",
    "2 + 2", "5 * 5",
    "25 * 25", "100 / 4",
    "1000 + 2000", "-1000 + 2000"
]

response = co.embed(
            texts = expressions,
            model = "embed-v4.0",
            input_type="classification",
            embedding_types=["float"],
            output_dimension=256
        )
embeddings = np.array(response.embeddings.float_)

embeddings = normalize(embeddings)

# Reduce dimensions using UMAP
reducer = UMAP(n_components=2, metric='cosine')  

# Reduce dimensions
projected_embeddings = reducer.fit_transform(embeddings)

plt.figure(figsize=(10, 8))
sns.set_style("whitegrid")

# Plot each point with expression as label
scatter = plt.scatter(
    projected_embeddings[:, 0],
    projected_embeddings[:, 1],
    s=100,
    alpha=0.7,
    cmap="viridis"
)

for i, expr in enumerate(expressions):
    plt.annotate(
        expr,
        (projected_embeddings[i, 0], projected_embeddings[i, 1]),
        textcoords="offset points",
        xytext=(0, 5),
        ha='center',
        fontsize=9
    )

# Add title and adjust layout
plt.title("UMAP Projection of Arithmetic Expression Embeddings", pad=20)
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.tight_layout()

# Save and show
plt.savefig("arithmetic_embeddings.png", dpi=300)
plt.show()