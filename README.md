# Assessing Information Loss of Semantic Embedding Models with SFT

**→** A controlled experiment testing whether Cohere's semantic embeddings enable arithmetic reasoning when augmented with lightweight transformer layers. **Implemented from scratch in PyTorch with both SFT training.**

## Key Question

Can frozen semantic embeddings (trained for similarity tasks) serve as useful inputs to solve math questions when combined with:

1. A basic neural network trained to minimize MSE?
2. A small transformer decoder trained via **SFT**?

I specifically:

- Use **precomputed Cohere's `embed-v4.0` embeddings** of math questions
- Train a neural network for regression as a baseline
- Train a Transformer decoder to generate answers digit-by-digit
- Evaluate the decoder’s capacity to “reason” from dense input embeddings

Both SFT and PPO training methods are written **from scratch**.

## Motivation

I am exploring how pre-trained semantic embeddings (like those from Cohere) can be leveraged to fine-tune a language model for simple arithmetic reasoning tasks. My goal is to understand whether these embeddings, usually trained for similarity or classification tasks, contain enough contextual information to guide fine-tuning for such a task.

![Model Diagram](./arithmetic_embeddings.png)

From this diagram, which plots embeddings of math operations,the embeddings evidently seem to be clustered based on the numbers being used, and are also separated based on the operator (add, subtract, etc.), which shows potential in training an external model to understand the embeddings for this task.

## Model Overview

- **Input:** A fixed-size embedding (e.g., 256-dim vector) representing a math question (e.g. ` What is 9 + 10?`)
- **Output:** A sequence of digits representing the numerical answer + end token
- **Architecture:** Transformer decoder with token embeddings for digits `0–9`, producing results in an autoregressive fashion.

## Current Progress

Currently writing the training loops for SFT. Will soon be testing both models and releasing my results.

## Next Steps

- **More complicated Math Arithmetic**: Extend to multi-step problems (e.g., "(12 + 3) \* 4")
- **Extension to Other Tasks with Curriculum-Based Learning:** Generalize approach to other reasoning tasks or datasets, such as logical inference or multi-step word problems.
