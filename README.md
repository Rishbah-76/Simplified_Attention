# Attention Mechanism Explained: A Deep Dive

This repository contains a pedagogical implementation of attention mechanisms in deep learning. The Jupyter notebook `Simplified_Attention.ipynb` walks through the fundamental concepts of attention, from word embeddings to context vector generation.

## Problem Addressed

Understanding attention mechanisms is crucial for modern NLP and sequence-to-sequence models. The notebook breaks down this complex concept by visualizing:

1. How word embeddings represent meaning in a vector space
2. How similarity between word vectors can be calculated using dot products
3. How attention weights are computed from these similarity scores
4. How context vectors are generated from attention-weighted combinations of inputs

## Implementation Approach

The notebook takes a step-by-step approach:

1. **Word Embeddings**: Starts with simple 3D vector representations of words
2. **Query-Key Matching**: Demonstrates how a query vector can be compared with key vectors using dot products
3. **Attention Weight Computation**: Shows the normalization of similarity scores using both naive and softmax approaches
4. **Context Vector Generation**: Illustrates how attention weights are used to create context-sensitive representations

## Key Technical Details

- **Dot Product Calculation**: The implementation shows both manual calculation and PyTorch's matrix multiplication
- **Softmax Implementation**: Includes both a naive version and the numerically stable version
- **Matrix-based Computation**: Demonstrates how to efficiently compute attention for multiple queries in parallel
- **Visualization**: Uses 3D plots to visualize word embeddings in vector space

## Limitations

The notebook concludes with an important insight about simple non-trainable attention: 

Without trainable weights, attention can only rely on semantic similarity in the embedding space. With trainable weights (as in modern transformer models), the model can learn more complex relationships that aren't captured by simple dot products alone.

This repository serves as a foundational building block for understanding more complex implementations in frameworks like PyTorch and TensorFlow. # Simplified_Attention
