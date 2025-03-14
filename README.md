# Code-to-Comment Generation Using Transformer and AST Tokenization

## Overview
This project aims to generate natural language comments for Java methods using deep learning techniques. By leveraging a **Transformer-based model**, the system translates source code into descriptive comments, enhancing code documentation automatically.

## Key Features
- **Multi-Input Training**: The model is trained using **source code, abstract syntax trees (ASTs), and comments** to capture structural and semantic relationships.
- **Transformer Architecture**: Uses a **6-layer encoder-decoder** model with self-attention mechanisms.
- **Custom Tokenization**: Separate tokenizers are trained for code, ASTs, and comments, improving understanding and reducing vocabulary ambiguity.
- **Structure-Based Traversal (SBT)**: Enhances AST representation to retain structural integrity.
- **BLEU Score Evaluation**: Measures output quality against human-written comments.

## Model Architecture
The system is built using the **Transformer** architecture:
- **Embedding Layers**: Separate embeddings for code, ASTs, and comments.
- **Multi-Head Attention**: Captures dependencies between tokens.
- **Feed-Forward Layers**: Processes representations efficiently.
- **Dropout and Normalization**: Improves generalization.

## Installation
### Prerequisites
- Python 3.8+
- PyTorch
- NumPy
- TensorFlow (optional, for BLEU score evaluation)
- Jupyter Notebook (optional, for experiments)

### Setup
Clone the repository:
```bash
git clone https://github.com/yourusername/code-comment-generator.git
cd code-comment-generator
