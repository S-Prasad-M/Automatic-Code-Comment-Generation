# Code-to-Comment Generation Using Transformer and AST Tokenization

## Overview
This project aims to generate natural language comments for Java methods using deep learning techniques. By leveraging a **Transformer-based model**, the system translates source code into descriptive comments, enhancing code documentation automatically.

## Key Features
- **Multi-Input Training**: The model is trained using **source code, abstract syntax trees (ASTs), and comments** to capture structural and semantic relationships.
- **Transformer Architecture**: Uses a **6-layer encoder-decoder** model with self-attention mechanisms.
- **Custom Tokenization**: Separate tokenizers are trained for code, ASTs, and comments, improving understanding and reducing vocabulary ambiguity.
- **Structure-Based Traversal (SBT)**: Enhances AST representation to retain structural integrity.
- **BLEU Score Evaluation**: Measures output quality against human-written comments.

## Web-Scraped Data
The data was scraped using the GitHub REST API and corresponding code is added in the repositary too.
The data can be accessed through this [Data Link](https://drive.google.com/drive/folders/19OIrEZihBfvMv8HhzR8Rldl_sEzs_OJ-)

## Model Architecture
The system is built using the **Transformer** architecture:
- **Embedding Layers**: Separate embeddings for code, ASTs, and comments.
- **Multi-Head Attention**: Captures dependencies between tokens.
- **Feed-Forward Layers**: Processes representations efficiently.
- **Dropout and Normalization**: Improves generalization.

### Transformer Model Details:
* **`TransformerEmbedding`**:
    * Embeddings for input and output tokens (`Embedding`).
    * Positional encoding to capture sequence order (`PositionalEncoding`).
    * Dropout for regularization (`Dropout`).
* **`EncoderBlock`**:
    * Multi-head attention mechanism (`MultiHeadAttention`).
    * Feed-forward network (`Sequential`).
    * Layer normalization (`LayerNorm`).
* **`DecoderBlock`**:
    * Two multi-head attention mechanisms (`MultiHeadAttention`).
    * Feed-forward network (`Sequential`).
    * Layer normalization (`LayerNorm`).
* **`MultiHeadAttention`**:
    * Linear layers for query, key, and value transformations (`Linear`).
    * Output linear layer (`Linear`).
    * Dropout for attention and residual connections (`Dropout`).
    * Layer normalization (`LayerNorm`).
* **`Feed-Forward Network (ff)`**:
    * Two linear layers with a ReLU activation in between (`Linear`, `ReLU`).
    * Dropout for regularization (`Dropout`).
    * Layer normalization (`LayerNorm`).
* **`Linear(out)`**:
    * Final linear layer to project the decoder output to the vocabulary size.
* **Key Parameters:**
    * `Embedding(445812, 256)`: Vocabulary size of 445812, embedding dimension of 256.
    * 6 Encoder and Decoder blocks.
    * Intermediate feed forward layer size of 1024.
    * Dropout probability of 0.1 for most dropout layers, except the decoder embedding dropout which is 0.0.


## Training with Separate Tokenizers
Training separate tokenizers for **code, comments, and ASTs** has significantly improved the understanding of relationships within the dataset. The specialized tokenizers capture **syntax, semantics, and structure more effectively**, leading to better generalization during model training. Code tokenization helps in handling keywords and identifiers, comment tokenization enhances natural language fluency, and AST tokenization preserves structural information.

## Readability Score
The generated comments were evaluated for readability and achieved the following scores:
- **Flesch Reading Ease Score**: **56.67**
- **Flesch-Kincaid Grade Level**: **7.59**

These scores indicate that the comments are moderately easy to read and suitable for a general audience with basic programming knowledge.

## Evaluation Comparison
## Comparison of BLEU-4 Scores Across Different Models

| Approach                | BLEU-4 Score (%) |
|-------------------------|------------------|
| CODE-NN                 | 25.30            |
| Seq2Seq                 | 34.87            |
| Attention-based Seq2Seq | 35.50            |
| DeepCom                 | 38.17            |
| Our Method              | 46.81            |


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
