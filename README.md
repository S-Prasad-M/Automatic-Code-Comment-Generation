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

### Transformer Model Details:
Transformer(
  (enc_emb): TransformerEmbedding(
    (embed): Embedding(445812, 256)
    (pos_enc): PositionalEncoding()
    (drop): Dropout(p=0.1, inplace=False)
  )
  (dec_emb): TransformerEmbedding(
    (embed): Embedding(445812, 256)
    (pos_enc): PositionalEncoding()
    (drop): Dropout(p=0.0, inplace=False)
  )
  (encoder): ModuleList(
    (0-5): 6 x EncoderBlock(
      (mha): MultiHeadAttention(
        (q_wgt): Linear(in_features=256, out_features=256, bias=True)
        (k_wgt): Linear(in_features=256, out_features=256, bias=True)
        (v_wgt): Linear(in_features=256, out_features=256, bias=True)
        (out): Linear(in_features=256, out_features=256, bias=True)
        (drop_att): Dropout(p=0.1, inplace=False)
        (drop_res): Dropout(p=0.1, inplace=False)
        (ln): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
      )
      (ff): Sequential(
        (0): Linear(in_features=256, out_features=1024, bias=True)
        (1): ReLU()
        (2): Dropout(p=0.1, inplace=False)
        (3): Linear(in_features=1024, out_features=256, bias=True)
        (4): Dropout(p=0.1, inplace=False)
        (5): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
      )
    )
  )
  (decoder): ModuleList(
    (0-5): 6 x DecoderBlock(
      (mha1): MultiHeadAttention(
        (q_wgt): Linear(in_features=256, out_features=256, bias=True)
        (k_wgt): Linear(in_features=256, out_features=256, bias=True)
        (v_wgt): Linear(in_features=256, out_features=256, bias=True)
        (out): Linear(in_features=256, out_features=256, bias=True)
        (drop_att): Dropout(p=0.1, inplace=False)
        (drop_res): Dropout(p=0.1, inplace=False)
        (ln): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
      )
      (mha2): MultiHeadAttention(
        (q_wgt): Linear(in_features=256, out_features=256, bias=True)
        (k_wgt): Linear(in_features=256, out_features=256, bias=True)
        (v_wgt): Linear(in_features=256, out_features=256, bias=True)
        (out): Linear(in_features=256, out_features=256, bias=True)
        (drop_att): Dropout(p=0.1, inplace=False)
        (drop_res): Dropout(p=0.1, inplace=False)
        (ln): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
      )
      (ff): Sequential(
        (0): Linear(in_features=256, out_features=1024, bias=True)
        (1): ReLU()
        (2): Dropout(p=0.1, inplace=False)
        (3): Linear(in_features=1024, out_features=256, bias=True)
        (4): Dropout(p=0.1, inplace=False)
        (5): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
      )
    )
  )
  (out): Linear(in_features=256, out_features=445812, bias=True)
)


## Training with Separate Tokenizers
Training separate tokenizers for **code, comments, and ASTs** has significantly improved the understanding of relationships within the dataset. The specialized tokenizers capture **syntax, semantics, and structure more effectively**, leading to better generalization during model training. Code tokenization helps in handling keywords and identifiers, comment tokenization enhances natural language fluency, and AST tokenization preserves structural information.

## Readability Score
The generated comments were evaluated for readability and achieved the following scores:
- **Flesch Reading Ease Score**: **56.67**
- **Flesch-Kincaid Grade Level**: **7.59**

These scores indicate that the comments are moderately easy to read and suitable for a general audience with basic programming knowledge.

## Evaluation Comparison
\begin{table}[h]
    \centering
    \begin{tabular}{|c|c|}
        \hline
        \textbf{Approach} & \textbf{BLEU-4 Score (\%)} \\
        \hline
        CODE-NN & 25.30 \\
        Seq2Seq & 34.87 \\
        Attention-based Seq2Seq & 35.50 \\
        DeepCom & 38.17 \\
        Our Method & 46.81 \\
        \hline
    \end{tabular}
    \caption{Comparison of BLEU-4 Scores Across Different Models}
    \label{tab:bleu_scores}
\end{table}


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
