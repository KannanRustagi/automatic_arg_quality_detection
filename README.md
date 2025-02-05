# Automatic Argument Quality Prediction

- This repository contains a Python implementation for automatic argument quality prediction using neural models based on BERT (Bidirectional Encoder Representations from Transformers). 
---

## Methods Overview

### Argument-Pair Classification
- Fine-tunes the BERT base model for binary classification of argument pairs.
- The input format follows `[CLS]argA[SEP]argB`, and the `[CLS]` token is used to extract the contextual embedding for the entire input.
- The fine-tuned classifier predicts which argument in the pair is of higher quality.

### Individual Argument Ranking
- Leverages the embeddings from the fine-tuned argument classifier.
- Concatenates the last four layers of BERT to form an embedding of size 3072.
- Passes the embedding through a two-layer neural network with ReLU activation and a sigmoid output to predict an argument's quality score.

---

## Pipeline Steps

### Data Preparation
- Load argument pairs and corresponding labels from TSV files.
- Encode the pairs using BERT's tokenizer.
  
### Fine-tuning BERT for Argument-Pair Classification
- Use cross-entropy loss to train the `ArgClassifier`.
- Evaluate accuracy and log training progress.

### Embedding Extraction for Argument Ranking
- Extract embeddings from the fine-tuned BERT model.
- Concatenate the last four hidden layers for each input argument.

### Training the Argument Ranker
- Train a custom ranker (`ArgRanker`) using the extracted embeddings.
- Use MSE loss for training to minimize the difference between predicted and actual scores.

### Evaluation Metrics
- Pearson and Spearman correlations are computed to assess the model's effectiveness in predicting argument quality.

---

## Usage

### Training the Classifier
```bash
python automatic_arg_quality.py
```
Ensure that the paths to the training data (`arg_pairs.tsv` and `arg.tsv`) are correctly set in the code.

### Saving and Loading Models
- Models are saved in `.pth` format after training.
- Fine-tuned models can be reloaded using PyTorch's `torch.load()` method.

### Evaluation
- Predictions are saved to a CSV file (`predictions_results.csv`) containing both predicted and actual scores.
- Pearson and Spearman correlations are printed to assess model performance.

---

## Requirements

To install the required packages, use:
```bash
pip install -r requirements.txt
```

---

