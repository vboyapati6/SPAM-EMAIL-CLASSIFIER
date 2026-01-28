# Spam Email Classifier

A machine learning project comparing **LSTM** and **BERT Transformer** approaches for spam email detection.

## Results

| Model | Test Accuracy | Parameters | Training Time |
|-------|---------------|------------|---------------|
| LSTM  | ~96%          | ~50K       | Fast          |
| BERT  | ~98%          | ~109M      | Slower (GPU)  |

## Project Structure

```
├── spam_email_detection.ipynb  # Main notebook with both models
├── spam_ham_dataset.csv        # Email dataset (ham/spam labeled)
└── README.md
```

## Models

### 1. LSTM Approach
- Text preprocessing (lowercase, stopword & punctuation removal)
- Word embeddings → LSTM → Dense layers
- Lightweight, suitable for production

### 2. BERT Transformer Approach
- Pre-trained `bert-base-uncased` with fine-tuning
- No manual preprocessing required
- State-of-the-art performance

## Quick Start

```bash
# Install dependencies
pip install tensorflow torch transformers pandas numpy matplotlib seaborn scikit-learn nltk tqdm

# Run the notebook
jupyter notebook spam_email_detection.ipynb
```

## Dataset

The dataset contains **5,171 emails** labeled as:
- `ham` - Legitimate emails
- `spam` - Spam emails

Balanced to **2,998 samples** (1,499 each) for training.

## Requirements

- Python 3.8+
- TensorFlow 2.x
- PyTorch
- Transformers (Hugging Face)
- CUDA (optional, for GPU acceleration)
