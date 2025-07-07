# 📩 Spam Classification using GPT-2 (124M)

This project demonstrates the fine-tuning of a pre-trained GPT-2 model (124M parameters) for binary SMS spam classification using the SMS Spam Collection dataset from the UCI Machine Learning Repository.

## 🚀 Overview

- Fine-tunes GPT-2 with a classification head on top
- Uses balanced SMS spam dataset
- Implements custom tokenization, padding, and data loading
- Achieves high accuracy with minimal compute using PyTorch

## 🧠 Model Architecture

- Base: GPT-2 (124M parameters)
- Layers: 12 Transformer blocks, 768-dimensional embeddings
- Fine-tuned only the last transformer block + final linear classification head
- Classification Head: Fully-connected layer mapping embeddings to 2 output classes

## 📊 Dataset

- 📁 Source: [UCI SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
- 📌 Labels: `spam` or `ham`
- Balanced for training using undersampling

## 🛠️ Technologies Used

- Python 3.10+
- PyTorch
- NumPy, Pandas
- Custom BPE tokenizer (no Hugging Face used)
- Pretrained weights: OpenAI GPT-2 (124M)
