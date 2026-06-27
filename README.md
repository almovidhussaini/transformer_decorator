
# Transformer Decoder from Scratch

This project implements a **decoder-only Transformer** from scratch using PyTorch. The implementation focuses on understanding the core building blocks of modern Large Language Models (LLMs) by developing each component manually instead of relying on high-level libraries.

## Features

### 📥 Data Downloading & Preprocessing

Downloaded text data from the Project Gutenberg dataset, trained a custom Byte Pair Encoding (BPE) tokenizer, and prepared the dataset for next-token prediction using PyTorch's Dataset and DataLoader classes.

### 🔤 Rotary Positional Encoding (RoPE)

Implemented Rotary Positional Encoding from scratch to inject positional information into query and key vectors without using traditional positional embeddings.

### 🎯 Grouped Query Attention (GQA)

Implemented Grouped Query Attention to improve memory efficiency by sharing Key and Value heads across multiple Query heads while maintaining strong attention performance.

### 🧠 Mixture of Experts (MoE)

Built a Mixture of Experts feed-forward layer using multiple SwiGLU expert networks and a learnable router that dynamically selects the top-k experts for each input token.

### 📏 RMS Normalization & Skip Connections

Implemented RMSNorm for stable training and residual (skip) connections to improve gradient flow and enable training of deeper transformer networks.

### 🏗️ Complete Transformer Decoder

Constructed a complete decoder-only Transformer architecture by combining token embeddings, RoPE, Grouped Query Attention, Mixture of Experts, RMSNorm, residual connections, and output projection layers.

### 🚀 Model Training

Implemented the complete training pipeline including causal masking, cross-entropy loss, AdamW optimizer, learning rate scheduling with warmup and cosine annealing, gradient clipping, checkpoint saving, and GPU support.

### ✨ Text Generation & Testing

Evaluated the trained model using prompt-based text generation to verify the decoder's ability to perform autoregressive next-token prediction.

## Technologies Used

* Python
* PyTorch
* Hugging Face Tokenizers
* Google Colab
* Project Gutenberg Dataset

## Learning Objectives

This project was developed as an educational implementation to gain a deep understanding of how modern decoder-only Large Language Models are built and trained. Every major component was implemented from scratch to better understand the underlying mathematics and engineering behind state-of-the-art LLMs.
