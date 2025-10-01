# Kannada-Character-Recognition-using-Swin-Transformer
Deep learning model for recognizing 587 handwritten Kannada character classes using Swin Transformer architecture with transfer learning.
## Overview

This project implements a Swin Transformer-based character recognition system for Kannada script, achieving 99.80% validation accuracy. The model uses pretrained weights from ImageNet and fine-tunes on a dataset of 146,556 handwritten Kannada character images.

## Results

- **Validation Accuracy**: 99.80%
- **Precision/Recall/F1-Score**: 0.998
- **Error Rate**: 0.20% (60 errors out of 29,312 samples)
- **Training Time**: ~2 hours (5 epochs on GPU)
- **Model Size**: 27.9M parameters

## Dataset

- **Source**: [Handwritten Kannada Characters Dataset](https://www.kaggle.com/datasets/sahilkumarjamwal/handwritten-kannada-main-aksharas)
- **Total Images**: 146,556
- **Classes**: 587 Kannada characters
- **Split**: 80% training (117,244), 20% validation (29,312)
- **Split Strategy**: Stratified sampling ensuring all classes in both sets
- **Image Size**: 224x224 (resized)

## Model Architecture

- **Base Model**: Swin Transformer Tiny (`swin_tiny_patch4_window7_224`)
- **Pretrained**: ImageNet-1K weights
- **Framework**: PyTorch with timm library
- **Input Resolution**: 224x224x3
- **Output**: 587 classes (softmax)

## Training Configuration
```python
Optimizer: AdamW
Learning Rate: 1e-4 (initial)
Weight Decay: 0.05
Scheduler: CosineAnnealingLR
Batch Size: 32
Epochs: 5
Loss Function: CrossEntropyLoss
Device: CUDA (GPU)
