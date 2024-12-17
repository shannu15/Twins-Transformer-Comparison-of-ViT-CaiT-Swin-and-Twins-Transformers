---

# Vision Transformers: Comparing Twins, ViT, CaiT, and Swin

This project implements and compares four Vision Transformer (ViT) architectures—**Twins**, **ViT**, **CaiT**, and **Swin**—to analyze their performance on CIFAR-10 and CIFAR-100 datasets. The study builds upon transformer frameworks and highlights the strengths of locally-grouped and global sub-sampled attention mechanisms.

## 📚 Overview

The Twins Vision Transformer improves on traditional ViT approaches by combining:
1. **Locally-Grouped Self-Attention (LSA)**: Captures fine-grained, short-distance image features.
2. **Global Sub-Sampled Attention (GSA)**: Extracts long-distance global context.

This dual-attention mechanism makes Twins efficient and scalable for dense prediction tasks like image segmentation and object detection. The project also benchmarks Twins against popular models like **ViT**, **CaiT**, and **Swin**.

---

## 🛠️ Features

1. **Twins Vision Transformer**:
   - Combines LSA and GSA for optimal local and global feature extraction.
   - Includes positional encoding for enhanced spatial relationships.

2. **Comparative Analysis**:
   - Implements and evaluates Twins, ViT, CaiT, and Swin transformers on identical datasets and conditions.
   - Uses CIFAR-10 and CIFAR-100 for fair comparison.

3. **Framework**:
   - Built on PyTorch with modular support for adding and benchmarking new architectures.

---

## 🗂️ Project Structure

```
VisionTransformers/
├── data/                     # Dataset storage
├── models/                   # Model implementations
│   ├── twins.py              # Twins Vision Transformer
│   ├── vit.py                # Vision Transformer
│   ├── cait.py               # CaiT Transformer
│   ├── swin.py               # Swin Transformer
├── checkpoint/               # Model checkpoints
├── Utils.py                  # Data preprocessing and utilities
├── VisionTransformers_main.py # Main training and evaluation script
```

---

## ⚙️ Setup and Installation

1. **Clone the Repository**:
   ```bash
   cd VisionTransformers
   ```

2. **Install Dependencies**:
   ```bash
   pip install torch torchvision timm einops
   ```

3. **Download CIFAR-10/CIFAR-100 Dataset**:
   - The dataset is automatically downloaded when the script runs.

---

## 🚀 How to Run

1. **Train a Specific Model**:
   Use the `--net` argument to select the desired transformer:
   ```bash
   python VisionTransformers_main.py --net twins --dataset CIFAR10 --bs 64 --n_epochs 200
   ```

   Options for `--net`:
   - `vit` (Vision Transformer)
   - `cait` (CaiT Transformer)
   - `swin` (Swin Transformer)
   - `twins` (Twins Transformer)

2. **Adjust Parameters**:
   Modify parameters like `batch_size`, `learning_rate`, and `patch_size`:
   ```bash
   python VisionTransformers_main.py --patch 4 --lr 1e-4
   ```

---

## 🧪 Experiments and Results

1. **Twins Transformer**:
   - Achieves ~85.3% accuracy on CIFAR-10 after 200 epochs.
   - Efficiently balances local and global feature extraction using LSA and GSA.

2. **Comparative Analysis**:
   - **ViT**: Performs well but is computationally expensive due to quadratic complexity.
   - **CaiT**: Improved attention mechanisms enhance robustness.
   - **Swin**: Uses non-overlapping windows but struggles with global context.
   - **Twins**: Combines local and global attention mechanisms, outperforming others in efficiency and accuracy.

---

## ✨ Key Insights

- **Twins Transformer** balances computational efficiency with prediction accuracy.
- **LSA and GSA** together improve performance on dense prediction tasks.
- **Benchmarking** under consistent conditions ensures a fair comparison of architectures.

---
