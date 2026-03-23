# Neural Network and Representation Learning on MNIST

## Overview
This project implements multiple machine learning and neural network models **from scratch using NumPy** to explore classification, representation learning, and unsupervised learning on the MNIST dataset.

The goal is to understand how different learning approaches affect model performance, feature representation, and generalization.

---

## Key Features
- Built a **Multilayer Perceptron (MLP)** classifier from scratch
- Implemented an **Autoencoder** for representation learning
- Developed a **Self-Organizing Feature Map (SOFM)** for unsupervised learning
- Compared:
  - Baseline classifier
  - Autoencoder-based models (frozen vs fine-tuned)
- Evaluated models using:
  - Training/test error curves
  - Confusion matrices
  - Reconstruction loss
  - Feature similarity (cosine similarity)

---

## Models Implemented

### 1. MLP Classifier (From Scratch)
- Architecture: 784 → 150 → 10
- Activation: Sigmoid
- Optimization:
  - Stochastic Gradient Descent (SGD)
  - Momentum (0.9)
- Initialization: Xavier/Glorot

**Performance:**
- Training error: < 5%
- Test error: ~7–8%

---

### 2. Autoencoder
- Architecture: 784 → 150 → 784
- Purpose: Learn compressed latent representations of digit images

**Insights:**
- Successfully learned meaningful feature representations
- Smooth convergence of reconstruction loss
- Some redundancy observed in hidden features

---

### 3. Autoencoder-Based Classifier

#### Case I — Frozen Features
- Hidden layer fixed after pretraining
- Only output layer trained

**Result:**
- Poor classification performance due to non-discriminative features

#### Case II — Fine-Tuned Features
- Entire network trained end-to-end

**Result:**
- Significant improvement in classification accuracy

---

### 4. Self-Organizing Feature Map (SOFM)
- Grid size: 12 × 12 neurons
- Learning: Competitive learning with neighborhood updates

**Results:**
- Captured structural relationships between digit classes
- Similar digits mapped close together
- Useful for visualization but less effective for classification

---

### 5. SOFM-Based Classifier
- Used winning neuron representation for classification

**Result:**
- Lower accuracy (~16% test error)
- Information loss due to discrete representation

---

## Analysis & Insights
- Fine-tuning learned representations significantly improves model performance
- Autoencoders capture general features but require tuning for classification tasks
- SOFM is effective for visualization and clustering but not for high-accuracy classification
- Feature similarity analysis revealed both redundancy and independence in learned representations

---

## Tech Stack
- Python
- NumPy
- Matplotlib

---

## Repository Structure
mnist-mlp-autoencoder/
│
├── src/ # Model implementations
│ ├── mlp_classifier.py
│ └── autoencoder.py
│
├── data/ # Dataset files
│ ├── MNISTnumImages5000_balanced.txt
│ └── MNISTnumLabels5000_balanced.txt
│
├── outputs/ # Model artifacts and results
│ └── autoencoder_artifacts.npz

---

## How to Run

1. Clone the repository:
```bash
git clone https://github.com/abhiisap/mnist-mlp-autoencoder.git
cd mnist-mlp-autoencoder

2. Run MLP classifier:

python src/mlp_classifier.py

3. Run autoencoder:

python src/autoencoder.py
