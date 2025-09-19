# Neural Network Regression (NumPy from Scratch)

This project implements a fully connected neural network from scratch using only NumPy, trained on a noisy synthetic regression dataset.  
The goal is to demonstrate how forward pass, backward pass, and parameter updates work without relying on deep learning libraries like TensorFlow or PyTorch.

---

## 📂 Project Structure
```
Task 1/
│── model.py       # Core implementation: Dense layer, activations, loss, optimizer
│── train.ipynb    # Notebook with data generation, training loop, plots
│── README.md      # Project description, setup, and results

```
---

## ⚙️ Features
- **Layers & Activations**: Dense, ReLU, Sigmoid  
- **Loss**: Mean Squared Error (MSE)  
- **Optimizer**: Stochastic Gradient Descent (SGD) with adjustable learning rate & mini-batch size  
- **Training**: Univariate noisy cubic regression  
- **Visualization**: Loss curve and predicted vs ground-truth plots  
- **Gradient Check**: Utility for debugging backpropagation  

---

## 🚀 Getting Started

### 1️⃣ Setup Environment
```bash
# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install numpy matplotlib jupyter
```

### 2️⃣ Run Notebook
```bash
jupyter notebook
```
- Open train.ipynb
- Run cells in order to generate data, train the network, and visualize results

## 📊 Results

- Loss curve decreases steadily, showing convergence.
- Predictions approximate the noisy cubic ground truth function.

## 🧩 Design Choices

- 2 hidden layers (16 + 8 neurons, ReLU activations) → chosen for non-linear representation power.
- MSE loss since this is regression.
- SGD with mini-batches to stabilize training compared to pure online updates.

## 🔍 Convergence Notes

- Larger learning rates (e.g., 0.1) may cause divergence.
- Very small rates (e.g., 0.0001) converge too slowly.
- Batch size between 16–64 works well.