# compressnet: Tiny MLP Permutation Invariance

A minimal, robust demonstration of neural network symmetry using Multilayer Perceptrons (MLPs).

[results](results.md)

## Overview

This project empirically verifies **permutation invariance** in neural networks. In an MLP, hidden neurons can be reordered at any layer without changing the model's output, provided that the corresponding weights in the downstream layer are permuted accordingly.

## Theory

For a hidden layer $i$ with activation $y_i = \sigma(W_i x + b_i)$, if we apply a permutation $P$ to the neurons:
$y_i' = P y_i = \sigma(P W_i x + P b_i)$

To preserve the input to the next layer $i+1$:
$W_{i+1} y_i = W_{i+1} P^{-1} y_i'$

Thus, we must adjust $W_{i+1}$ to $W_{i+1} P^{-1}$ (which, for a permutation matrix $P$, is $W_{i+1} P^T$). In code, this corresponds to reordering the columns of $W_{i+1}$ using the same permutation used for the rows of $W_i$.

## Canonicalization & Sorting

While any permutation of hidden neurons maintains output invariance, we can use this symmetry to "canonicalize" a model by sorting neurons into a standard order.

### The Sorting Constraint
Neural network weights are coupled in "blocks" (neurons). You can reorder the blocks (rows of $W_i$), but you cannot independently sort every individual weight entry without changing the model's output.

### Strategies
- **Bias Sorting**: Sorting neurons by their bias value ($b_i$) creates a perfectly ascending "staircase" in the bias plot and groups associated weights.
- **2D Sorting**: By permuting Layer $N-1$, you sort the **columns** of Layer $N$. By permuting Layer $N$, you sort the **rows** of Layer $N$. Propagating this forward allows you to "smooth" the weight matrices across the entire network.

### Row-Major Data Layout
The visualization in `results.md` represents the flattened weight matrix in **row-major order**. For a layer with $M$ neurons and $N$ inputs:
- The first $N$ indices on the X-axis (0–15) are the weights of **Neuron 0**.
- The next $N$ indices (16–31) are **Neuron 1**, and so on.

### The "Family" Constraint
Permutation invariance allows us to reorder entire **rows** (neurons) and entire **columns** (input features). 
- **Row Permutation**: Moves 16-element "blocks" as atomic units.
- **Column Permutation**: Reorders the 16 elements *inside every block* simultaneously (i.e., you must use the same index order for all neurons).

Because every neuron block must share the same internal column order, you cannot achieve a perfectly monotonic "flat" list while maintaining output invariance. **2D Sorting** is the physical limit of how "ordered" the weights can appear.

## Installation

This project uses `uv` for fast, reproducible dependency management.

```bash
# Initialize virtual environment and install dependencies
uv venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
uv pip install -e .
```

## Usage

Follow these steps to verify permutation invariance:

### 1. Create Master Model
Generates a deterministic base model with fixed seeds.
```bash
python create_master.py
```
*Creates `master.pt`.*

### 2. Permute and Sort
Applies bias-based sorting to hidden layers. You can permute a single layer or the entire model.
```bash
# Permute all hidden layers (1-6) using weight-mean sorting
python permute_model.py --all

# Apply 2D sorting (rows AND columns) to a specific layer
python permute_model.py --layer-2d 4

# Or permute a specific layer (1D)
python permute_model.py --layer 2
```
*Creates `modified_all.pt`, `modified_layer_4_2d.pt`, or `modified_layer_2.pt`.*

### 3. Verify Equivalence
```bash
python test_compare.py --mutant modified_all.pt
```
*Should print `PASS` with a max difference ≈ 0.*

### 4. Visualize Results
Generates high-density dual-axis plots of weights and biases.
```bash
python publish.py --mutant modified_all.pt
```
*Updates `results.md` with the new plots.*

## Project Structure

- `model.py`: TinyMLP architecture (8 layers, 16 -> 32 -> 8).
- `create_master.py`: Script to generate the base model.
- `permute_model.py`: Script to apply weight permutations.
- `test_compare.py`: Script to verify identical outputs.
- `pyproject.toml`: Configuration and dependencies (Torch, NumPy).

## License

MIT
