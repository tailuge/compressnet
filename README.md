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

### 2. Permute a Layer
Applies a random permutation to a chosen hidden layer and fixes downstream weights.
```bash
python permute_model.py --layer 2
```
*Creates `modified_layer_2.pt`.*

### 3. Verify Equivalence
Compares the outputs of the master and permuted model on random inputs.
```bash
python test_compare.py --mutant modified_layer_2.pt
```
*Should print `PASS` with a max difference ≈ 0.*

## Project Structure

- `model.py`: TinyMLP architecture (8 layers, 16 -> 32 -> 8).
- `create_master.py`: Script to generate the base model.
- `permute_model.py`: Script to apply weight permutations.
- `test_compare.py`: Script to verify identical outputs.
- `pyproject.toml`: Configuration and dependencies (Torch, NumPy).

## License

MIT
