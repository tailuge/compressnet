# Technology Stack: Tiny MLP Permutation Invariance

## Overview
The project is built using a modern, efficient, and well-supported Python-based machine learning stack. The choices prioritize technical accuracy, ease of setup, and reproducible results.

## Core Language & Runtime
- **Python 3.10+**: Provides modern language features and robust support for the selected machine learning libraries.
- **uv**: A fast and reliable Python package manager used for virtual environment management and dependency installation, ensuring reproducible development environments.

## Machine Learning Framework
- **PyTorch**: The primary library for defining the neural network architecture, managing model weights, and performing tensor operations. Its intuitive API is well-suited for the manual weight manipulation required by this project.

## Weight Storage & Management
- **PyTorch State Dictionaries (.pt)**: A standard, efficient format for saving and loading model parameters. This ensures that weights are preserved with full precision during the permutation process.

## Verification & Testing
- **Custom Verification Scripts**: A set of focused Python scripts for model creation, permutation, and empirical comparison.
- **Pytest (Optional/Recommended)**: Can be used to provide a more structured and automated testing framework for the core permutation logic.

## Project Structure (Target)
- `pyproject.toml`: Project configuration and dependency manifest.
- `model.py`: The `TinyMLP` architecture definition.
- `create_master.py`: Script to generate the base model.
- `permute_model.py`: Script to apply weight permutations.
- `test_compare.py`: Script to verify identical outputs.
