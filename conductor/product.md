# Initial Concept
I want to implement a minimal project specified in spec.md after refining it

# Product Guide: Tiny MLP Permutation Invariance

## Overview
This project focuses on implementing a minimal, robust demonstration of neural network symmetry, specifically **permutation invariance in Multilayer Perceptrons (MLPs)**. The core goal is to build a standard neural network and empirically verify that permuting hidden neurons—when accompanied by corresponding adjustments to downstream weights—preserves the model's output.

## Target Audience
- **AI Beginners/Educators**: Providing a clear, documented example of fundamental neural network properties.
- **Machine Learning Engineers**: Offering a tool for testing and debugging weight transformations and architectural symmetries.

## Core Goals
- **Core Verification**: Implementing the precise logic required to permute layers 0–6 of an 8-layer MLP while maintaining output equivalence.
- **Deterministic Reproducibility**: Ensuring that the process can be reliably reproduced across different runs using fixed seeds.

## Key Features
- **Base Implementation**: A set of three focused Python scripts (`create_master.py`, `permute_model.py`, `test_compare.py`) that handle model creation, weight transformation, and verification.
- **Improved CLI Options**: An enhanced command-line interface providing clear options for layer selection and detailed output for verification results.

## Success Criteria
- Successful creation and saving of a deterministic master model (`master.pt`).
- Correct application of weight permutations to a chosen hidden layer.
- Empirical validation showing that the permuted model produces identical outputs to the master model (max difference ≈ 0).
