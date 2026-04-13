# Track Specification: Setup Model and Verify Identity

## Overview
The goal of this track is to establish the fundamental neural network model (`TinyMLP`) and verify that we can reliably save, copy, and load its weights while maintaining identical outputs. This provides a baseline for the subsequent permutation invariance work.

## Functional Requirements
- **TinyMLP Architecture**: Implement an 8-layer MLP with ReLU activations (except the final layer).
- **Master Weight Generation**: Create a script (`create_master.py`) to generate and save a deterministic base model to `master.pt`.
- **Identity Copy**: Manually copy `master.pt` to `mut1.pt` to simulate a "transformation" that is actually an identity.
- **Output Comparison**: Create a script (`test_compare.py`) to load both `master.pt` and `mut1.pt` and verify they produce identical results.

## Success Criteria
- Successful implementation of the `TinyMLP` class.
- Generation of `master.pt`.
- Successful verification that `master.pt` and its copy `mut1.pt` produce identical outputs (max difference ≈ 0).
