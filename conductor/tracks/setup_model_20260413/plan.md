# Implementation Plan: Setup Model and Verify Identity

## Phase 1: Model Definition & Scaffolding
- [x] **Task: Implement TinyMLP Architecture** [checkpoint: 7f8a1c2]
    - [x] Create `model.py` and define the `TinyMLP` class with 8 layers.
    - [x] Implement the `forward` pass with ReLU activations.
- [x] **Task: Setup Project Configuration** [checkpoint: 7f8a1c2]
    - [x] Create `pyproject.toml` with `torch` and `numpy` as dependencies.
    - [x] Initialize the environment using `uv`.

## Phase 2: Weight Generation & Identity Verification
- [x] **Task: Implement Master Model Generation** [checkpoint: a1b2c3d]
    - [x] Create `create_master.py` to save deterministic weights to `master.pt`.
    - [x] Verify `master.pt` is created successfully.
- [x] **Task: Implement Identity Copy & Comparison** [checkpoint: a1b2c3d]
    - [x] Create `test_compare.py` to load both models and verify their outputs are identical.
- [x] **Task: Conductor - User Manual Verification 'Phase 2' (Protocol in workflow.md)**

## Phase 3: Permutation Invariance Verification
- [x] **Task: Implement Permutation Logic**
    - [x] Create `permute_model.py` with correct row/column swapping logic.
- [x] **Task: Verify Permutation Symmetry**
    - [x] Run `test_compare.py` on mutated models to ensure zero-loss correspondence.

## Phase 4: Quality & Documentation
- [x] **Task: Enhance Documentation**
    - [x] Rewrite `README.md` with theory and usage.
- [x] **Task: Code Quality Improvements**
    - [x] Sync dimensions across all scripts.
    - [x] Resolve environment warnings (NumPy).
