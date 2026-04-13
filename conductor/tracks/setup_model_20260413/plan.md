# Implementation Plan: Setup Model and Verify Identity

## Phase 1: Model Definition & Scaffolding
- [ ] **Task: Implement TinyMLP Architecture**
    - [ ] Create `model.py` and define the `TinyMLP` class with 8 layers.
    - [ ] Implement the `forward` pass with ReLU activations.
- [ ] **Task: Setup Project Configuration**
    - [ ] Create `pyproject.toml` with `torch` as a dependency.
    - [ ] Initialize the environment using `uv`.

## Phase 2: Weight Generation & Identity Verification
- [ ] **Task: Implement Master Model Generation**
    - [ ] Create `create_master.py` to save deterministic weights to `master.pt`.
    - [ ] Verify `master.pt` is created successfully.
- [ ] **Task: Implement Identity Copy & Comparison**
    - [ ] Create a shell script or manual step to copy `master.pt` to `mut1.pt`.
    - [ ] Create `test_compare.py` to load both models and verify their outputs are identical.
- [ ] **Task: Conductor - User Manual Verification 'Phase 2' (Protocol in workflow.md)**
