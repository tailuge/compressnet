# Project Setup: Quick Start

## Prerequisites
- Python 3.10+
- `uv` package manager

## Setup Instructions

### Step 1: Initialize the Environment
```bash
uv venv
source .venv/bin/activate
```

### Step 2: Install Dependencies
```bash
uv pip install torch
```

### Step 3: Create the Master Model
Run the create script to generate deterministic weights:
```bash
python create_master.py
```

This will:
- Initialize the `TinyMLP` architecture (8 layers with ReLU activations)
- Set fixed random seeds for deterministic reproducibility
- Save the trained weights to `master.pt`

### Step 4: Verify Output
After running the script, confirm that:
- `master.pt` file exists in the project root
- Console output shows successful weight generation and saving

## Next Steps
Once the master model is created, you can proceed with:
1. Copy `master.pt` to `mut1.pt` for identity verification
2. Run `python test_compare.py` to verify identical outputs
3. Explore permutation invariance with `permute_model.py`

## Project Structure
```
compressnet/
├── model.py              # TinyMLP architecture definition
├── create_master.py      # Master model generation script
├── permute_model.py      # Weight permutation script
├── test_compare.py       # Output comparison/verification
├── master.pt             # Generated master weights
├── pyproject.toml        # Project configuration
└── conductor/            # Project documentation and planning
```

## Tech Stack
- **Python 3.10+**
- **PyTorch** - Neural network framework
- **uv** - Package and environment management
