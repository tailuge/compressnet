Here’s a **clean, minimal spec** for a 3-script Python project that:

1. creates a tiny standard model and saves “master” weights
2. loads and permutes a layer (correctly) and saves “modified” weights
3. tests that outputs are identical

Designed so a coding LLM (or a beginner) can implement it reliably.

---

# 🧾 Project Spec: Tiny MLP Permutation Invariance

## 🎯 Goal

Build a tiny, standard neural network and verify:

> Permuting hidden neurons (with correct downstream adjustment) preserves outputs.

---

# 📦 Tech choices

* Language: **Python 3.10+**
* Library: **PyTorch**
* Package manager: **uv** (fast, reproducible, beginner-friendly)

---

# 📁 Project structure

```text
tiny-mlp-permute/
├── pyproject.toml
├── README.md
├── model.py
├── create_master.py
├── permute_model.py
└── test_compare.py
```

---

# ⚙️ Dependency setup (uv)

## pyproject.toml

```toml
[project]
name = "tiny-mlp-permute"
version = "0.1.0"
dependencies = [
    "torch"
]

[tool.uv]
```

---

## Setup commands

```bash
uv venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
uv pip install -e .
```

---

# 🧠 Model definition (model.py)

## Requirements

* 8 linear layers
* ReLU between layers (except last)
* Deterministic init support
* Easy access to layers

## Spec

```python
class TinyMLP(nn.Module):
    layers: nn.ModuleList of 8 Linear layers

forward(x):
    for all layers except last:
        x = relu(layer(x))
    x = last_layer(x)
    return x
```

## Dimensions

* Input: 16
* Hidden: 32
* Output: 8

---

# 🧪 Program 1: create_master.py

## Purpose

Create deterministic base model and save weights.

---

## Requirements

* Set seed for reproducibility
* Instantiate model
* Save weights to `master.pt`

---

## CLI

```bash
python create_master.py
```

---

## Spec

```python
torch.manual_seed(0)

model = TinyMLP()

torch.save(model.state_dict(), "master.pt")
print("Saved master.pt")
```

---

# 🔁 Program 2: permute_model.py

## Purpose

Load master weights, permute a chosen layer, save result.

---

## Key rule (CRITICAL)

For layer `i`:

* Permute rows of `W_i`
* Permute `b_i`
* Permute columns of `W_{i+1}`

---

## Constraints

* Only allow permutation of layers 0–6 (since last layer has no downstream)
* Use a random permutation with fixed seed

---

## CLI

```bash
python permute_model.py --layer 2
```

---

## Spec

### Steps

1. Load model
2. Load `master.pt`
3. Choose layer `i`
4. Create permutation:

```python
perm = torch.randperm(out_features)
```

5. Apply:

```python
# current layer
W_i = layer.weight.data
b_i = layer.bias.data

layer.weight.data = W_i[perm, :]
layer.bias.data = b_i[perm]

# next layer
W_next = next_layer.weight.data
next_layer.weight.data = W_next[:, perm]
```

6. Save as:

```text
modified_layer_{i}.pt
```

---

## Output

```text
Saved modified_layer_2.pt
```

---

# 🧪 Program 3: test_compare.py

## Purpose

Verify both models produce identical outputs.

---

## CLI

```bash
python test_compare.py --layer 2
```

---

## Spec

### Steps

1. Load master model
2. Load modified model
3. Set both to eval mode
4. Disable grad

```python
model.eval()
torch.set_grad_enabled(False)
```

---

### Test loop

Run multiple random inputs:

```python
for i in range(100):
    x = torch.randn(10, 16)

    y1 = master(x)
    y2 = modified(x)

    if not torch.allclose(y1, y2, atol=1e-6):
        print("FAIL")
        exit(1)

print("PASS")
```

---

## Expected output

```text
PASS
```

---

# ⚠️ Important rules (must follow)

### 1. Always deepcopy model before modifying

```python
import copy
model2 = copy.deepcopy(model1)
```

---

### 2. Never permute last layer

No downstream layer to fix → breaks invariance

---

### 3. Always permute BOTH:

* current layer rows
* next layer columns

---

### 4. Use `.data` or `with torch.no_grad()`

Avoid autograd issues.

---

# 🧪 Optional debug (recommended)

Add in test:

```python
diff = (y1 - y2).abs().max()
print("max diff:", diff.item())
```

Expected:

```text
~1e-7 or 0
```

---

# 🧾 README.md (brief)

## Usage

```bash
uv venv
source .venv/bin/activate
uv pip install -e .

python create_master.py
python permute_model.py --layer 2
python test_compare.py --layer 2
```

---

# ✅ Acceptance criteria

Project is correct if:

* `master.pt` is created
* `modified_layer_X.pt` is created
* `test_compare.py` prints `PASS`
* Max difference ≈ 0

---

# 🔥 Nice property of this setup

* Fully deterministic
* No external data
* Tiny (<50KB weights)
* Demonstrates real NN symmetry
* Easy to extend later to transformers

