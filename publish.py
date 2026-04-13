"""Publish results by plotting weight comparisons.

Caveman style: Me load weights. Me draw graph. Me save PNG. Me write MD.
"""

import argparse
import matplotlib.pyplot as plt
import torch
import os

def publish(master_path, mutant_path, layer_index):
    # 1. Load weights
    print(f"Loading {master_path} and {mutant_path}...")
    master_state = torch.load(master_path, weights_only=True)
    mutant_state = torch.load(mutant_path, weights_only=True)

    weight_key = f"layers.{layer_index}.weight"
    
    if weight_key not in master_state:
        print(f"Error: {weight_key} not found in master")
        return
    if weight_key not in mutant_state:
        print(f"Error: {weight_key} not found in mutant")
        return

    # 2. Get weights and flatten
    w_master = master_state[weight_key].flatten().numpy()
    w_mutant = mutant_state[weight_key].flatten().numpy()

    # 3. Draw graph
    plt.figure(figsize=(10, 6))
    plt.plot(w_master, label="Master weights", alpha=0.7, linewidth=1)
    plt.plot(w_mutant, label="Mutant weights", alpha=0.7, linewidth=1, linestyle="--")
    plt.title(f"Weight Comparison: {weight_key}")
    plt.xlabel("Weight Index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    png_path = "results.png"
    plt.savefig(png_path)
    print(f"Saved {png_path}")

    # 4. Write MD
    md_path = "results.md"
    with open(md_path, "w") as f:
        f.write("# Results\n\n")
        f.write(f"Weight comparison for layer {layer_index}.\\n\n")
        f.write(f"- **Master**: `{master_path}`\n")
        f.write(f"- **Mutant**: `{mutant_path}`\n\n")
        f.write(f"![Weight Comparison]({png_path})\n")
    
    print(f"Updated {md_path}")

def main():
    parser = argparse.ArgumentParser(description="Publish weight graphs")
    parser.add_argument("--master", default="master.pt", help="Master weights")
    parser.add_argument("--mutant", required=True, help="Mutant weights")
    parser.add_argument("--layer", type=int, default=0, help="Layer index")
    args = parser.parse_args()

    publish(args.master, args.mutant, args.layer)

if __name__ == "__main__":
    main()
