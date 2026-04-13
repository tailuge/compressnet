"""Publish results by plotting weight comparisons.

Caveman style: Me load weights. Me draw small graphs for all layers. Me write clean MD.
"""

import argparse
import matplotlib.pyplot as plt
import torch
import os

def publish(master_path, mutant_path):
    print(f"Loading {master_path} and {mutant_path}...")
    master_state = torch.load(master_path, weights_only=True)
    mutant_state = torch.load(mutant_path, weights_only=True)

    # Find all weight keys
    weight_keys = [k for k in master_state.keys() if "weight" in k]
    weight_keys.sort()
    
    png_paths = []
    
    for i, key in enumerate(weight_keys):
        # 1. Get weights and flatten
        w_master = master_state[key].flatten().numpy()
        w_mutant = mutant_state[key].flatten().numpy()

        # 2. Draw small graph
        plt.figure(figsize=(10, 1.0)) # Further reduced height
        plt.plot(w_master, label="Master", alpha=0.8, linewidth=0.8)
        plt.plot(w_mutant, label="Mutant", alpha=0.8, linewidth=0.8, linestyle="--")
        plt.title(f"Layer {i}: {key}", fontsize=8, pad=2)
        plt.legend(fontsize=6, loc='upper right', frameon=False)
        plt.tick_params(axis='both', which='major', labelsize=6)
        plt.grid(True, alpha=0.2)
        plt.tight_layout(pad=0.2)
        
        png_path = f"results_layer_{i}.png"
        plt.savefig(png_path)
        plt.close()
        png_paths.append(png_path)
        print(f"Saved {png_path}")

    # 3. Write MD - minimal text
    md_path = "results.md"
    with open(md_path, "w") as f:
        f.write("# Results\n\n")
        f.write(f"Compare `{master_path}` vs `{mutant_path}`\n\n")
        for png in png_paths:
            f.write(f"![{png}]({png})\n\n")
    
    print(f"Updated {md_path}")

def main():
    parser = argparse.ArgumentParser(description="Publish all weight graphs")
    parser.add_argument("--master", default="master.pt", help="Master weights")
    parser.add_argument("--mutant", required=True, help="Mutant weights")
    # Removed --layer since we do all now
    args = parser.parse_args()

    publish(args.master, args.mutant)

if __name__ == "__main__":
    main()
