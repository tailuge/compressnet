"""Publish results by plotting weight and bias comparisons.

Caveman style: Me load weights. Me load bias. Me repeat bias to match weights. Me draw two-axis graph.
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
        bias_key = key.replace("weight", "bias")
        
        # 1. Get weights and biases
        w_master = master_state[key]
        w_mutant = mutant_state[key]
        b_master = master_state[bias_key]
        b_mutant = mutant_state[bias_key]

        # 2. Align bias with weights (repeat each bias N times where N is input_dim)
        # Weights shape: (out_features, in_features)
        n_inputs = w_master.shape[1]
        b_master_stepped = b_master.repeat_interleave(n_inputs).numpy()
        b_mutant_stepped = b_mutant.repeat_interleave(n_inputs).numpy()
        
        w_master_flat = w_master.flatten().numpy()
        w_mutant_flat = w_mutant.flatten().numpy()

        # 3. Draw dual-axis graph
        fig, ax1 = plt.subplots(figsize=(10, 1.2))
        
        # Weights on Primary Axis
        ax1.plot(w_master_flat, label="W Master", alpha=0.6, linewidth=0.7, color='tab:blue')
        ax1.plot(w_mutant_flat, label="W Mutant", alpha=0.6, linewidth=0.7, color='tab:blue', linestyle="--")
        ax1.set_ylabel("Weights", fontsize=6, color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=5)
        ax1.tick_params(axis='x', labelsize=6)
        
        # Biases on Secondary Axis
        ax2 = ax1.twinx()
        ax2.plot(b_master_stepped, label="B Master", alpha=0.9, linewidth=1.0, color='tab:red')
        ax2.plot(b_mutant_stepped, label="B Mutant", alpha=0.9, linewidth=1.0, color='tab:red', linestyle=":")
        ax2.set_ylabel("Bias", fontsize=6, color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red', labelsize=5)
        
        plt.title(f"Layer {i}: {key}", fontsize=8, pad=2)
        
        # Combined Legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=5, loc='upper right', frameon=False, ncol=2)
        
        ax1.grid(True, alpha=0.1)
        plt.tight_layout(pad=0.2)
        
        png_path = f"results_layer_{i}.png"
        plt.savefig(png_path)
        plt.close()
        png_paths.append(png_path)
        print(f"Saved {png_path}")

    # 4. Write MD - minimal text
    md_path = "results.md"
    with open(md_path, "w") as f:
        f.write("# Results\n\n")
        f.write(f"Compare `{master_path}` vs `{mutant_path}` (Sorted by Bias)\n\n")
        for png in png_paths:
            f.write(f"![{png}]({png})\n\n")
    
    print(f"Updated {md_path}")

def main():
    parser = argparse.ArgumentParser(description="Publish all weight and bias graphs")
    parser.add_argument("--master", default="master.pt", help="Master weights")
    parser.add_argument("--mutant", required=True, help="Mutant weights")
    args = parser.parse_args()

    publish(args.master, args.mutant)

if __name__ == "__main__":
    main()
