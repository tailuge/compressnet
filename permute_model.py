"""Apply weight permutations to a model.

Permutes hidden neurons in the TinyMLP while maintaining output equivalence
by applying corresponding inverse permutations to downstream weights.
"""

import argparse
import copy
import sys

import torch

from model import TinyMLP


def permute_weights(
    source_path: str = "master.pt",
    layer: int = 0,
    seed: int = 42,
) -> str:
    """Permute weights at a specific hidden layer.

    Applies a random permutation to the neurons in the specified layer
    and adjusts downstream weights to maintain output equivalence.

    Args:
        source_path: Path to source model weights.
        layer: Index of the hidden layer to permute (0-6).
        seed: Random seed for generating the permutation.

    Returns:
        Path to the saved permuted weights file.

    Raises:
        ValueError: If layer index is out of range.
    """
    state = torch.load(source_path, weights_only=True)

    # Determine hidden dimension from state dict
    hidden_dim = state[f"layers.{layer}.weight"].shape[0]

    # Generate custom swap permutation
    torch.manual_seed(seed)
    perm = torch.arange(hidden_dim)
    num_swaps = int(hidden_dim * 0.05)
    swap_indices = torch.randperm(hidden_dim)[:num_swaps * 2]
    swap_indices = swap_indices.view(-1, 2)

    for i, j in swap_indices:
        i, j = int(i), int(j)
        perm[i], perm[j] = perm[j], perm[i]

    # Apply permutation to layer weights (rows)
    state[f"layers.{layer}.weight"] = state[f"layers.{layer}.weight"][perm]

    # Apply inverse permutation to layer bias
    state[f"layers.{layer}.bias"] = state[f"layers.{layer}.bias"][perm]

    # Apply permutation to next layer weights (columns) if not the last hidden layer
    next_layer = layer + 1
    if next_layer < len([k for k in state if "weight" in k]):
        state[f"layers.{next_layer}.weight"] = state[f"layers.{next_layer}.weight"][:, perm]

    # Save permuted weights
    target_path = f"modified_layer_{layer}.pt"
    torch.save(state, target_path)
    print(f"Saved {target_path}")

    return target_path


def main() -> None:
    """Main entry point for weight permutation."""
    parser = argparse.ArgumentParser(description="Permute model weights")
    parser.add_argument("--source", default="master.pt", help="Path to source model weights")
    parser.add_argument(
        "--layer", type=int, default=0, help="Hidden layer index to permute (0-6)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for permutation")
    args = parser.parse_args()

    if args.layer < 0 or args.layer > 6:
        print("Error: Layer must be between 0 and 6")
        sys.exit(1)

    print(f"Applying permutation to layer {args.layer}...")
    path = permute_weights(args.source, args.layer, args.seed)
    print(f"Done: {path}")


if __name__ == "__main__":
    main()
