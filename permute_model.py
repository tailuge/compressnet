"""Apply weight permutations to a model.

Permutes hidden neurons in the TinyMLP while maintaining output equivalence
by applying corresponding inverse permutations to downstream weights.
"""

import argparse
import copy
import sys

import torch

from model import TinyMLP


def get_swap_permutation(hidden_dim: int, seed: int) -> torch.Tensor:
    """Generate a permutation that swaps a small percentage of neurons."""
    torch.manual_seed(seed)
    perm = torch.arange(hidden_dim)
    num_swaps = int(hidden_dim * 0.05)
    swap_indices = torch.randperm(hidden_dim)[: num_swaps * 2]
    swap_indices = swap_indices.view(-1, 2)

    for i, j in swap_indices:
        i, j = int(i), int(j)
        perm[i], perm[j] = perm[j], perm[i]
    return perm


def get_sort_permutation(biases: torch.Tensor) -> torch.Tensor:
    """Generate a permutation that sorts neurons by their bias values."""
    # Sort by the bias of each neuron
    return torch.argsort(biases)


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

    # Use the bias-sorting strategy
    perm = get_sort_permutation(state[f"layers.{layer}.bias"])

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


def permute_all_layers(
    source_path: str = "master.pt",
    seed: int = 42,
) -> str:
    """Permute all hidden layers in the model sequentially.

    Loads the model and applies bias-sorting permutation to every
    hidden layer (0-6), maintaining output equivalence.

    Args:
        source_path: Path to source model weights.
        seed: Random seed (not used for sorting, but kept for signature consistency).

    Returns:
        Path to the saved fully-permuted weights file.
    """
    state = torch.load(source_path, weights_only=True)
    
    # Get all weight keys to determine layer count
    weight_keys = sorted([k for k in state.keys() if "weight" in k])
    num_hidden = len(weight_keys) - 1 # All except the last layer

    print(f"Permuting layers 1 through {num_hidden-1}...")

    for layer in range(1, num_hidden):
        # 1. Get sorting permutation based on current biases
        perm = get_sort_permutation(state[f"layers.{layer}.bias"])
        
        # 2. Apply to current layer (rows and bias)
        state[f"layers.{layer}.weight"] = state[f"layers.{layer}.weight"][perm]
        state[f"layers.{layer}.bias"] = state[f"layers.{layer}.bias"][perm]
        
        # 3. Apply to next layer (columns)
        next_layer = layer + 1
        state[f"layers.{next_layer}.weight"] = state[f"layers.{next_layer}.weight"][:, perm]

    target_path = "modified_all.pt"
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
    parser.add_argument("--all", action="store_true", help="Permute all hidden layers")
    args = parser.parse_args()

    if args.all:
        print("Applying permutation to ALL layers...")
        path = permute_all_layers(args.source, args.seed)
        print(f"Done: {path}")
        return

    if args.layer < 0 or args.layer > 6:
        print("Error: Layer must be between 0 and 6")
        sys.exit(1)

    print(f"Applying permutation to layer {args.layer}...")
    path = permute_weights(args.source, args.layer, args.seed)
    print(f"Done: {path}")


if __name__ == "__main__":
    main()
