"""Compare outputs of master and mutant models.

Loads both models and verifies they produce identical outputs by
computing the maximum difference between their predictions.
"""

import argparse
import sys

import torch

from model import TinyMLP


def compare_models(
    master_path: str = "master.pt",
    mutant_path: str = "modified_layer_0.pt",
    tolerance: float = 1e-6,
) -> float:
    """Compare model outputs and return the maximum difference.

    Loads both models, runs them on the same random input, and computes
    the maximum absolute difference between their outputs.

    Args:
        master_path: Path to the master model weights.
        mutant_path: Path to the mutant model weights.
        tolerance: Maximum allowed difference for models to be considered identical.

    Returns:
        The maximum absolute difference between model outputs.
    """
    # Load models
    master_state = torch.load(master_path, weights_only=True)
    mutant_state = torch.load(mutant_path, weights_only=True)

    master_model = TinyMLP()
    mutant_model = TinyMLP()

    master_model.load_state_dict(master_state)
    mutant_model.load_state_dict(mutant_state)

    # Set to eval mode
    master_model.eval()
    mutant_model.eval()

    # Generate test input
    torch.manual_seed(123)
    test_input = torch.randn(1, master_model.layers[0].in_features)

    # Run forward passes
    with torch.no_grad():
        master_output = master_model(test_input)
        mutant_output = mutant_model(test_input)

    # Compute max difference
    max_diff = (master_output - mutant_output).abs().max().item()

    return max_diff


def main() -> None:
    """Main entry point for model comparison."""
    parser = argparse.ArgumentParser(description="Compare model outputs")
    parser.add_argument("--master", default="master.pt", help="Path to master model weights")
    parser.add_argument("--mutant", default="modified_layer_0.pt", help="Path to mutant model weights")
    parser.add_argument(
        "--tolerance", type=float, default=1e-6, help="Maximum allowed difference"
    )
    args = parser.parse_args()

    print(f"Loading master: {args.master}")
    print(f"Loading mutant: {args.mutant}")

    max_diff = compare_models(args.master, args.mutant, args.tolerance)

    print(f"Max difference: {max_diff:.2e}")

    if max_diff < args.tolerance:
        print("**PASS**: Models are identical")
    else:
        print("**FAIL**: Models differ beyond tolerance")
        sys.exit(1)


if __name__ == "__main__":
    main()
