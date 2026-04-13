"""Create and save a deterministic master model.

Generates a TinyMLP model with fixed random seeds for reproducibility
and saves the weights to master.pt.
"""

import torch

from model import TinyMLP


MASTER_PATH = "master.pt"


def create_master() -> str:
    """Create and save a deterministic master model.

    Sets random seeds for reproducibility, initializes the TinyMLP
    architecture, and saves the state dictionary to master.pt.

    Returns:
        Path to the saved master weights file.
    """
    # Set seeds for deterministic reproducibility
    torch.manual_seed(42)

    # Initialize model architecture
    model = TinyMLP()

    # Save weights
    torch.save(model.state_dict(), MASTER_PATH)
    print(f"Master weights saved to {MASTER_PATH}")

    return MASTER_PATH


def main() -> None:
    """Main entry point for master model creation."""
    print("Creating master model...")
    path = create_master()
    print(f"Done: {path}")


if __name__ == "__main__":
    main()
