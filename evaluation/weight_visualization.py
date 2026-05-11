import argparse
import re

import matplotlib.pyplot as plt
import torch

depth_re = re.compile(r".*\.(\d+)\.(.*)")


def percentile(data, p):
    """Calculate the p-th percentile of the data.

    Args:
      data: A list of data.
      p: The percentile to compute.

    Returns:
      The pth percentile of the data.

    """
    return sorted(data)[int(len(data) * p)]


def visualize_weight_distribution(state_dict):
    """Visualizes the weight distribution of a neural network given a torch state dict.

    Args:
      state_dict: A dictionary containing the state of the model.

    """
    all_weights = []
    depth_weights = {}
    layer_type_weights = {}

    for name, param in state_dict.items():
        weights = param.cpu().detach().numpy().flatten()
        all_weights.extend(weights)

        # Extract depth from layer name (assuming sequential naming convention)
        match = depth_re.match(name)
        depth = int(match.group(1)) if match else -1
        layer_type = match.group(2) if match else name

        depth_weights.setdefault(depth, []).extend(weights)
        layer_type_weights.setdefault(layer_type, []).extend(weights)
    # Determine overall min and max for consistent binning
    overall_min = percentile(all_weights, 0.1)  # min(all_weights)
    overall_max = percentile(all_weights, 0.9)  # max(all_weights)

    # Overall distribution
    plt.figure(figsize=(12, 4))
    plt.hist(all_weights, bins=1000, range=(overall_min, overall_max))
    plt.title("Overall Weight Distribution")
    plt.xlabel("Weight Value")
    plt.ylabel("Frequency")
    plt.show()

    # Distribution by depth
    fig, axs = plt.subplots(1, len(depth_weights), figsize=(12, 4))
    for ax, (depth, weights) in zip(axs, depth_weights.items()):
        min_val = percentile(weights, 0.1)  # min(weights)
        max_val = percentile(weights, 0.9)
        ax.hist(weights, bins=1000, range=(min_val, max_val), label=f"Depth {depth}")
        ax.set_title(f"Depth {depth}")
    fig.suptitle("Weight Distribution by Depth")
    plt.show()

    # Distribution by layer type
    fig, axs = plt.subplots(1, len(layer_type_weights), figsize=(12, 4))
    for ax, (layer_type, weights) in zip(axs, layer_type_weights.items()):
        min_val = percentile(weights, 0.1)  # min(weights)
        max_val = percentile(weights, 0.9)
        ax.hist(weights, bins=1000, range=(min_val, max_val), label=layer_type)
        ax.set_title(layer_type)

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize weight distribution of a neural network")
    parser.add_argument("-m", "--model", type=str, help="Path to the model checkpoint tar", required=True)
    args = parser.parse_args()

    save_state = torch.load(args.model, map_location=torch.device("cpu"))
    state_dict = save_state["model_state"]

    visualize_weight_distribution(state_dict)
