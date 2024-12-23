import numpy as np
import torch
from matplotlib import pyplot as plt

from data_structures.keyframe_buffer import FlowObservation, SyntheticFlowObservation


def visualize_optical_flow_errors(template_image: torch.Tensor, target_image: torch.Tensor,
                                  observed_flow: FlowObservation, synthetic_flow: SyntheticFlowObservation):
    # Ensure images are in CPU and convert to numpy
    template_image_np = template_image.squeeze().cpu().permute(1, 2, 0).numpy()
    target_image_np = target_image.squeeze().cpu().permute(1, 2, 0).numpy()

    # Ensure flows are in CPU and convert to numpy
    observed_flow_np = observed_flow.observed_flow.squeeze().cpu().permute(1, 2, 0).numpy()
    synthetic_flow_np = synthetic_flow.observed_flow.squeeze().cpu().permute(1, 2, 0).numpy()

    # Get the image dimensions
    height, width, _ = template_image_np.shape
    step = max(width, height) // 20

    # Create grid points
    x = np.arange(0, width, step)
    y = np.arange(0, height, step)
    xv, yv = np.meshgrid(x, y)

    # Get flow vectors at the grid points
    observed_u = observed_flow_np[yv, xv, 0]
    observed_v = observed_flow_np[yv, xv, 1]
    synthetic_u = synthetic_flow_np[yv, xv, 0]
    synthetic_v = synthetic_flow_np[yv, xv, 1]

    # Calculate the end points of the flow vectors for observed and synthetic
    observed_end_x = xv + observed_u
    observed_end_y = yv + observed_v
    synthetic_end_x = xv + synthetic_u
    synthetic_end_y = yv + synthetic_v

    # Calculate the error vectors
    error_u = synthetic_u - observed_u
    error_v = synthetic_v - observed_v

    # Setup the colormap
    norm = np.sqrt(error_u ** 2 + error_v ** 2)
    # Handle NaN values
    norm = np.nan_to_num(norm, nan=0.0)
    if norm.max() != 0:
        norm = (norm - norm.min()) / (norm.max() - norm.min())  # Normalize to [0, 1]
    else:
        norm = np.zeros_like(norm)
    colors = plt.cm.jet(norm)
    colors = colors.reshape(-1, colors.shape[-1])

    # Create the figure
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Plot source image with error vectors
    ax[0].imshow(template_image_np)
    ax[0].quiver(xv, yv, error_u, error_v, color=colors, scale=1, scale_units='xy')
    ax[0].grid(visible=False)
    ax[0].set_title('Source Image with Error Vectors')

    # Plot target image with connected flow vectors
    ax[1].imshow(target_image_np)
    ax[1].quiver(xv, yv, synthetic_end_x - xv, synthetic_end_y - yv, color='r', scale=1, scale_units='xy',
                 label='Synthetic Flow')
    ax[1].quiver(xv, yv, observed_end_x - xv, observed_end_y - yv, color='b', scale=1, scale_units='xy',
                 label='Observed Flow')
    ax[1].grid(visible=False)
    ax[1].legend(loc='upper right')
    ax[1].set_title('Target Image with Flow Vectors')

    fig.tight_layout()

    return fig
