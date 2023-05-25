import torch
from torchvision import transforms
import imageio
from pathlib import Path

from GMA.core.utils import flow_viz
from flow import visualize_flow_with_images


def visualize_flow(flow_video_up, image, image_new, image_prev, segment, stepi, output_dir):
    """
    Visualize optical flow between two images and save the results as image files.

    Args:
        flow_video_up (torch.Tensor): Upsampled optical flow tensor.
        image (torch.Tensor): Original image tensor.
        image_new (torch.Tensor): New (second) image tensor.
        image_prev (torch.Tensor): Previous (first) image tensor.
        segment (torch.Tensor): Segmentation mask tensor.
        stepi (int): Index of the current step in the frame sequence.

    Returns:
        None. The function saves multiple visualization images to the disk.
    """
    flow_image = transforms.ToTensor()(flow_viz.flow_to_image(flow_video_up))
    image_small_dims = image.shape[-2], image.shape[-1]
    flow_image_small = transforms.Resize(image_small_dims)(flow_image)
    segmentation_mask = segment[0, 0, -1, :, :].to(torch.bool).unsqueeze(0).repeat(3, 1, 1).cpu().detach()
    flow_image_segmented = flow_image_small.mul(segmentation_mask)
    image_prev_reformatted: torch.Tensor = image_prev.to(torch.uint8)[0]
    image_new_reformatted: torch.Tensor = image_new.to(torch.uint8)[0]

    flow_illustration = visualize_flow_with_images(image_prev_reformatted, image_new_reformatted, flow_video_up)
    transform = transforms.ToPILImage()
    image_pure_flow_segmented = transform(flow_image_segmented)
    image_new_pil = transform(image_new[0] / 255.0)
    image_old_pil = transform(image_prev[0] / 255.0)

    # Define output file paths
    prev_image_path = output_dir / Path('gt_img_' + str(stepi) + '_' + str(stepi + 1) + '_1.png')
    new_image_path = output_dir / Path('gt_img_' + str(stepi) + '_' + str(stepi + 1) + '_2.png')
    flow_segm_path = output_dir / Path('flow_segmented_' + str(stepi) + '_' + str(stepi + 1) + '.png')
    flow_image_path = output_dir / Path('flow_' + str(stepi) + '_' + str(stepi + 1) + '.png')

    # Save the images to disk
    imageio.imwrite(flow_segm_path, image_pure_flow_segmented)
    imageio.imwrite(new_image_path, image_new_pil)
    imageio.imwrite(prev_image_path, image_old_pil)
    imageio.imwrite(flow_image_path, flow_illustration)
