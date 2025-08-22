import random

import torch
import shutil
import numpy as np
from pathlib import Path
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor

# Configuration
ONBOARD_DIR = "/mnt/personal/jelint19/results/sam2_wide_baseline/hammer/onboarding"
TEST_DIR = "/mnt/personal/jelint19/results/sam2_wide_baseline/hammer/test"
OUTPUT_DIR = "/mnt/personal/jelint19/results/sam2_wide_baseline/hammer/output"
INIT_SEGMENT_PATH = "/mnt/personal/jelint19/results/sam2_wide_baseline/hammer/init_segment.png"
CHECKPOINT = "/mnt/personal/jelint19/weights/SegmentAnything2/sam2.1_hiera_large.pt"
MODEL_CFG = 'configs/sam2.1/sam2.1_hiera_l.yaml'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# Load images from directory
image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']

test_imgs = sorted(Path(TEST_DIR).iterdir())
test_imgs = random.sample(test_imgs, len(test_imgs))

image_paths = sorted(Path(ONBOARD_DIR).iterdir()) + test_imgs[:1]

if Path(OUTPUT_DIR).exists():
    shutil.rmtree(OUTPUT_DIR)
Path(OUTPUT_DIR).mkdir(exist_ok=True, parents=True)

print(f"Found {len(image_paths)} images")

# Load images
images = []
for path in image_paths:
    img = Image.open(path).convert('RGB')
    images.append(img)
    print(f"Loaded: {path}")

# Create initial mask (example: create a simple mask in the center)
# Replace this with your actual initial segmentation mask
img_width, img_height = images[0].size

init_mask_img = Image.open(INIT_SEGMENT_PATH).convert('L')

# Setup paths
out_path = Path(OUTPUT_DIR)
out_path.mkdir(exist_ok=True, parents=True)

tmp_path = out_path / 'tmp_imgs'
tmp_path.mkdir(exist_ok=True, parents=True)

# Save images as indexed PNGs
print("Saving images to temporary directory...")
for i, img in enumerate(images):
    img.save(tmp_path / f'{i:06d}.jpg')

# Initialize SAM2
print("Initializing SAM2...")
predictor = build_sam2_video_predictor(MODEL_CFG, str(CHECKPOINT), device=DEVICE)

# Initialize state
print("Initializing video state...")
state = predictor.init_state(
    str(tmp_path),
    offload_video_to_cpu=True,
    offload_state_to_cpu=True,
)

# Convert initial mask to SAM format
# Note: You may need to adapt this based on your specific mask format
initial_mask_sam_format = np.asarray(init_mask_img).squeeze().astype('bool')

# Add initial mask
print("Adding initial mask...")
out_frame_idx, out_obj_ids, out_mask_logits = predictor.add_new_mask(
    state, 0, 0, initial_mask_sam_format
)


def overlay_mask(image, mask, color=(255, 0, 0), alpha=0.5):
    overlay = image.copy()
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    color_img = Image.new("RGB", image.size, color)
    overlay.paste(color_img, mask=mask_img)
    return Image.blend(image, overlay, alpha)


# Propagate through video
print("Propagating segmentation through video...")
for i, (_, out_obj_ids, out_mask_logits) in enumerate(predictor.propagate_in_video(state)):
    frame_idx = i
    if frame_idx < len(images):
        for proposal_idx in range(out_mask_logits.shape[0]):
            mask = (out_mask_logits[proposal_idx] > 0).cpu().numpy().astype(bool).squeeze()
            seg_img = overlay_mask(images[frame_idx], mask)
            seg_img.save(out_path / f"{frame_idx:06d}_proposal_{proposal_idx}.png")
            print(f"Saved frame {frame_idx} segmentation")

# Cleanup
shutil.rmtree(tmp_path)
print(f"Segmentation complete! Results saved to {OUTPUT_DIR}")
